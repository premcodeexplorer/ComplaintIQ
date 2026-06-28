"""Agent 5 -- SLA Monitor.

Two responsibilities:
  1. Compute the SLA *due date* for a complaint (from `data/sla_rules.json`),
     adjusted by severity multiplier.
  2. Predict the *probability of SLA breach* using the trained Random Forest.

The training script lives in `models/train_sla_model.py`; this agent only loads
the joblib artefact and applies it. If the artefact is missing it falls back to
a rule-based probability so the pipeline still runs.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SLA_RULES_PATH = ROOT / "data" / "sla_rules.json"
MODEL_PATH = ROOT / "models" / "sla_rf.joblib"

_rules: dict[str, Any] | None = None
_artefact: dict[str, Any] | None = None


def _rules_cache() -> dict[str, Any]:
    global _rules
    if _rules is None:
        _rules = json.loads(SLA_RULES_PATH.read_text(encoding="utf-8"))
    return _rules


def _artefact_cache() -> dict[str, Any] | None:
    global _artefact
    if _artefact is None and MODEL_PATH.exists():
        _artefact = joblib.load(MODEL_PATH)
    return _artefact


def compute_due_date(complaint: dict[str, Any]) -> tuple[str, int]:
    """Return (iso_due_date, effective_sla_days)."""
    rules = _rules_cache()
    category = (complaint.get("category") or "General")
    base = rules["sla_days"].get(category, rules["sla_days"]["General"])
    mult = rules["severity_multiplier"].get(complaint.get("severity") or "Medium", 1.0)
    eff = max(1, round(base * mult))
    start = pd.to_datetime(complaint.get("date") or datetime.utcnow().date())
    due = (start + timedelta(days=eff)).date().isoformat()
    return due, eff


def _row_for_model(complaint: dict[str, Any]) -> pd.DataFrame:
    """Build the feature vector expected by the trained SLA model.

    Mirrors `models.train_sla_model.engineer_features` for a single row so the
    pipeline can score live submissions.
    """
    art = _artefact_cache()
    assert art is not None
    helpers = art["feature_helpers"]
    fraud_keywords = helpers["fraud_keywords"]
    severity_order = helpers.get("severity_order",
                                  {"Critical": 4, "High": 3, "Medium": 2, "Low": 1})
    sentiment_order = helpers.get("sentiment_order",
                                   {"Angry": 4, "Frustrated": 3, "Neutral": 2, "Polite": 1})
    high_vis = set(helpers.get("high_visibility_channels", ["twitter", "whatsapp"]))

    text = (complaint.get("complaint_text") or "")
    text_lower = text.lower()
    amount = float(complaint.get("amount_involved") or 0)
    filed = pd.to_datetime(complaint.get("date"), errors="coerce")
    if pd.notna(filed) and filed.tzinfo is not None:
        filed = filed.tz_localize(None)
    weekday = int(filed.weekday()) if pd.notna(filed) else 0
    hours_since = (pd.Timestamp.utcnow().tz_localize(None) - filed).total_seconds() / 3600.0 \
                  if pd.notna(filed) else 0.0
    customer_count = int(complaint.get("customer_complaint_count") or 0)
    if customer_count == 0:
        try:
            from database import db
            customer_count = len(db.customer_history(complaint.get("customer_name") or ""))
        except Exception:
            customer_count = 1
    is_duplicate = 1 if complaint.get("duplicate_of") else 0

    rules = _rules_cache()
    category = complaint.get("category") or "General"
    severity = complaint.get("severity") or "Medium"
    sentiment = complaint.get("sentiment") or "Neutral"
    base_sla = rules["sla_days"].get(category, rules["sla_days"]["General"])
    sev_mult = rules["severity_multiplier"].get(severity, 1.0)
    days_to_sla = max(1, int(round(base_sla * sev_mult)))
    pct_sla = max(0.0, hours_since / (days_to_sla * 24.0))

    row = {
        # categorical
        "channel": complaint.get("channel") or "email",
        "language": complaint.get("language") or "english",
        "account_type": complaint.get("account_type") or "savings",
        "category": category,
        "severity": severity,
        "sentiment": sentiment,
        # numeric
        "amount_involved": amount,
        "complaint_text_length": len(text),
        "complaint_word_count": len(text.split()),
        "hours_since_filed": float(hours_since),
        "day_of_week": weekday,
        "is_weekend_filed": int(weekday >= 5),
        "is_high_amount": int(amount >= 25000),
        "is_high_value": int(amount > 50000),
        "is_fraud_keyword": int(any(k in text_lower for k in fraud_keywords)),
        "is_duplicate": is_duplicate,
        "is_repeat_customer": int(customer_count > 1),
        "has_amount": int(amount > 0),
        "channel_risk": int((complaint.get("channel") or "") in high_vis),
        "sentiment_score": int(sentiment_order.get(sentiment, 2)),
        "severity_score": int(severity_order.get(severity, 2)),
        "customer_complaint_count": max(1, customer_count),
        "days_to_sla": days_to_sla,
        "pct_sla_elapsed": float(pct_sla),
    }
    return pd.DataFrame([row])


def predict_breach(complaint: dict[str, Any]) -> dict[str, Any]:
    """Return {sla_due_date, sla_days, breach_probability, model_used}."""
    due, eff = compute_due_date(complaint)
    art = _artefact_cache()
    if art is None:
        prob = _rule_based_prob(complaint)
        return {
            "sla_due_date": due, "sla_days": eff,
            "breach_probability": round(prob, 3),
            "model_used": "rule-based-fallback",
        }
    X = _row_for_model(complaint)[art["categorical"] + art["numeric"]]
    prob = float(art["model"].predict_proba(X)[0, 1])
    return {
        "sla_due_date": due, "sla_days": eff,
        "breach_probability": round(prob, 3),
        "model_used": art.get("winner", "random_forest").lower(),
    }


def _rule_based_prob(complaint: dict[str, Any]) -> float:
    p = 0.18
    if complaint.get("severity") == "Critical": p += 0.35
    elif complaint.get("severity") == "High":   p += 0.20
    elif complaint.get("severity") == "Low":    p -= 0.08
    if (complaint.get("amount_involved") or 0) >= 25000: p += 0.10
    return max(0.02, min(0.92, p))


def days_until_due(complaint: dict[str, Any], today: datetime | None = None) -> int:
    today = today or datetime.utcnow()
    due = pd.to_datetime(complaint.get("sla_due_date"))
    return int((due.date() - today.date()).days)


if __name__ == "__main__":
    from database import db
    from agents.classifier import classify

    db.init_db()
    samples = db.list_complaints(limit=5)
    for c in samples:
        c.update(classify(c))
        r = predict_breach(c)
        print(f"{c['id']:10s} {c['category']:10s} {c['severity']:8s} "
              f"due={r['sla_due_date']} ({r['sla_days']}d)  "
              f"P(breach)={r['breach_probability']:.2f}  [{r['model_used']}]")
