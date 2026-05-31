"""Customer risk score broken into three explainable components.

Per the PDF spec, the 0-100 score predicts:
  - Ombudsman Escalation Risk -- formal RBI complaint risk
  - Churn Risk                -- customer leaving the bank
  - Social Media Risk         -- public Twitter / WhatsApp blow-up

Overall = weighted average of the three. Inputs come from the complaint row
and the customer's full history (already in SQLite).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

# Weights for the overall score (must sum to 1.0).
WEIGHTS = {"ombudsman": 0.45, "churn": 0.30, "social": 0.25}


def compute(complaint: dict[str, Any], history: list[dict[str, Any]],
            breach_prob: float) -> dict[str, int]:
    """Return {ombudsman, churn, social, overall} -- all 0..100 ints."""
    ombudsman = _ombudsman_score(complaint, history, breach_prob)
    churn = _churn_score(complaint, history)
    social = _social_score(complaint, history)
    overall = int(round(
        WEIGHTS["ombudsman"] * ombudsman
        + WEIGHTS["churn"] * churn
        + WEIGHTS["social"] * social
    ))
    return {
        "ombudsman": _clip(ombudsman),
        "churn": _clip(churn),
        "social": _clip(social),
        "overall": _clip(overall),
    }


# --- sub-scores --------------------------------------------------------------

def _ombudsman_score(c: dict[str, Any], history: list[dict[str, Any]],
                     breach_prob: float) -> int:
    """RBI Ombudsman escalation: severity, breach probability, complaint count,
    amount stuck, recent angry sentiment."""
    score = int(round(breach_prob * 40))  # 0..40 from RF
    sev = c.get("severity") or "Medium"
    score += {"Critical": 30, "High": 20, "Medium": 10, "Low": 3}.get(sev, 10)
    score += min(len(history) * 3, 15)   # repeat complainer
    amt = c.get("amount_involved") or 0
    if amt >= 100000: score += 12
    elif amt >= 25000: score += 6
    angry_recent = sum(1 for h in history if h.get("sentiment") in ("Angry", "Frustrated"))
    score += min(angry_recent * 2, 8)
    return score


def _churn_score(c: dict[str, Any], history: list[dict[str, Any]]) -> int:
    """Churn risk: repeated unresolved issues, category breadth, frustrated tone."""
    score = 0
    n = len(history)
    score += min(n * 6, 30)              # # of total complaints
    unresolved = sum(1 for h in history if (h.get("status") or "open") == "open")
    score += min(unresolved * 5, 25)
    categories_touched = len({h.get("category") for h in history if h.get("category")})
    score += min(categories_touched * 6, 20)  # systemic dissatisfaction
    if c.get("sentiment") in ("Angry", "Frustrated"):
        score += 12
    if (c.get("severity") or "") in ("Critical", "High"):
        score += 8
    return score


def _social_score(c: dict[str, Any], history: list[dict[str, Any]]) -> int:
    """Social media blow-up risk: Twitter / WhatsApp channels with angry tone."""
    score = 0
    public_channels = {"twitter", "whatsapp"}
    public_complaints = [h for h in history if h.get("channel") in public_channels]
    score += min(len(public_complaints) * 12, 45)
    if (c.get("channel") or "") in public_channels:
        score += 18
    if c.get("sentiment") == "Angry":
        score += 22
    elif c.get("sentiment") == "Frustrated":
        score += 10
    if (c.get("severity") or "") == "Critical":
        score += 8
    return score


def _clip(v: int) -> int:
    return max(0, min(100, int(v)))


# --- bulk recompute helper ---------------------------------------------------

def recompute_all() -> int:
    """Recompute Ombudsman / Churn / Social / overall for every processed row.
    Used after schema migration or after the scoring function changes."""
    from database import db

    db.init_db()
    rows = db.list_complaints(where="processed_at IS NOT NULL")
    by_customer: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_customer.setdefault(r.get("customer_name") or "", []).append(r)

    n = 0
    for r in rows:
        hist = by_customer.get(r.get("customer_name") or "", [])
        scores = compute(r, hist, breach_prob=r.get("sla_breach_prob") or 0.0)
        db.update_complaint(
            r["id"],
            risk_score=scores["overall"],
            risk_ombudsman=scores["ombudsman"],
            risk_churn=scores["churn"],
            risk_social=scores["social"],
        )
        n += 1
    return n


if __name__ == "__main__":
    print("Recomputing risk scores for all processed complaints...")
    n = recompute_all()
    print(f"Updated {n} rows.")
