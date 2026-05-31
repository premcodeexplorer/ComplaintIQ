"""End-to-end orchestration of the 6-agent complaint pipeline.

For each complaint:
  1. Intake          -> structured fields
  2. Classifier      -> category / severity / sentiment
  3. Duplicate Check -> compare against prior complaints from same customer
  4. Response Draft  -> bank reply
  5. SLA Monitor     -> due date + breach probability
  6. (Root Cause)    -> run once at the end across all processed complaints

Skipping rule: if a complaint is detected as a duplicate, we still classify and
record an SLA window, but skip the LLM response draft (the prior complaint
already has one).

Customer Risk Score (0-100): combines breach probability, severity weight,
complaint count for the customer, and recent-angry-sentiment count. Stored on
the complaint row so the dashboard can render it directly.
"""
from __future__ import annotations

# --- Windows DLL hygiene -----------------------------------------------------
# torch (MKL) + tensorflow (used by sentence-transformers' model card) + sklearn
# all bring their own OpenMP runtimes. On Windows whichever loads second sometimes
# fails with `c10.dll` init errors. Set the standard escape hatch BEFORE imports.
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# We only use PyTorch for sentence-transformers; transformers' optional TF/JAX
# integrations trigger a protobuf version clash with tensorflow 2.15 on Windows.
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("USE_FLAX", "0")
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
try:  # preload torch first so its MKL DLL wins over sklearn/numpy's
    import torch  # noqa: F401
except Exception:  # torch is optional at this layer; sentence-transformers will surface real errors
    pass

import json
from datetime import datetime
from typing import Any, Iterable

from agents import classifier, duplicate_detector as dd, intake, response_drafter, sla_monitor
from agents import root_cause, risk_score as risk_module
from agents import ml_category, sentiment_ml, priority as priority_module
from database import db


def process_one_streaming(complaint_id: str, *, draft_response: bool = True,
                          on_step=None) -> dict[str, Any]:
    """Same as `process_one` but invokes `on_step(label, status, payload)` between
    each agent so a UI can show live progress. `status` is one of
    'started'/'done'/'skipped'. `payload` is the agent's structured output (None
    when started)."""
    def emit(label, status="done", payload=None):
        if on_step:
            try:
                on_step(label, status, payload)
            except Exception:
                pass

    row = db.get_complaint(complaint_id)
    if not row:
        raise KeyError(complaint_id)

    emit("Agent 1 (Intake)", "started")
    intake_out = intake.extract(row)
    emit("Agent 1 (Intake)", "done", intake_out)

    emit("Agent 2 (Classifier)", "started")
    cls = classifier.classify(row)
    row.update(cls)
    emit("Agent 2 (Classifier)", "done", cls)

    emit("Agent 3 (Duplicate Detector)", "started")
    dup = dd.find_duplicate(row)
    emit("Agent 3 (Duplicate Detector)", "done", dup)

    if dup["is_duplicate"] or not draft_response:
        draft = None
        emit("Agent 4 (Response Drafter)", "skipped", None)
    else:
        emit("Agent 4 (Response Drafter)", "started")
        draft = response_drafter.draft(row, sla_days=_lookup_sla_days(row))
        emit("Agent 4 (Response Drafter)", "done", {"draft": draft})

    emit("Agent 5 (SLA Monitor)", "started")
    sla = sla_monitor.predict_breach(row)
    row["sla_breach_prob"] = sla["breach_probability"]
    row["duplicate_of"] = dup["duplicate_of"]
    emit("Agent 5 (SLA Monitor)", "done", sla)

    # --- ML second opinions ---------------------------------------------------
    emit("ML Category (TF-IDF + LogReg)", "started")
    ml_cat = ml_category.predict(row.get("complaint_text") or "")
    cat_conf = ml_category.agreement(cls.get("category"),
                                     ml_cat["category"] if ml_cat else None)
    emit("ML Category (TF-IDF + LogReg)", "done",
         {**(ml_cat or {}), "agreement": cat_conf})

    emit("ML Sentiment (HF Roberta)", "started")
    ml_sent = sentiment_ml.predict(row.get("complaint_text") or "")
    sent_conf = sentiment_ml.agreement(cls.get("sentiment"), ml_sent)
    emit("ML Sentiment (HF Roberta)", "done",
         {**(ml_sent or {}), "agreement": sent_conf})

    # --- composite priority + risk -------------------------------------------
    history = db.customer_history(row.get("customer_name") or "")
    risk = risk_module.compute(row, history, breach_prob=sla["breach_probability"])

    emit("ML Priority (Gradient Boosting)", "started")
    priority = priority_module.score({**row, "sla_breach_prob": sla["breach_probability"]})
    emit("ML Priority (Gradient Boosting)", "done", {"priority_score": priority})

    # --- Auto-resolution rule (FIX 5) ---------------------------------------
    # Low/Medium severity + Polite sentiment + category has a standard template
    # => mark as Auto-Resolved (Standard Reply Sent). Duplicates flagged by
    # Agent 3 get a separate auto-resolved status.
    auto_resolved_at: str | None = None
    if dup["is_duplicate"]:
        status_value = "auto_resolved_dup"
        auto_resolved_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    elif (cls["severity"] in ("Low", "Medium")
          and cls["sentiment"] == "Polite"
          and cls["category"] in ("UPI", "ATM", "Card", "NetBanking", "Loan", "General")):
        status_value = "auto_resolved_std"
        auto_resolved_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    else:
        status_value = "open"

    db.update_complaint(
        complaint_id,
        category=cls["category"], severity=cls["severity"], sentiment=cls["sentiment"],
        intake_json=json.dumps(intake_out, ensure_ascii=False),
        duplicate_of=dup["duplicate_of"], similarity=dup["similarity"],
        draft_response=draft,
        sla_due_date=sla["sla_due_date"], sla_breach_prob=sla["breach_probability"],
        risk_score=risk["overall"], risk_ombudsman=risk["ombudsman"],
        risk_churn=risk["churn"], risk_social=risk["social"],
        ml_category=(ml_cat or {}).get("category"),
        ml_category_prob=(ml_cat or {}).get("probability"),
        category_confidence=cat_conf,
        ml_sentiment=(ml_sent or {}).get("bucket"),
        ml_sentiment_prob=(ml_sent or {}).get("score"),
        sentiment_confidence=sent_conf,
        priority_score=priority,
        status=status_value,
        resolved_at=auto_resolved_at,
        processed_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    row.update({"category": cls["category"], "severity": cls["severity"]})
    dd.index_complaint(row)

    emit("Agent 6 (Root Cause)", "started")
    # Root Cause runs cluster-wide. We only refresh the global alert set when
    # the live caller asks for it (cheaper than re-running per submission).
    emit("Agent 6 (Root Cause)", "done",
         {"note": "Run pipeline.refresh_root_cause() for global re-clustering."})

    return {
        "id": complaint_id,
        "intake": intake_out,
        "classification": cls,
        "duplicate": dup,
        "draft_response": draft,
        "sla": sla,
        "risk": risk,
        "ml_category": ml_cat,
        "category_confidence": cat_conf,
        "ml_sentiment": ml_sent,
        "sentiment_confidence": sent_conf,
        "priority_score": priority,
        "status": status_value,
        "auto_resolved": status_value.startswith("auto_resolved"),
    }


def refresh_root_cause() -> int:
    """Re-run global clustering and overwrite stored alerts. Returns alert count."""
    alerts = root_cause.detect()
    root_cause.store_alerts(alerts)
    return len(alerts)


def ingest_new_complaint(raw: dict[str, Any]) -> str:
    """Insert a raw complaint row, returning its new id. Caller is responsible
    for then calling `process_one_streaming` on the returned id."""
    db.init_db()
    if not raw.get("id"):
        # Allocate next UBI-id by counting existing rows.
        existing = db.list_complaints()
        next_n = len(existing) + 1
        raw = {**raw, "id": f"UBI-{next_n:04d}"}
    raw.setdefault("date", datetime.utcnow().date().isoformat())
    db.upsert_raw_complaint(raw)
    return raw["id"]


def process_one(complaint_id: str, *, draft_response: bool = True) -> dict[str, Any]:
    """Thin alias for `process_one_streaming` -- single source of truth."""
    result = process_one_streaming(complaint_id, draft_response=draft_response,
                                   on_step=None)
    cls = result["classification"]; dup = result["duplicate"]
    sla = result["sla"]; risk = result["risk"]
    return {
        "id": complaint_id,
        "category": cls["category"],
        "severity": cls["severity"],
        "sentiment": cls["sentiment"],
        "is_duplicate": dup["is_duplicate"],
        "duplicate_of": dup["duplicate_of"],
        "sla_due_date": sla["sla_due_date"],
        "breach_prob": sla["breach_probability"],
        "risk_score": risk["overall"],
        "risk_ombudsman": risk["ombudsman"],
        "risk_churn": risk["churn"],
        "risk_social": risk["social"],
        "ml_category": (result.get("ml_category") or {}).get("category"),
        "category_confidence": result.get("category_confidence"),
        "ml_sentiment": (result.get("ml_sentiment") or {}).get("bucket"),
        "sentiment_confidence": result.get("sentiment_confidence"),
        "priority_score": result.get("priority_score"),
        "draft_response": result.get("draft_response"),
        "drafted": result.get("draft_response") is not None,
    }


def process_all(limit: int | None = None, *, draft_response: bool = True,
                progress: bool = True) -> dict[str, Any]:
    """Process every unprocessed complaint, then run Root Cause once."""
    db.init_db()
    pending = db.list_unprocessed(limit=limit)
    summaries: list[dict[str, Any]] = []
    for i, row in enumerate(pending, 1):
        try:
            summaries.append(process_one(row["id"], draft_response=draft_response))
        except Exception as e:
            summaries.append({"id": row["id"], "error": str(e)})
        if progress and (i % 10 == 0 or i == len(pending)):
            print(f"  processed {i}/{len(pending)}")

    alerts = root_cause.detect()
    root_cause.store_alerts(alerts)
    return {
        "processed": len(summaries),
        "errors": sum(1 for s in summaries if "error" in s),
        "duplicates_found": sum(1 for s in summaries if s.get("is_duplicate")),
        "alerts_detected": len(alerts),
        "summaries": summaries,
        "alerts": alerts,
    }


# --- helpers -----------------------------------------------------------------

def _lookup_sla_days(complaint: dict[str, Any]) -> int:
    rules_path = sla_monitor.SLA_RULES_PATH
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    return int(rules["sla_days"].get(complaint.get("category") or "General",
                                     rules["sla_days"]["General"]))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run the ComplaintIQ pipeline.")
    p.add_argument("--limit", type=int, default=None, help="process at most N complaints")
    p.add_argument("--no-draft", action="store_true", help="skip response drafting (faster smoke test)")
    args = p.parse_args()

    res = process_all(limit=args.limit, draft_response=not args.no_draft)
    print(f"\nDONE  processed={res['processed']}  errors={res['errors']}  "
          f"duplicates={res['duplicates_found']}  alerts={res['alerts_detected']}")
