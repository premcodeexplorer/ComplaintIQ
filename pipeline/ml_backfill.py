"""Apply the 3 ML models (category / sentiment / priority) + the latest SLA
model to every already-LLM-processed row.

Run this after the LLM pipeline has labelled enough complaints to train the ML
models -- it does NOT re-run any LLM calls, so it's fast (~1-2 minutes for
1000 rows).
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("USE_FLAX", "0")
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# Preload torch BEFORE sklearn/numpy so its MKL DLLs win on Windows.
try:
    import torch  # noqa: F401
except Exception:
    pass

from datetime import datetime

from agents import sentiment_ml, sla_monitor, ml_category, priority as priority_module
from database import db


def backfill(force_sla: bool = True, force_ml: bool = True) -> dict[str, int]:
    db.init_db()
    rows = db.list_complaints(where="processed_at IS NOT NULL")
    print(f"Rows to backfill: {len(rows)}")
    updated_sla = updated_cat = updated_sent = updated_pri = 0
    # Force a fresh artefact load for the SLA model (caller may have retrained it).
    if force_sla:
        sla_monitor._artefact = None

    for i, r in enumerate(rows, 1):
        update_kwargs: dict[str, object] = {}

        if force_sla:
            sla = sla_monitor.predict_breach(r)
            update_kwargs["sla_due_date"] = sla["sla_due_date"]
            update_kwargs["sla_breach_prob"] = sla["breach_probability"]
            r["sla_breach_prob"] = sla["breach_probability"]
            updated_sla += 1

        if force_ml:
            ml_cat = ml_category.predict(r.get("complaint_text") or "")
            if ml_cat:
                update_kwargs["ml_category"] = ml_cat["category"]
                update_kwargs["ml_category_prob"] = ml_cat["probability"]
                update_kwargs["category_confidence"] = ml_category.agreement(
                    r.get("category"), ml_cat["category"])
                updated_cat += 1

            ml_sent = sentiment_ml.predict(r.get("complaint_text") or "")
            if ml_sent:
                update_kwargs["ml_sentiment"] = ml_sent["bucket"]
                update_kwargs["ml_sentiment_prob"] = ml_sent["score"]
                update_kwargs["sentiment_confidence"] = sentiment_ml.agreement(
                    r.get("sentiment"), ml_sent)
                updated_sent += 1

            pri = priority_module.score(r)
            if pri is not None:
                update_kwargs["priority_score"] = pri
                updated_pri += 1

        if update_kwargs:
            db.update_complaint(r["id"], **update_kwargs)

        if i % 50 == 0 or i == len(rows):
            print(f"  backfilled {i}/{len(rows)}")

    print(f"\nDone. updates: sla={updated_sla} cat={updated_cat} "
          f"sent={updated_sent} priority={updated_pri}")
    return {"sla": updated_sla, "category": updated_cat,
            "sentiment": updated_sent, "priority": updated_pri}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--no-sla", action="store_true")
    p.add_argument("--no-ml", action="store_true")
    args = p.parse_args()
    backfill(force_sla=not args.no_sla, force_ml=not args.no_ml)
