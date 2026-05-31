"""Priority inference -- loads `models/priority_gbm.joblib`."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "priority_gbm.joblib"

SEVERITY_ORDER = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}
SENTIMENT_ORDER = {"Angry": 3, "Frustrated": 2, "Neutral": 1, "Polite": 0}

_artefact: dict[str, Any] | None = None


def _get():
    global _artefact
    if _artefact is None and MODEL_PATH.exists():
        _artefact = joblib.load(MODEL_PATH)
    return _artefact


def score(complaint: dict[str, Any]) -> int | None:
    art = _get()
    if art is None:
        return None
    today_ts = pd.Timestamp(date.today())
    filed = pd.to_datetime(complaint.get("date"), errors="coerce")
    days_since = (today_ts - filed).days if pd.notna(filed) else 0
    customer_count = int(complaint.get("customer_complaint_count") or 0)
    if customer_count == 0:
        try:
            from database import db
            customer_count = len(db.customer_history(complaint.get("customer_name") or ""))
        except Exception:
            customer_count = 1
    row = pd.DataFrame([{
        "severity_encoded": SEVERITY_ORDER.get(complaint.get("severity") or "Medium", 1),
        "sentiment_encoded": SENTIMENT_ORDER.get(complaint.get("sentiment") or "Neutral", 1),
        "amount_involved": float(complaint.get("amount_involved") or 0),
        "customer_complaint_count": max(1, customer_count),
        "days_since_filed": max(0, days_since),
        "is_duplicate": 1 if complaint.get("duplicate_of") else 0,
        "breach_probability": float(complaint.get("sla_breach_prob")
                                    or complaint.get("breach_probability") or 0.3),
    }])[art["features"]]
    pred = float(art["model"].predict(row)[0])
    return max(0, min(100, int(round(pred))))
