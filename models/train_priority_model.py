"""Gradient Boosting priority regressor (target = composite priority 0-100).

Training labels are synthesized from a human-interpretable weighted formula
combining severity, sentiment, amount, complaint_count, days_since_filed,
duplication and breach probability. The model then learns to predict this
score from features alone -- so the dashboard can produce a priority for a
NEW complaint before all signals are known.

Why a model when the score is a formula?
  - The formula captures intent but is brittle when one signal is missing.
  - The trained GBM smooths over those gaps and lets us replace coefficients
    with learned weights once we have real outcome data (e.g. which complaints
    actually escalated).
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("USE_TF", "0")

import json
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "complaints.json"
MODEL_OUT = ROOT / "models" / "priority_gbm.joblib"

SEVERITY_ORDER = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}
SENTIMENT_ORDER = {"Angry": 3, "Frustrated": 2, "Neutral": 1, "Polite": 0}

FEATURES = ["severity_encoded", "sentiment_encoded", "amount_involved",
            "customer_complaint_count", "days_since_filed", "is_duplicate",
            "breach_probability"]


def _engineer(df: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    df = df.copy()
    today_ts = pd.Timestamp(today or date.today())
    df["amount_involved"] = df["amount_involved"].fillna(0).astype(float)
    dates = pd.to_datetime(df.get("date"), errors="coerce")
    df["days_since_filed"] = (today_ts - dates).dt.days.fillna(0).astype(int)
    df["severity_encoded"] = df.get("severity").map(SEVERITY_ORDER).fillna(1).astype(int)
    df["sentiment_encoded"] = df.get("sentiment").map(SENTIMENT_ORDER).fillna(1).astype(int)
    if "is_duplicate" not in df.columns or df["is_duplicate"].isna().all():
        df["is_duplicate"] = df.get("duplicate_of").notna().astype(int) \
            if "duplicate_of" in df.columns else 0
    if "customer_complaint_count" not in df.columns:
        counts = df.groupby("customer_name").size().rename("customer_complaint_count")
        df = df.merge(counts, on="customer_name", how="left")
    if "breach_probability" not in df.columns:
        df["breach_probability"] = df.get("sla_breach_prob")
    df["breach_probability"] = df["breach_probability"].fillna(0.3).astype(float)
    return df


def synthesize_priority(df: pd.DataFrame) -> np.ndarray:
    """Weighted composite -- the score we want the model to learn to mimic."""
    score = (
        df["severity_encoded"] * 12        # 0..36
        + df["sentiment_encoded"] * 7      # 0..21
        + np.clip(df["amount_involved"] / 5000.0, 0, 20)   # 0..20
        + np.clip(df["customer_complaint_count"] * 3, 0, 12)   # 0..12
        + np.clip(df["days_since_filed"] * 0.4, 0, 10)     # 0..10 (older = higher)
        + df["breach_probability"] * 18    # 0..18
        - df["is_duplicate"] * 8           # duplicates can wait
    )
    return np.clip(score, 0, 100).astype(float)


def load_training_data() -> pd.DataFrame:
    from database import db

    db.init_db()
    rows = db.list_complaints(where="processed_at IS NOT NULL")
    if len(rows) >= 100:
        return pd.DataFrame(rows)
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    df = pd.DataFrame(raw)
    df["severity"] = "Medium"; df["sentiment"] = "Neutral"
    df["duplicate_of"] = None; df["sla_breach_prob"] = 0.3
    return df


def main() -> None:
    df = load_training_data()
    df = _engineer(df)
    y = synthesize_priority(df)
    print(f"Training rows: {len(df)} | priority mean={y.mean():.1f} std={y.std():.1f}")

    X = df[FEATURES]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
    gbm = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.06,
        subsample=0.9, random_state=42,
    )
    gbm.fit(X_tr, y_tr)
    pred = gbm.predict(X_te)
    mae = float(mean_absolute_error(y_te, pred))
    r2 = float(r2_score(y_te, pred))
    print(f"Hold-out MAE: {mae:.2f}  R^2: {r2:.3f}")

    importances = {n: round(float(v), 5)
                   for n, v in zip(FEATURES, gbm.feature_importances_)}
    print("Feature importance:")
    for n, v in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {n:28s} {v:.4f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": gbm,
        "features": FEATURES,
        "training_rows": len(df),
        "test_metrics": {"mae": round(mae, 3), "r2": round(r2, 4)},
        "feature_importances": importances,
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }, MODEL_OUT)
    print(f"\nSaved artefact -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
