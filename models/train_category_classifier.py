"""Train a TF-IDF + Logistic Regression category classifier.

Training labels come from the LLM Classifier (Agent 2). We bootstrap from
heuristic labels for any rows the LLM hasn't processed yet, so the script can
run before the full LLM pipeline finishes. As more LLM labels accumulate, the
classifier improves on retrain.

Artefact: `models/category_clf.joblib` with model + label list + confusion
matrix + accuracy for the dashboard's Model Performance tab.
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("USE_TF", "0")

import json
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "complaints.json"
MODEL_OUT = ROOT / "models" / "category_clf.joblib"

CATEGORIES = ["UPI", "ATM", "Card", "Loan", "NetBanking", "General"]
_CAT_PATTERNS = {
    "UPI":        re.compile(r"\b(upi|paytm|gpay|google pay|phonepe|bhim)\b", re.I),
    "ATM":        re.compile(r"\batm\b|cash.{0,15}machine|दिखाई नहीं|कैश मशीन", re.I),
    "Card":       re.compile(r"credit.?card|debit.?card|\bcard\b|कार्ड", re.I),
    "Loan":       re.compile(r"\bloan\b|emi|home.?loan|personal.?loan|ऋण", re.I),
    "NetBanking": re.compile(r"net.?banking|mobile.?banking|internet.?banking|netbanking|mobile app", re.I),
}


def _weak_label(text: str) -> str:
    for cat, pat in _CAT_PATTERNS.items():
        if pat.search(text or ""):
            return cat
    return "General"


def load_training_data() -> tuple[pd.DataFrame, str]:
    """Return (df with 'complaint_text' + 'category' columns, label-source tag)."""
    from database import db

    db.init_db()
    rows = db.list_complaints(where="processed_at IS NOT NULL")
    if len(rows) >= 100:
        df = pd.DataFrame(rows)[["complaint_text", "category"]].copy()
        df = df.dropna(subset=["complaint_text", "category"])
        df = df[df["category"].isin(CATEGORIES)]
        return df, f"llm_labels (n={len(df)})"

    # Fallback while LLM backfill is running -- use heuristic weak labels.
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    df = pd.DataFrame(raw)[["complaint_text"]].copy()
    df["category"] = df["complaint_text"].apply(_weak_label)
    return df, f"weak_labels (n={len(df)})"


def main() -> None:
    df, source = load_training_data()
    print(f"Training rows: {len(df)} | label source: {source}")
    print("Class distribution:")
    print(df["category"].value_counts().to_string())

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=10000,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=2.0,
            random_state=42,
        )),
    ])

    X = df["complaint_text"]
    y = df["category"]
    if y.nunique() < 2:
        print("Not enough class diversity; skipping.")
        return

    # stratify only if every class has at least 2 samples
    stratify = y if y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify,
    )
    pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_te)
    acc = float(accuracy_score(y_te, pred))
    print(f"\nHold-out accuracy: {acc:.3f}")
    print(classification_report(y_te, pred, digits=3))

    labels_present = sorted(set(y))
    cm = confusion_matrix(y_te, pred, labels=labels_present).tolist()

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": pipe,
        "labels": labels_present,
        "accuracy": round(acc, 4),
        "confusion_matrix": cm,
        "training_rows": len(df),
        "label_source": source,
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }, MODEL_OUT)
    print(f"\nSaved artefact -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
