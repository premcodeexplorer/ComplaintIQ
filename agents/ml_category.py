"""ML category classifier -- TF-IDF + Logistic Regression second opinion to
the LLM classifier in Agent 2."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "category_clf.joblib"

_artefact: dict[str, Any] | None = None


_load_failed = False
_last_error: str | None = None


def _get():
    global _artefact, _load_failed, _last_error
    if _load_failed:
        return None
    if _artefact is None and MODEL_PATH.exists():
        try:
            _artefact = joblib.load(MODEL_PATH)
        except Exception as e:
            _load_failed = True
            _last_error = f"load: {type(e).__name__}: {e}"
            return None
    return _artefact


def last_error() -> str | None:
    return _last_error


def predict(text: str) -> dict[str, Any] | None:
    """Return {category, probability} or None if the model is unavailable.

    Wrapped in a broad try/except so a sklearn-version mismatch (e.g. the
    pickled `LogisticRegression` carrying a removed `multi_class` attribute
    when loaded on sklearn >= 1.7) **never** crashes the pipeline -- the LLM
    classification still stands; the ML second-opinion is just skipped.
    """
    global _last_error
    art = _get()
    if art is None:
        return None
    try:
        model = art["model"]
        labels = art["labels"]
        probs = model.predict_proba([text or ""])[0]
    except Exception as e:
        _last_error = f"predict: {type(e).__name__}: {e}"
        return None
    idx = int(probs.argmax())
    _last_error = None
    return {
        "category": labels[idx],
        "probability": round(float(probs[idx]), 4),
        "all_probabilities": {labels[i]: round(float(p), 4)
                              for i, p in enumerate(probs)},
    }


def agreement(llm_category: str | None, ml_category: str | None) -> str:
    """Return 'High Confidence' / 'Needs Review' / 'Unknown'."""
    if not llm_category or not ml_category:
        return "Unknown"
    return "High Confidence" if llm_category == ml_category else "Needs Review"
