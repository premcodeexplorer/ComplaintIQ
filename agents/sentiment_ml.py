"""HuggingFace sentiment model (cardiffnlp/twitter-roberta-base-sentiment-latest).

Provides a second opinion on the LLM-assigned sentiment. The HF model returns
{Positive, Neutral, Negative}; the LLM returns {Angry, Frustrated, Neutral,
Polite}. We map HF -> LLM-equivalent so agreement is well-defined.

Mapping:
  Positive -> Polite        Negative -> Angry        Neutral -> Neutral

Agreement levels:
  - 'High Confidence' if both buckets agree (positive / neutral / negative)
  - 'Needs Review' otherwise
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("USE_FLAX", "0")
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# Memory limits for PyTorch to survive on Railway 500MB
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from typing import Any

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

_pipeline = None

LLM_TO_BUCKET = {
    "Angry": "negative", "Frustrated": "negative",
    "Neutral": "neutral",
    "Polite": "positive",
}
HF_LABEL_TO_BUCKET = {
    "positive": "positive", "neutral": "neutral", "negative": "negative",
    # Roberta returns LABEL_0/1/2 in some versions
    "LABEL_2": "positive", "LABEL_1": "neutral", "LABEL_0": "negative",
}


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    from transformers import pipeline
    # device=-1 forces CPU; the model is ~500MB once downloaded.
    _pipeline = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=-1,
        truncation=True,
        max_length=512,
    )
    return _pipeline


_last_error: str | None = None


def last_error() -> str | None:
    return _last_error


def predict(text: str) -> dict[str, Any] | None:
    """Return {label, score, bucket} where bucket in {positive/neutral/negative}.
    Returns None if the model can't be loaded (no internet / no torch).
    On failure, the underlying exception text is stashed in `last_error()`."""
    global _last_error
    
    try:
        pipe = _get_pipeline()
    except Exception as e:
        _last_error = f"_get_pipeline: {type(e).__name__}: {e}"
        return None
    if not text:
        return {"label": "neutral", "score": 1.0, "bucket": "neutral"}
    try:
        out = pipe(text[:1000])[0]
    except Exception as e:
        _last_error = f"inference: {type(e).__name__}: {e}"
        return None
    _last_error = None
    label = str(out.get("label", "neutral"))
    bucket = HF_LABEL_TO_BUCKET.get(label, label.lower())
    return {
        "label": label,
        "score": round(float(out.get("score", 0.0)), 4),
        "bucket": bucket,
    }


def agreement(llm_sentiment: str | None, ml_result: dict[str, Any] | None) -> str:
    if not llm_sentiment or not ml_result:
        return "Unknown"
    llm_bucket = LLM_TO_BUCKET.get(llm_sentiment, "neutral")
    return "High Confidence" if llm_bucket == ml_result.get("bucket") else "Needs Review"
