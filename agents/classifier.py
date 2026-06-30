"""Agent 2 -- Classifier.

Assigns (category, severity, sentiment) to a complaint using the LLM.
Includes a deterministic keyword-based fallback when the LLM is unreachable so
the pipeline still completes.
"""
from __future__ import annotations

import re
from typing import Any

from .llm_client import chat_json

CATEGORIES = ["UPI", "ATM", "Card", "Loan", "NetBanking", "General"]
SEVERITIES = ["Critical", "High", "Medium", "Low"]
SENTIMENTS = ["Angry", "Frustrated", "Neutral", "Polite"]

SYSTEM = (
    "You are a complaint classification agent for an Indian bank. "
    "You assign a single category, severity, and sentiment label to each complaint. "
    "Respond with JSON only."
)

PROMPT = """Classify the complaint below.

Category MUST be one of: {cats}
Severity MUST be one of: {sevs}
  - Critical: fraud, unauthorized debit, locked out, large amounts >= 50000
  - High: amount-stuck issues, repeated failures, account access issues
  - Medium: typical service issues, single failed transaction, delays
  - Low: information requests, minor inconvenience
Sentiment MUST be one of: {sens}
  - Angry: aggressive, uses caps/exclamations, threats to escalate
  - Frustrated: tired, repeated attempts, demands action
  - Neutral: factual, no strong emotion
  - Polite: courteous phrasing

Return ONLY this JSON shape:
{{"category":"...","severity":"...","sentiment":"...","rationale":"one short sentence"}}

Complaint text:
\"\"\"{text}\"\"\"
Amount involved (INR): {amount}
Channel: {channel}
Account type: {account_type}
"""


def classify(complaint: dict[str, Any]) -> dict[str, Any]:
    """`complaint` is a row from the DB (must have complaint_text). Returns
    {category, severity, sentiment, rationale}. Never raises."""
    prompt = PROMPT.format(
        cats=", ".join(CATEGORIES),
        sevs=", ".join(SEVERITIES),
        sens=", ".join(SENTIMENTS),
        text=complaint.get("complaint_text", ""),
        amount=complaint.get("amount_involved"),
        channel=complaint.get("channel"),
        account_type=complaint.get("account_type"),
    )
    try:
        data = chat_json(prompt, system=SYSTEM, temperature=0.0, max_tokens=200,
                         pii_values=[complaint.get("customer_name")])
    except Exception:
        return _fallback(complaint)

    return {
        "category": _pick(data.get("category"), CATEGORIES, default="General"),
        "severity": _pick(data.get("severity"), SEVERITIES, default="Medium"),
        "sentiment": _pick(data.get("sentiment"), SENTIMENTS, default="Neutral"),
        "rationale": str(data.get("rationale", ""))[:240],
    }


def _pick(value: Any, choices: list[str], default: str) -> str:
    if not value:
        return default
    v = str(value).strip()
    for c in choices:
        if v.lower() == c.lower():
            return c
    # tolerate partial matches like "Credit Card" -> "Card"
    for c in choices:
        if c.lower() in v.lower():
            return c
    return default


# --- deterministic fallback (no LLM) -----------------------------------------

_CAT_PATTERNS = [
    ("UPI",        re.compile(r"\bUPI\b|upi", re.I)),
    ("ATM",        re.compile(r"\bATM\b|cash.{0,15}machine", re.I)),
    ("Card",       re.compile(r"credit.?card|debit.?card|\bcard\b", re.I)),
    ("Loan",       re.compile(r"\bloan\b|EMI|home.?loan|personal.?loan", re.I)),
    ("NetBanking", re.compile(r"net.?banking|mobile.?banking|internet.?banking", re.I)),
]
_HIGH_KEYWORDS = re.compile(r"unauthor|fraud|stuck|debit.{0,15}fail|debited", re.I)
_ANGRY_KEYWORDS = re.compile(r"unacceptable|disgust|outrage|terrible|worst|!!", re.I)
_POLITE_KEYWORDS = re.compile(r"kindly|please|would you|appreciate|sir/madam|कृपया", re.I)


def _fallback(complaint: dict[str, Any]) -> dict[str, Any]:
    text = complaint.get("complaint_text", "") or ""
    category = "General"
    for cat, pat in _CAT_PATTERNS:
        if pat.search(text):
            category = cat
            break

    amt = complaint.get("amount_involved") or 0
    if amt and amt >= 50000:
        severity = "Critical"
    elif _HIGH_KEYWORDS.search(text):
        severity = "High"
    elif amt and amt > 0:
        severity = "Medium"
    else:
        severity = "Low"

    if _ANGRY_KEYWORDS.search(text):
        sentiment = "Angry"
    elif _POLITE_KEYWORDS.search(text):
        sentiment = "Polite"
    elif "!" in text or "?" in text:
        sentiment = "Frustrated"
    else:
        sentiment = "Neutral"

    return {
        "category": category,
        "severity": severity,
        "sentiment": sentiment,
        "rationale": "fallback (LLM unavailable) -- keyword heuristic",
    }


if __name__ == "__main__":
    from database import db

    db.init_db()
    samples = db.list_complaints(limit=4)
    for c in samples:
        r = classify(c)
        print(f"{c['id']:10s} amt={c['amount_involved']!s:>8} -> "
              f"{r['category']:10s} {r['severity']:8s} {r['sentiment']:10s} | {r['rationale'][:80]}")
