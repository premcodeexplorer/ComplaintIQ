"""Agent 1 -- Intake.

Takes a raw complaint (dict with at minimum `complaint_text`, optional
`customer_name`, `channel`, `language`, `account_type`, `amount_involved`)
and uses the LLM to extract a normalized structured record. The model output
is merged with whatever fields the caller already provided -- caller-provided
fields win (we trust the source-of-truth metadata that came from the channel).
"""
from __future__ import annotations

from typing import Any

from .llm_client import chat_json

SYSTEM = (
    "You are an intake agent for an Indian bank's complaint system. "
    "You receive raw customer complaints in English, Hindi or Marathi "
    "(often code-mixed) and return a JSON object with normalized fields. "
    "Be conservative: if a field is not present, return null. Do NOT invent values."
)

PROMPT_TEMPLATE = """Extract a structured complaint record from the message below.

Return ONLY a JSON object with these exact keys (use null for unknown):
{{
  "customer_name": string or null,
  "issue_summary": short one-sentence summary in English,
  "account_type": one of ["savings", "current", "credit_card", "loan", "demat", null],
  "amount_involved": number (INR, no commas) or null,
  "transaction_id": string or null,
  "location_mentioned": string or null,
  "urgency_keywords": list of strings (e.g. ["urgent","blocked","unable"]) or [],
  "detected_language": one of ["english","hindi","marathi","mixed"]
}}

Channel: {channel}
Known customer name (may be empty): {customer_name}

Complaint text:
\"\"\"{text}\"\"\"
"""

EXPECTED_KEYS = {
    "customer_name", "issue_summary", "account_type", "amount_involved",
    "transaction_id", "location_mentioned", "urgency_keywords", "detected_language",
}


def extract(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a structured intake dict. Never raises -- on LLM failure returns a
    minimal fallback so downstream agents can still proceed."""
    text = raw.get("complaint_text") or ""
    prompt = PROMPT_TEMPLATE.format(
        channel=raw.get("channel", "unknown"),
        customer_name=raw.get("customer_name", "") or "",
        text=text,
    )
    try:
        data = chat_json(prompt, system=SYSTEM, temperature=0.0, max_tokens=400,
                         pii_values=[raw.get("customer_name")])
    except Exception as e:
        return _fallback(raw, error=str(e))

    # Normalize and fill in caller-known fields.
    out: dict[str, Any] = {k: data.get(k) for k in EXPECTED_KEYS}
    if not out.get("customer_name") and raw.get("customer_name"):
        out["customer_name"] = raw["customer_name"]
    if raw.get("account_type") and not out.get("account_type"):
        out["account_type"] = raw["account_type"]
    if raw.get("amount_involved") is not None and out.get("amount_involved") is None:
        out["amount_involved"] = raw["amount_involved"]
    if raw.get("language") and not out.get("detected_language"):
        out["detected_language"] = raw["language"]

    # Always coerce urgency_keywords to list[str].
    uk = out.get("urgency_keywords") or []
    if isinstance(uk, str):
        uk = [uk]
    out["urgency_keywords"] = [str(x) for x in uk if x]
    return out


def _fallback(raw: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "customer_name": raw.get("customer_name"),
        "issue_summary": (raw.get("complaint_text") or "")[:140],
        "account_type": raw.get("account_type"),
        "amount_involved": raw.get("amount_involved"),
        "transaction_id": None,
        "location_mentioned": raw.get("location"),
        "urgency_keywords": [],
        "detected_language": raw.get("language") or "english",
        "_llm_error": error,
    }


if __name__ == "__main__":
    # Smoke test against one row from the seed data.
    import json
    from pathlib import Path

    seed = Path(__file__).resolve().parent.parent / "data" / "complaints.json"
    rows = json.loads(seed.read_text(encoding="utf-8"))
    sample = rows[0]
    print("INPUT :", sample["id"], "-", sample["complaint_text"][:80])
    result = extract(sample)
    print("OUTPUT:")
    for k, v in result.items():
        print(f"  {k}: {v}")
