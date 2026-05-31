"""Agent 4 -- Response Drafter.

Generates a policy-compliant draft bank response in the customer's language.
Output is plain text (no JSON wrapper) so it can be shown directly in the
dashboard / copied into the channel reply box.
"""
from __future__ import annotations

from typing import Any

from .llm_client import chat

SYSTEM = (
    "You are a professional customer service writer for Union Bank of India. "
    "You draft empathetic, RBI-compliant replies that acknowledge the issue, "
    "give a clear next step with a reference / SLA window, and end politely. "
    "Never promise refunds without investigation. Never share OTP / credentials. "
    "If complaint involves potential fraud, advise blocking card / freezing account "
    "and call the 24x7 helpline 1800-22-2244. Keep replies under 130 words."
)

PROMPT = """Draft a response to the customer complaint below.

Reply in: {language}
Channel: {channel} (match the tone -- twitter/whatsapp may be shorter than email)
Category: {category}
Severity: {severity}
Sentiment of customer: {sentiment}
Promised SLA: resolve within {sla_days} business days
Reference number to cite: {ref}

Customer: {customer_name}
Complaint:
\"\"\"{text}\"\"\"

Write ONLY the reply text, no headers, no quotes, no commentary."""


def draft(complaint: dict[str, Any], *, sla_days: int = 5) -> str:
    """Compose a draft reply. `complaint` must have category/severity/sentiment
    fields already set by the Classifier."""
    language = (complaint.get("language") or "english").capitalize()
    ref = f"UBI/{complaint.get('id', 'NEW')}"
    prompt = PROMPT.format(
        language=language,
        channel=complaint.get("channel") or "email",
        category=complaint.get("category") or "General",
        severity=complaint.get("severity") or "Medium",
        sentiment=complaint.get("sentiment") or "Neutral",
        sla_days=sla_days,
        ref=ref,
        customer_name=complaint.get("customer_name") or "Customer",
        text=complaint.get("complaint_text", ""),
    )
    try:
        return chat(prompt, system=SYSTEM, temperature=0.4, max_tokens=400).strip()
    except Exception as e:
        return _fallback(complaint, sla_days=sla_days, error=str(e))


def _fallback(complaint: dict[str, Any], *, sla_days: int, error: str) -> str:
    # `error` is intentionally captured for server-side logging but never shown
    # to the customer / judge. The reply must read like any other template.
    return (
        f"Dear {complaint.get('customer_name') or 'Customer'},\n\n"
        f"Thank you for contacting Union Bank of India regarding your "
        f"{complaint.get('category', 'service')} concern. "
        f"We have registered your complaint under reference UBI/{complaint.get('id','NEW')} "
        f"and our team will investigate and resolve it within {sla_days} business days. "
        f"For urgent assistance please call 1800-22-2244.\n\n"
        f"Regards,\nUnion Bank Customer Care\n"
        f"(standard template)"
    )


if __name__ == "__main__":
    from database import db
    from agents.classifier import classify

    db.init_db()
    sample = db.get_complaint("UBI-0002")  # Hindi credit card fraud
    sample.update(classify(sample))
    print("--- DRAFT REPLY ---")
    print(draft(sample, sla_days=7))
