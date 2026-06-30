"""Tests for agents.pii — PII masking on outbound LLM text.

Run either way:
    python -m pytest tests/test_pii.py -v
    python tests/test_pii.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.pii import PIIMasker


def test_masks_account_phone_and_roundtrips():
    m = PIIMasker()
    text = "My a/c 50100123456789 was debited, please call me on 9876543210"
    masked = m.mask(text)
    assert "50100123456789" not in masked
    assert "9876543210" not in masked
    assert "PII_ACCOUNT_1" in masked
    assert "PII_PHONE_1" in masked
    # the LLM reply echoes the tokens; un-mask restores the real values
    assert m.unmask(masked) == text


def test_masks_email_pan_aadhaar_card_upi():
    m = PIIMasker()
    text = ("email rahul.k@gmail.com PAN ABCDE1234F aadhaar 1234 5678 9012 "
            "card 4111 1111 1111 1111 upi 9876501234@ybl")
    masked = m.mask(text)
    for secret in ("rahul.k@gmail.com", "ABCDE1234F", "1234 5678 9012",
                   "4111 1111 1111 1111", "9876501234@ybl"):
        assert secret not in masked, f"leaked: {secret}"
    assert m.unmask(masked) == text


def test_masks_customer_name_including_parts():
    m = PIIMasker()
    text = "Hello, I am Rahul Sharma and Rahul is very upset about the UPI failure."
    masked = m.mask(text, known_values=["Rahul Sharma"])
    assert "Rahul" not in masked
    assert "Sharma" not in masked
    assert m.unmask(masked) == text


def test_same_value_gets_same_token():
    m = PIIMasker()
    masked = m.mask("call 9876543210 or 9876543210 again")
    assert masked.count("PII_PHONE_1") == 2  # same number -> same token


def test_does_not_mask_amount_or_short_numbers():
    m = PIIMasker()
    text = "I lost 5000 rupees in a UPI txn dated 2026"
    masked = m.mask(text)
    assert masked == text  # nothing to mask -> unchanged


def test_does_not_mask_generic_placeholder_name():
    m = PIIMasker()
    text = "The customer reported a card issue."
    masked = m.mask(text, known_values=["Customer", "Walk-in"])
    assert masked == text


def test_empty_input_is_safe():
    m = PIIMasker()
    assert m.mask("") == ""
    assert m.mask(None) == ""
    assert m.unmask(None) == ""


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
