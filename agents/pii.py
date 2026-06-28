"""PII masking for outbound LLM calls.

Every call to the Groq LLM (intake / classifier / response drafter) is routed
through this masker by `llm_client.chat()`. Customer identifiers are replaced
with stable tokens BEFORE the text leaves the bank, and the LLM's reply is
re-hydrated locally afterwards. The real values never reach the external API.

Design:
  * Reversible token masking -> the draft reply can still greet the real
    customer and intake can still "extract" the real values, while Groq only
    ever sees `[NAME_1]`, `[ACCOUNT_1]`, `[PHONE_1]`, ...
  * Identifiers only. Amounts and locations are NOT masked -- they are not
    identifying and the classifier / SLA logic needs them.
  * India-specific patterns: mobile, bank account, card, Aadhaar, PAN, IFSC,
    UPI VPA, email. Plus literal masking of known values (the customer name).

Pure standard library (`re`) -- no new dependencies.
"""
from __future__ import annotations

import re
from typing import Iterable

# Order matters: most specific patterns first so a generic digit-run pattern
# does not swallow part of a structured identifier (email before UPI, card/
# Aadhaar before the generic account run, etc.).
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # name@host.tld  (must come before the UPI VPA rule, which has no dot+TLD)
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    # UPI VPA e.g. 9876543210@ybl, john.doe@okhdfc  (local part needs a digit
    # or dot so public Twitter handles like @UnionBankIndia are NOT masked)
    ("UPI", re.compile(r"\b[A-Za-z0-9._\-]*[0-9.][A-Za-z0-9._\-]*@[a-z]{2,}\b")),
    # PAN: ABCDE1234F
    ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
    # IFSC: 4 letters, then 0, then 6 alnum
    ("IFSC", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")),
    # 16-digit card, optionally grouped by space/hyphen
    ("CARD", re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b")),
    # Aadhaar: 12 digits, optionally grouped 4-4-4
    ("AADHAAR", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    # Indian mobile: optional +91, then 10 digits starting 6-9
    ("PHONE", re.compile(r"\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b")),
    # Generic bank account / long reference: 9-18 contiguous digits
    ("ACCOUNT", re.compile(r"\b\d{9,18}\b")),
]


class PIIMasker:
    """Stateful masker for a single LLM call.

    Masking the system prompt and the user prompt with the *same* instance
    keeps tokens consistent across both, and lets the same instance un-mask the
    response.
    """

    def __init__(self) -> None:
        self._map: dict[str, str] = {}      # token  -> original value
        self._seen: dict[str, str] = {}     # original value -> token
        self._counts: dict[str, int] = {}   # kind -> running count

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self._map)

    def _token(self, kind: str, value: str) -> str:
        if value in self._seen:
            return self._seen[value]
        self._counts[kind] = self._counts.get(kind, 0) + 1
        # No brackets: the LLM strips `[ ]` (treats them as formatting) but
        # preserves a solid alnum/underscore token like PII_NAME_1 verbatim.
        token = f"PII_{kind}_{self._counts[kind]}"
        self._seen[value] = token
        self._map[token] = value
        return token

    def mask(self, text: str | None, known_values: Iterable[str | None] | None = None) -> str:
        """Return `text` with identifiers replaced by stable tokens."""
        if not text:
            return text or ""

        # 1. Structured identifiers via regex, in priority order. Done first so
        #    that an identifier containing the customer's name (e.g. the local
        #    part of an email) is masked as a whole before name-masking runs.
        for kind, pattern in _PATTERNS:
            text = pattern.sub(lambda m, k=kind: self._token(k, m.group(0)), text)

        # 2. Known literal values (customer name + its parts). Names are not
        #    regex-detectable, so we replace them explicitly. Longest first so
        #    "Rahul Sharma" is masked before the lone "Rahul".
        candidates: set[str] = set()
        for val in (known_values or []):
            if not val:
                continue
            v = str(val).strip()
            if len(v) >= 2 and v.lower() not in _NAME_STOPWORDS:
                candidates.add(v)
            for part in v.split():
                if len(part) >= 3 and part.lower() not in _NAME_STOPWORDS:
                    candidates.add(part)
        for cand in sorted(candidates, key=len, reverse=True):
            token = self._token("NAME", cand)
            text = re.sub(rf"\b{re.escape(cand)}\b", token, text, flags=re.IGNORECASE)
        return text

    def unmask(self, text: str | None) -> str:
        """Restore original values from tokens (used on the LLM's reply).

        Tolerant of the LLM lightly reformatting a token -- case changes and
        stray wrapping brackets/whitespace (e.g. `[PII_NAME_1]`, `pii_name_1`)
        are all restored. Longest tokens first so PII_NAME_1 is handled before
        any shorter prefix could partially match.
        """
        if not text:
            return text or ""
        for token in sorted(self._map, key=len, reverse=True):
            original = self._map[token]
            esc = re.escape(token)
            # Match the bare token, OR a bracket-wrapped form (consuming inner
            # whitespace only) -- never the surrounding spaces of the sentence.
            pattern = rf"\[\s*{esc}\s*\]|{esc}"
            text = re.sub(pattern, lambda _m, o=original: o, text, flags=re.IGNORECASE)
        return text


# Generic placeholder names that should never be masked as a real customer name.
_NAME_STOPWORDS = {
    "customer", "walk-in", "walk in", "walkin", "user", "the", "and", "sir",
    "madam", "none", "null", "unknown", "na", "n/a",
}
