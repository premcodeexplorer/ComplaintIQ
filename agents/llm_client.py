"""Shared Groq client used by all LLM-backed agents.

Reads the API key from one of (in order):
  1. Streamlit secrets (st.secrets["GROQ_API_KEY"])  -- when running in the dashboard
  2. Environment variable GROQ_API_KEY              -- when running CLI / scripts
  3. .env file (loaded via python-dotenv)
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv

from . import pii

load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# PII masking is ON by default. Set PII_MASKING=0 to disable (e.g. to show
# judges the raw-vs-masked difference live).
_PII_MASKING = os.getenv("PII_MASKING", "1").strip().lower() not in (
    "0", "false", "no", "off", "",
)


def _get_api_key() -> str | None:
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st  # type: ignore

        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return None


_client = None


def get_client():
    global _client
    if _client is not None:
        return _client
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Add it to your .env file or .streamlit/secrets.toml."
        )
    from groq import Groq

    _client = Groq(api_key=api_key)
    return _client


def chat(prompt: str, *, system: str | None = None, temperature: float = 0.2,
         max_tokens: int = 800, model: str | None = None, retries: int = 2,
         pii_values: list[str | None] | None = None) -> str:
    """Single-shot chat call. Returns the assistant message string.

    PII is masked before the text leaves for Groq and the reply is un-masked
    locally, so customer identifiers never reach the external API. `pii_values`
    lets the caller pass known literals to mask (e.g. the customer name).
    """
    masker = None
    if _PII_MASKING:
        masker = pii.PIIMasker()
        system = masker.mask(system, pii_values) if system else system
        prompt = masker.mask(prompt, pii_values)

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    client = get_client()
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model or DEFAULT_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            return masker.unmask(content) if masker else content
        except Exception as e:  # network blip, rate limit, etc.
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Groq call failed after {retries + 1} attempts: {last_err}")


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def chat_json(prompt: str, *, system: str | None = None, temperature: float = 0.1,
              max_tokens: int = 800, model: str | None = None,
              pii_values: list[str | None] | None = None) -> dict[str, Any]:
    """Chat call that expects a JSON object back. Strips ```json fences and grabs first {...} block."""
    raw = chat(prompt, system=system, temperature=temperature,
               max_tokens=max_tokens, model=model, pii_values=pii_values)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(raw)
        if not match:
            raise
        return json.loads(match.group(0))
