"""Gmail IMAP Poller for ComplaintIQ.

Polls a Gmail inbox every 30 seconds using Python's built-in imaplib.
When a new unread email is found it is normalized into a complaint dict
and passed through the full 9-stage AI pipeline.

Zero new dependencies — uses only imaplib + email from the Python stdlib.

Quick-start (see step-by-step guide at the bottom of this file):
  1.  Create a Gmail account  e.g.  complaintiq.demo@gmail.com
  2.  Enable 2-Step Verification on that account
  3.  Generate a 16-character App Password (Google Account → Security)
  4.  Add two lines to your .env (or .streamlit/secrets.toml):
        GMAIL_ADDRESS  = complaintiq.demo@gmail.com
        GMAIL_APP_PASS = xxxx xxxx xxxx xxxx
  5.  In a separate terminal run:
        python -m channels.gmail_poller
  6.  Send a test email → watch the dashboard refresh in ≤ 30 s.

Environment variables (all optional — poller skips gracefully if missing):
  GMAIL_ADDRESS   full Gmail address to poll
  GMAIL_APP_PASS  16-char App Password (NOT your normal Gmail password)
  GMAIL_POLL_INTERVAL  seconds between polls (default: 30)
  GMAIL_MAILBOX   IMAP mailbox name (default: INBOX)
  GMAIL_MAX_BATCH max emails processed per poll cycle (default: 5)
"""
from __future__ import annotations

import email
import email.header
import email.utils
import imaplib
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Make the project root importable when run as  python -m channels.gmail_poller
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("complaintiq.gmail_poller")

# ---------------------------------------------------------------------------
# Config (reads from environment / .env)
# ---------------------------------------------------------------------------
GMAIL_IMAP_HOST = "imap.gmail.com"
GMAIL_IMAP_PORT = 993  # TLS

_GMAIL_ADDRESS  = os.getenv("GMAIL_ADDRESS", "").strip()
_GMAIL_APP_PASS = os.getenv("GMAIL_APP_PASS", "").strip()
_POLL_INTERVAL  = int(os.getenv("GMAIL_POLL_INTERVAL", "30"))
_MAILBOX        = os.getenv("GMAIL_MAILBOX", "INBOX")
_MAX_BATCH      = int(os.getenv("GMAIL_MAX_BATCH", "5"))

# Sender domains / addresses to silently skip.
# Google's own account notification emails, marketing mailers, etc.
# are never real banking complaints — skip them before ingestion.
_BLOCKED_SENDERS: set[str] = {
    "no-reply@accounts.google.com",
    "noreply@google.com",
    "no-reply@google.com",
    "googleplay-noreply@google.com",
    "mail-noreply@google.com",
    "noreply@youtube.com",
    "noreply@linkedin.com",
    "noreply@twitter.com",
    "noreply@x.com",
    "noreply@facebook.com",
}
# Also skip any sender whose address contains these substrings.
_BLOCKED_SENDER_PATTERNS: tuple[str, ...] = (
    "no-reply@",
    "noreply@",
    "do-not-reply@",
    "donotreply@",
    "notifications@",
    "mailer-daemon@",
    "postmaster@",
)

# Map common email subject / body patterns to a preferred language tag.
_LANG_HINTS: list[tuple[str, str]] = [
    # Hindi unicode block U+0900-U+097F
    ("\u0900", "hindi"),
    # Marathi-specific common words
    ("आहे",    "marathi"),
    ("माझ्",   "marathi"),
    # Devanagari without Marathi specifics → Hindi
    ("\u0901", "hindi"),
]


# ---------------------------------------------------------------------------
# IMAP helpers
# ---------------------------------------------------------------------------

def _connect() -> imaplib.IMAP4_SSL:
    """Open an authenticated IMAP-over-TLS connection. Raises on failure."""
    conn = imaplib.IMAP4_SSL(GMAIL_IMAP_HOST, GMAIL_IMAP_PORT)
    conn.login(_GMAIL_ADDRESS, _GMAIL_APP_PASS)
    conn.select(_MAILBOX, readonly=False)
    log.info("IMAP connected — %s  mailbox=%s", _GMAIL_ADDRESS, _MAILBOX)
    return conn


def _fetch_unseen(conn: imaplib.IMAP4_SSL) -> list[bytes]:
    """Return a list of message UIDs that are still UNSEEN (unread)."""
    status, data = conn.search(None, "UNSEEN")
    if status != "OK":
        return []
    uid_list = data[0].split()
    return uid_list


def _decode_header(raw: str | bytes | None) -> str:
    """Safely decode a possibly-encoded RFC-2047 email header value."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    parts = email.header.decode_header(raw)
    decoded: list[str] = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(str(part))
    return " ".join(decoded)


def _extract_body(msg: email.message.Message) -> str:
    """Walk a MIME message and return the first plaintext payload."""
    body_parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disposition = str(part.get("Content-Disposition") or "")
            if ctype == "text/plain" and "attachment" not in disposition:
                charset = part.get_content_charset() or "utf-8"
                raw = part.get_payload(decode=True)
                if raw:
                    body_parts.append(raw.decode(charset, errors="replace"))
    else:
        charset = msg.get_content_charset() or "utf-8"
        raw = msg.get_payload(decode=True)
        if raw:
            body_parts.append(raw.decode(charset, errors="replace"))

    return "\n".join(body_parts).strip()


def _detect_language(text: str) -> str:
    """Best-effort language detection using character/keyword hints."""
    for hint, lang in _LANG_HINTS:
        if hint in text:
            return lang
    return "english"


def _extract_amount(text: str) -> float | None:
    """Find the first 'Rs X' / '₹X' / 'INR X' pattern in the body."""
    import re
    patterns = [
        r"(?:Rs\.?|₹|INR)\s*([0-9,]+(?:\.[0-9]{1,2})?)",
        r"([0-9,]+(?:\.[0-9]{1,2})?)\s*(?:rupees?|/-)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                pass
    return None


def _extract_location(text: str, subject: str) -> str | None:
    """Return a city name if any known city appears in subject or body."""
    # Import the city list from the dashboard without pulling all of Streamlit.
    # Fallback: hardcoded common cities from the seed dataset.
    KNOWN_CITIES = {
        "Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Hyderabad",
        "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Kanpur", "Nagpur",
        "Indore", "Bhopal", "Patna", "Vadodara", "Nashik", "Aurangabad",
        "Solapur", "Thane", "Noida", "Agra", "Varanasi", "Coimbatore",
        "Gurgaon", "Rajkot", "Surat", "Vizag", "Allahabad", "Meerut",
        "Gorakhpur", "Bareilly", "Moradabad", "Aligarh", "Amravati",
        "Nagpur", "Kolhapur", "Sangli", "Latur", "Nanded",
    }
    combined = f"{subject} {text}"
    for city in KNOWN_CITIES:
        if city.lower() in combined.lower():
            return city
    return None


# ---------------------------------------------------------------------------
# Core email → complaint pipeline
# ---------------------------------------------------------------------------

def _email_to_complaint(uid: bytes, conn: imaplib.IMAP4_SSL) -> dict[str, Any] | None:
    """
    Fetch one email by UID, parse it, and return a normalized complaint dict
    ready for ingest_new_complaint().  Returns None on parse errors.
    """
    # Fetch full RFC-822 message
    status, data = conn.fetch(uid, "(RFC822)")
    if status != "OK" or not data or not data[0]:
        log.warning("fetch failed for uid=%s  status=%s", uid, status)
        return None

    raw_bytes = data[0][1]                                   # type: ignore[index]
    msg = email.message_from_bytes(raw_bytes)

    # ----- Sender / name ---------------------------------------------------
    from_raw = _decode_header(msg.get("From", ""))
    sender_name, sender_addr = email.utils.parseaddr(from_raw)
    sender_name = sender_name.strip() or sender_addr.split("@")[0]
    sender_addr_lower = sender_addr.lower()

    # ----- Block system / noreply senders -----------------------------------
    # Google security alerts, marketing emails, etc. are never real complaints.
    if sender_addr_lower in _BLOCKED_SENDERS or any(
        p in sender_addr_lower for p in _BLOCKED_SENDER_PATTERNS
    ):
        log.info("uid=%s  from=%s  SKIPPED (system/noreply sender)", uid, sender_addr)
        return None

    # ----- Subject + body --------------------------------------------------
    subject = _decode_header(msg.get("Subject", "(no subject)"))
    body    = _extract_body(msg)

    # Combine subject + body as the complaint text so the LLM sees full context.
    # Prefix the subject so the classifier sees what the customer described up-front.
    if subject and subject.lower() not in body.lower():
        complaint_text = f"Subject: {subject}\n\n{body}"
    else:
        complaint_text = body

    if not complaint_text.strip():
        log.info("uid=%s  from=%s  skipped (empty body)", uid, sender_addr)
        return None

    # ----- Date ------------------------------------------------------------
    date_raw = msg.get("Date", "")
    try:
        parsed_date = email.utils.parsedate_to_datetime(date_raw)
        date_str = parsed_date.date().isoformat()
    except Exception:
        date_str = datetime.utcnow().date().isoformat()

    # ----- Enriched metadata -----------------------------------------------
    language = _detect_language(complaint_text)
    amount   = _extract_amount(complaint_text)
    location = _extract_location(complaint_text, subject)

    complaint: dict[str, Any] = {
        "customer_name":  sender_name,
        "channel":        "email",
        "complaint_text": complaint_text.strip()[:4000],   # hard cap
        "language":       language,
        "date":           date_str,
        "account_type":   "savings",           # default; LLM Intake will refine
        "amount_involved": amount,
        "location":        location,
        # Store email metadata for audit trail (non-schema; ignored by upsert)
        "_email_uid":     uid.decode(),
        "_email_from":    sender_addr,
        "_email_subject": subject,
    }
    log.info(
        "parsed  uid=%-6s  from=%-30s  subject=%.50s  lang=%s  amt=%s  loc=%s",
        uid.decode(), sender_addr, subject, language, amount, location,
    )
    return complaint


def _mark_seen(conn: imaplib.IMAP4_SSL, uid: bytes) -> None:
    """Mark a message as \\Seen so it is not re-processed on the next poll."""
    conn.store(uid, "+FLAGS", "\\Seen")


# ---------------------------------------------------------------------------
# Poll loop
# ---------------------------------------------------------------------------

def poll_once() -> list[str]:
    """
    Single poll cycle.  Opens a fresh IMAP connection, fetches UNSEEN emails,
    processes each one through the ComplaintIQ pipeline, and returns the list
    of new complaint IDs created.

    Raises RuntimeError if credentials are missing.
    """
    if not _GMAIL_ADDRESS or not _GMAIL_APP_PASS:
        raise RuntimeError(
            "GMAIL_ADDRESS and GMAIL_APP_PASS must be set in your .env file.\n"
            "See the step-by-step guide at the bottom of channels/gmail_poller.py"
        )

    # Import pipeline here (not at module level) so tests can import this
    # file without triggering model loading.
    from pipeline.orchestrator import ingest_new_complaint, process_one

    new_ids: list[str] = []
    conn: imaplib.IMAP4_SSL | None = None
    try:
        conn = _connect()
        unseen_uids = _fetch_unseen(conn)

        if not unseen_uids:
            log.debug("no unseen emails")
            return []

        log.info("found %d unseen email(s) — processing up to %d",
                 len(unseen_uids), _MAX_BATCH)

        # Process newest emails first (reverse UID order is oldest→newest,
        # so we slice from the end)
        to_process = unseen_uids[-_MAX_BATCH:]

        for uid in to_process:
            try:
                complaint = _email_to_complaint(uid, conn)
                if complaint is None:
                    _mark_seen(conn, uid)
                    continue

                # --- Ingest into DB ----------------------------------------
                new_id = ingest_new_complaint(complaint)

                # --- Run all 9 pipeline stages ------------------------------
                log.info("pipeline starting  complaint_id=%s  from=%s",
                         new_id, complaint.get("_email_from"))
                t0 = time.monotonic()
                result = process_one(new_id)
                elapsed = round((time.monotonic() - t0), 1)

                log.info(
                    "pipeline done  id=%s  cat=%s  sev=%s  sent=%s  "
                    "breach=%.0f%%  risk=%s  dup=%s  elapsed=%.1fs",
                    new_id,
                    result.get("category", "?"),
                    result.get("severity", "?"),
                    result.get("sentiment", "?"),
                    (result.get("breach_prob") or 0) * 100,
                    result.get("risk_score", "?"),
                    result.get("is_duplicate", False),
                    elapsed,
                )
                new_ids.append(new_id)

                # Mark SEEN only AFTER successful ingestion so a crash during
                # processing doesn't silently swallow the complaint.
                _mark_seen(conn, uid)

            except Exception as exc:
                log.error("error processing uid=%s: %s", uid, exc, exc_info=True)
                # Still mark seen to avoid infinite re-processing of a broken email.
                try:
                    _mark_seen(conn, uid)
                except Exception:
                    pass

    except imaplib.IMAP4.error as imap_err:
        log.error("IMAP error: %s", imap_err)
    except OSError as net_err:
        log.error("Network error: %s", net_err)
    finally:
        if conn:
            try:
                conn.logout()
            except Exception:
                pass

    return new_ids


def run_forever(interval: int = _POLL_INTERVAL) -> None:
    """
    Blocking poll loop.  Runs poll_once() every `interval` seconds.
    Safe to Ctrl-C — prints a clean shutdown message.
    """
    log.info(
        "Gmail poller started  account=%s  interval=%ds  mailbox=%s",
        _GMAIL_ADDRESS or "(not configured)",
        interval,
        _MAILBOX,
    )

    if not _GMAIL_ADDRESS or not _GMAIL_APP_PASS:
        log.error(
            "\n"
            "  GMAIL_ADDRESS and GMAIL_APP_PASS are not set.\n"
            "  Add them to your .env file:\n"
            "\n"
            "    GMAIL_ADDRESS  = complaintiq.demo@gmail.com\n"
            "    GMAIL_APP_PASS = xxxx xxxx xxxx xxxx\n"
            "\n"
            "  See the step-by-step guide at the bottom of channels/gmail_poller.py"
        )
        return

    try:
        while True:
            cycle_start = time.monotonic()
            try:
                new_ids = poll_once()
                if new_ids:
                    log.info("cycle complete — %d new complaint(s): %s",
                             len(new_ids), new_ids)
                else:
                    log.debug("cycle complete — no new complaints")
            except Exception as exc:
                log.error("unexpected error in poll cycle: %s", exc, exc_info=True)

            elapsed = time.monotonic() - cycle_start
            sleep_for = max(0.0, interval - elapsed)
            log.debug("sleeping %.0fs until next poll", sleep_for)
            time.sleep(sleep_for)

    except KeyboardInterrupt:
        log.info("Poller stopped by user (Ctrl-C).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║          STEP-BY-STEP SETUP GUIDE — GMAIL IMAP POLLER               ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                      ║
    ║  STEP 1 — Create a dedicated Gmail account                           ║
    ║  ─────────────────────────────────────────                           ║
    ║  • Go to https://accounts.google.com/signup                          ║
    ║  • Create: complaintiq.demo@gmail.com  (or any name you like)        ║
    ║  • Use a strong password and save it                                 ║
    ║                                                                      ║
    ║  STEP 2 — Enable 2-Step Verification (required for App Passwords)    ║
    ║  ──────────────────────────────────────────────────────────────      ║
    ║  • Log in to the NEW Gmail account                                   ║
    ║  • Go to: https://myaccount.google.com/security                      ║
    ║  • Click "2-Step Verification" → follow the prompts                  ║
    ║  • This takes about 2 minutes                                        ║
    ║                                                                      ║
    ║  STEP 3 — Generate an App Password                                   ║
    ║  ───────────────────────────────                                      ║
    ║  • Still on https://myaccount.google.com/security                    ║
    ║  • Search "App passwords" in the search bar at the top               ║
    ║  • Click "App passwords"                                             ║
    ║  • Select app: "Mail"  |  Select device: "Windows Computer"         ║
    ║  • Click "Generate"                                                  ║
    ║  • Copy the 16-character password shown  (e.g. abcd efgh ijkl mnop) ║
    ║  • IMPORTANT: copy it NOW — Google only shows it once                ║
    ║                                                                      ║
    ║  STEP 4 — Add credentials to your .env file                          ║
    ║  ─────────────────────────────────────────                           ║
    ║  Open E:\ComplaintIQ\.env and add these two lines:                   ║
    ║                                                                      ║
    ║      GMAIL_ADDRESS  = complaintiq.demo@gmail.com                     ║
    ║      GMAIL_APP_PASS = abcd efgh ijkl mnop                            ║
    ║                                                                      ║
    ║  (The spaces in the App Password are fine — Python strips them)      ║
    ║                                                                      ║
    ║  STEP 5 — Run the poller in a separate terminal                      ║
    ║  ──────────────────────────────────────────────                      ║
    ║  Open a new PowerShell window in E:\ComplaintIQ\ and run:            ║
    ║                                                                      ║
    ║      python -m channels.gmail_poller                                 ║
    ║                                                                      ║
    ║  You should see:                                                     ║
    ║      2026-06-27T14:01:00  INFO  complaintiq.gmail_poller             ║
    ║          Gmail poller started  account=complaintiq.demo@gmail.com    ║
    ║          interval=30s  mailbox=INBOX                                 ║
    ║                                                                      ║
    ║  STEP 6 — Run the dashboard in your other terminal (already running) ║
    ║  ────────────────────────────────────────────────────────────────    ║
    ║      streamlit run dashboard/app.py                                  ║
    ║                                                                      ║
    ║  STEP 7 — The LIVE DEMO moment                                       ║
    ║  ──────────────────────────────                                      ║
    ║  Tell the judge:                                                     ║
    ║  "Sir/Ma'am, please send an email to complaintiq.demo@gmail.com      ║
    ║   describing any banking complaint."                                 ║
    ║                                                                      ║
    ║  Within 30 seconds the dashboard will auto-refresh and you will      ║
    ║  see their complaint appear — already classified, with:              ║
    ║    • Category (UPI / ATM / Card / Loan / NetBanking / General)       ║
    ║    • Severity (Critical / High / Medium / Low)                       ║
    ║    • Sentiment (Angry / Frustrated / Neutral / Polite)               ║
    ║    • SLA deadline                                                    ║
    ║    • Breach probability                                              ║
    ║    • Risk score (0–100)                                              ║
    ║    • Drafted reply in their language                                 ║
    ║                                                                      ║
    ║  TROUBLESHOOTING                                                     ║
    ║  ───────────────                                                     ║
    ║  "IMAP error: b'[AUTHENTICATIONFAILED]'"                             ║
    ║    → App Password is wrong. Regenerate and paste again.              ║
    ║    → Make sure you're using App Password NOT your Gmail password.    ║
    ║                                                                      ║
    ║  "IMAP error: b'[UNAVAILABLE]'"                                      ║
    ║    → IMAP not enabled. Go to Gmail Settings → See all settings       ║
    ║      → Forwarding and POP/IMAP → Enable IMAP → Save                 ║
    ║                                                                      ║
    ║  Dashboard doesn't refresh?                                          ║
    ║    → The dashboard caches data for 20 seconds (ttl=20 in app.py).   ║
    ║    → Wait up to 20s after the poller logs "pipeline done".           ║
    ║    → Or add st.rerun() manually if needed.                           ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="ComplaintIQ Gmail IMAP Poller — polls inbox every N seconds.",
    )
    parser.add_argument(
        "--interval", type=int, default=_POLL_INTERVAL,
        help=f"Seconds between polls (default: {_POLL_INTERVAL})",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Poll once and exit (useful for testing)",
    )
    args = parser.parse_args()

    if args.once:
        log.info("Running single poll cycle...")
        try:
            ids = poll_once()
            log.info("Done. New complaint IDs: %s", ids or "(none)")
        except RuntimeError as e:
            log.error("%s", e)
            sys.exit(1)
    else:
        run_forever(interval=args.interval)
