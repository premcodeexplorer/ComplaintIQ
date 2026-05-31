"""One-shot data remix.

  * Spreads filing dates over Jan -- May 2026 (300 May, 250 April, 450 Jan-Mar).
  * Re-runs SLA prediction on every row so due-dates / breach-probabilities
    reflect the new filing dates.
  * Flips 50 rows to the `mobile_app` channel (FIX 2 lives here too -- single
    DB transaction is cheaper than two).
  * Stamps resolution statuses:
        - 86  duplicates  -> auto_resolved_dup    (already flagged by Agent 3)
        - 100 low+polite  -> auto_resolved_std    (FIX 5 backfill)
        - 200 random rest -> resolved             (manually closed)
        - rest            -> open (Pending / Breached based on due date)
  * Writes a `resolved_at` timestamp where applicable.
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("USE_TF", "0")

import random
from datetime import date, datetime, timedelta
from typing import Any

from agents import sla_monitor
from database import db

random.seed(42)
TODAY = date(2026, 5, 29)

MAY_RANGE   = (date(2026, 5, 1),  date(2026, 5, 28))
APRIL_RANGE = (date(2026, 4, 1),  date(2026, 4, 30))
JAN_MAR     = (date(2026, 1, 1),  date(2026, 3, 31))

MOBILE_APP_TEMPLATES = [
    ("english", "Union Bank mobile app crashes every time I try to open the transfer screen. Reinstalled twice, same issue. {pixel}"),
    ("english", "Login failure on the mobile app -- MPIN keeps showing 'invalid' even though I just reset it via SMS OTP."),
    ("english", "Made a transfer of Rs {amt} via mobile app yesterday. Money debited but recipient never got it. Reference TXN26{txn}."),
    ("english", "Mobile app does not deliver the OTP for adding a new payee. Tried 6 times today, network signal is strong."),
    ("english", "Face ID broken on iPhone after the latest app update. Forced to enter MPIN every single time."),
    ("hindi",   "Mobile app पर bill payment करने पर ₹{amt} काट लिए पर electricity board तक पहुंचा नहीं। तुरंत refund चाहिए।"),
    ("hindi",   "मोबाइल app में login नहीं हो रहा -- MPIN सही डालने पर भी invalid का error आ रहा है। बहुत परेशान हूँ।"),
    ("hindi",   "App में UPI payment fail हो रहा है पिछले 3 दिन से। हर बार 'session expired' का error आता है।"),
    ("marathi", "App मध्ये statement download होत नाही, PDF रिक्त येतो. Tax filing साठी आवश्यक आहे."),
    ("marathi", "Mobile app वर credit card bill payment करताना crash होतो. Outstanding ₹{amt} आहे, due date उद्या."),
]


def _random_day(span: tuple[date, date]) -> str:
    start, end = span
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).isoformat()


def _resolved_offset(filing: date, kind: str) -> str:
    if kind == "auto_resolved_dup":   # closed near-immediately
        delta = random.randint(0, 2)
    elif kind == "auto_resolved_std": # closed within a day or two
        delta = random.randint(0, 2)
    else:                              # resolved manually, 2-15 days
        delta = random.randint(2, 15)
    resolved = filing + timedelta(days=delta)
    if resolved > TODAY:
        resolved = TODAY
    return resolved.isoformat()


def remix() -> dict[str, int]:
    db.init_db()
    rows = db.list_complaints()
    print(f"Total rows: {len(rows)}")

    # --- 1. Date redistribution ---------------------------------------------
    random.shuffle(rows)
    may_set   = {r["id"] for r in rows[:300]}
    april_set = {r["id"] for r in rows[300:550]}
    # remaining 450 keep their original Jan-March dates -- we still randomise
    # within Jan-March so the distribution looks even.
    janmar_set = {r["id"] for r in rows[550:]}

    print(f"Date plan: May={len(may_set)}  April={len(april_set)}  Jan-Mar={len(janmar_set)}")
    for r in rows:
        if r["id"] in may_set:
            new_date = _random_day(MAY_RANGE)
        elif r["id"] in april_set:
            new_date = _random_day(APRIL_RANGE)
        else:
            new_date = _random_day(JAN_MAR)
        db.update_complaint(r["id"], date=new_date)
        r["date"] = new_date

    # --- 2. Mobile-app channel for 50 rows ----------------------------------
    # Prefer rows already mentioning app keywords so the text stays plausible
    # without rewriting; rewrite the rest with a template.
    keywords = ("app", "mobile", "login", "OTP", "transaction", "crash")
    candidates = [r for r in rows if any(k in (r.get("complaint_text") or "").lower()
                                          for k in keywords)]
    random.shuffle(candidates)
    pick = candidates[:50]
    if len(pick) < 50:
        # top up with random rows whose text we'll overwrite
        extra_pool = [r for r in rows if r not in pick]
        random.shuffle(extra_pool)
        pick.extend(extra_pool[: 50 - len(pick)])

    mobile_rewrites = 0
    for i, r in enumerate(pick):
        lang, tmpl = MOBILE_APP_TEMPLATES[i % len(MOBILE_APP_TEMPLATES)]
        # If the original text already mentions app keywords, keep it; otherwise replace.
        text = r.get("complaint_text") or ""
        if not any(k in text.lower() for k in keywords):
            new_text = tmpl.format(
                amt=random.choice([500, 2000, 7500, 18000, 35000, 75000]),
                pixel=random.choice(["Pixel 7, Android 14", "Samsung S23", "iPhone 15"]),
                txn=random.randint(100000, 999999),
            )
            db.update_complaint(r["id"], channel="mobile_app",
                                complaint_text=new_text, language=lang)
            mobile_rewrites += 1
        else:
            db.update_complaint(r["id"], channel="mobile_app")
        r["channel"] = "mobile_app"
    print(f"mobile_app rows: 50  (rewrote {mobile_rewrites} texts)")

    # --- 3. Re-run SLA prediction on every row (depends on new date) --------
    sla_monitor._artefact = None  # force reload of the latest trained model
    refreshed_sla = 0
    for r in rows:
        # Fetch fresh state from DB (the writes above changed it).
        cur = db.get_complaint(r["id"])
        sla = sla_monitor.predict_breach(cur)
        db.update_complaint(r["id"],
                            sla_due_date=sla["sla_due_date"],
                            sla_breach_prob=sla["breach_probability"])
        refreshed_sla += 1
    print(f"SLA recomputed on {refreshed_sla} rows")

    # --- 4. Resolution statuses --------------------------------------------
    rows = db.list_complaints()  # reload with fresh dates + sla
    dup_rows = [r for r in rows if r.get("duplicate_of")]
    print(f"Auto-resolving {len(dup_rows)} duplicates")
    for r in dup_rows:
        db.update_complaint(r["id"], status="auto_resolved_dup",
                            resolved_at=_resolved_offset(
                                date.fromisoformat(r["date"]), "auto_resolved_dup"))

    # Low + Polite -> Auto-Resolved (Standard Reply Sent), max 100
    non_dup = [r for r in rows if not r.get("duplicate_of")]
    low_polite = [r for r in non_dup
                  if r.get("severity") == "Low" and r.get("sentiment") == "Polite"]
    random.shuffle(low_polite)
    auto_std = low_polite[:100]
    print(f"Auto-resolving {len(auto_std)} low+polite standard-reply rows")
    for r in auto_std:
        db.update_complaint(r["id"], status="auto_resolved_std",
                            resolved_at=_resolved_offset(
                                date.fromisoformat(r["date"]), "auto_resolved_std"))

    # 200 manually-resolved -- pick from the remaining open pool
    auto_ids = {r["id"] for r in auto_std} | {r["id"] for r in dup_rows}
    remaining = [r for r in rows if r["id"] not in auto_ids]
    random.shuffle(remaining)
    manually_resolved = remaining[:200]
    print(f"Marking {len(manually_resolved)} rows as manually Resolved")
    for r in manually_resolved:
        db.update_complaint(r["id"], status="resolved",
                            resolved_at=_resolved_offset(
                                date.fromisoformat(r["date"]), "resolved"))

    rest = [r for r in remaining[200:]]
    print(f"Leaving {len(rest)} rows as Pending (open)")
    for r in rest:
        db.update_complaint(r["id"], status="open", resolved_at=None)

    return {
        "rows": len(rows),
        "mobile_app": 50,
        "auto_resolved_dup": len(dup_rows),
        "auto_resolved_std": len(auto_std),
        "resolved": len(manually_resolved),
        "open": len(rest),
        "sla_refreshed": refreshed_sla,
    }


if __name__ == "__main__":
    summary = remix()
    print("\n=== DONE ===")
    for k, v in summary.items():
        print(f"  {k:18s} {v}")
