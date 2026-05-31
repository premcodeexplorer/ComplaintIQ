"""RBI compliance report generator.

Produces a CSV matching the schema expected by RBI Master Circular on Customer
Service in Banks (2024). Columns:
  Complaint ID, Customer Name, Type, Severity, Channel, Date Filed,
  SLA Deadline, Days Remaining, Breach Status, Resolution Status
"""
from __future__ import annotations

from datetime import date
from io import StringIO
from typing import Any

import pandas as pd


REPORT_COLUMNS = [
    "Complaint ID", "Customer Name", "Type", "Severity", "Channel",
    "Date Filed", "SLA Deadline", "Days Remaining",
    "Breach Status", "Resolution Status",
]


def _breach_status(due_date: pd.Timestamp | None, breach_prob: float | None,
                   today: pd.Timestamp) -> str:
    if due_date is pd.NaT or due_date is None or pd.isna(due_date):
        return "Unknown"
    if due_date < today:
        return "Breached"
    if breach_prob is not None and not pd.isna(breach_prob) and breach_prob >= 0.5:
        return "At Risk"
    return "On Track"


def _resolution_status(row: dict[str, Any], today: pd.Timestamp) -> str:
    status = (row.get("status") or "open").lower()
    if status == "resolved":
        return "Resolved"
    if status == "auto_resolved_dup":
        return "Auto-Resolved (Duplicate)"
    if status == "auto_resolved_std":
        return "Auto-Resolved (Standard Reply Sent)"
    if status == "escalated":
        return "Escalated"
    # Legacy: rows whose duplicate flag was set but status not migrated.
    if row.get("duplicate_of"):
        return "Auto-Resolved (Duplicate)"
    due = row.get("sla_due_date")
    if due is not None and not pd.isna(due) and pd.Timestamp(due) < today:
        return "Breached"
    return "Pending"


def build_report(rows: list[dict[str, Any]] | pd.DataFrame,
                 today: date | None = None) -> pd.DataFrame:
    today_ts = pd.Timestamp(today or date.today())
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    if df.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sla_due_date"] = pd.to_datetime(df["sla_due_date"], errors="coerce")

    out = pd.DataFrame({
        "Complaint ID":      df["id"],
        "Customer Name":     df["customer_name"],
        "Type":              df["category"].fillna("General"),
        "Severity":          df["severity"].fillna("Medium"),
        "Channel":           df["channel"],
        "Date Filed":        df["date"].dt.date,
        "SLA Deadline":      df["sla_due_date"].dt.date,
        "Days Remaining":    (df["sla_due_date"] - today_ts).dt.days,
        "Breach Status":     [
            _breach_status(d, p, today_ts)
            for d, p in zip(df["sla_due_date"], df["sla_breach_prob"])
        ],
        "Resolution Status": [
            _resolution_status(r.to_dict(), today_ts) for _, r in df.iterrows()
        ],
    })
    return out[REPORT_COLUMNS]


def summary_stats(rows: list[dict[str, Any]] | pd.DataFrame,
                  today: date | None = None) -> dict[str, Any]:
    """Counts shown above the download button."""
    today_ts = pd.Timestamp(today or date.today())
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    if df.empty:
        return {"total": 0, "resolved": 0, "pending": 0, "breached": 0, "by_category": {}}

    df["sla_due_date"] = pd.to_datetime(df["sla_due_date"], errors="coerce")
    status = df["status"].fillna("open")
    resolved = int((status == "resolved").sum())
    auto_resolved = int(status.isin(("auto_resolved_dup", "auto_resolved_std")).sum())
    # Breached = still open AND past due.
    open_mask = status == "open"
    breached = int(((open_mask) & (df["sla_due_date"] < today_ts)
                    & df["sla_due_date"].notna()).sum())
    pending = int(((open_mask) & ((df["sla_due_date"] >= today_ts)
                                  | df["sla_due_date"].isna())).sum())
    by_category = df["category"].fillna("General").value_counts().to_dict()
    return {
        "total": len(df),
        "resolved": resolved,
        "auto_resolved": auto_resolved,
        "pending": pending,
        "breached": breached,
        "by_category": by_category,
    }


def to_csv(rows: list[dict[str, Any]] | pd.DataFrame,
           today: date | None = None) -> bytes:
    df = build_report(rows, today=today)
    buf = StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8-sig")  # BOM helps Excel open Hindi names correctly
