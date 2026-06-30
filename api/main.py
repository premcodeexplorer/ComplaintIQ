"""FastAPI backend for ComplaintIQ.

Endpoints:
  POST /complaint   -- submit a new complaint, returns the processed result
  GET  /complaints  -- list complaints with filters
  GET  /stats       -- dashboard KPI stats
  GET  /report      -- download RBI compliance CSV

Run locally:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

# DLL hygiene -- see orchestrator.py for context.
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("USE_FLAX", "0")
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
try:
    import torch  # noqa: F401
except Exception:
    pass

import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional

import os
import tempfile
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form, Body
from fastapi.responses import Response
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware


# Make project root importable when run as `uvicorn api.main:app`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import db                       # noqa: E402
from dashboard import rbi_report              # noqa: E402
from pipeline.orchestrator import (           # noqa: E402
    ingest_new_complaint, process_one_streaming, refresh_root_cause,
)
from agents.llm_client import get_client      # noqa: E402
from agents.spam_filter import check_spam     # noqa: E402



app = FastAPI(
    title="ComplaintIQ API",
    version="1.0.0",
    description="Programmatic access to the 6-agent complaint intelligence pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# --- request / response models -----------------------------------------------

class ComplaintIn(BaseModel):
    complaint_text: str = Field(min_length=4, max_length=4000)
    customer_name: str = "Walk-in"
    channel: str = "email"
    language: str = "english"
    account_type: str = "savings"
    location: Optional[str] = None
    amount_involved: Optional[float] = None
    refresh_root_cause: bool = Field(
        default=False,
        description="If true, re-run KMeans cluster detection after processing.",
    )


class ComplaintOut(BaseModel):
    id: str
    intake: dict[str, Any]
    classification: dict[str, Any]
    duplicate: dict[str, Any]
    draft_response: Optional[str]
    sla: dict[str, Any]
    risk: dict[str, int]
    transcribed_text: Optional[str] = None



# --- endpoints ---------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index() -> dict[str, str]:
    return {"service": "ComplaintIQ", "docs": "/docs",
            "endpoints": "POST /complaint, GET /complaints, GET /stats, GET /report"}


@app.post("/complaint", response_model=ComplaintOut)
def submit_complaint(payload: ComplaintIn) -> dict[str, Any]:
    """Submit a new complaint and run it through all 6 agents synchronously."""
    new_id = ingest_new_complaint(payload.model_dump(exclude={"refresh_root_cause"}))
    try:
        result = process_one_streaming(new_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
    if payload.refresh_root_cause:
        refresh_root_cause()
    return {"id": new_id, **result}


@app.post("/verify-account")
def verify_account(payload: dict = Body(...)) -> dict[str, Any]:
    account_number = payload.get("account_number")
    if not account_number:
        raise HTTPException(status_code=400, detail="account_number is required")
        
    customer = db.verify_customer(account_number)
    if not customer:
        raise HTTPException(status_code=401, detail="Invalid account number")
        
    if db.is_user_blocked(account_number):
        raise HTTPException(status_code=403, detail="Account is blocked due to excessive invalid requests")
        
    return {"success": True, "customer_name": customer["customer_name"]}


@app.post("/voice-complaint", response_model=ComplaintOut)
async def submit_voice_complaint(
    audio: UploadFile = File(...),
    account_number: str = Form(...),
    customer_name: str = Form(...),
    location: Optional[str] = Form(None),
    account_type: Optional[str] = Form(None),
    amount_involved: Optional[float] = Form(None)
) -> dict[str, Any]:
    # Verify account
    customer = db.verify_customer(account_number)
    if not customer:
        raise HTTPException(status_code=401, detail="Invalid account number")
        
    if db.is_user_blocked(account_number):
        raise HTTPException(status_code=403, detail="Account is blocked")
    
    # Save audio temporarily
    fd, path = tempfile.mkstemp(suffix=".webm")
    try:
        with open(path, "wb") as f:
            content = await audio.read()
            f.write(content)
            
        client = get_client()
        with open(path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(audio.filename or "audio.webm", f.read()),
                model="whisper-large-v3",
                response_format="json",
            )
            
        text = transcription.text
    finally:
        os.close(fd)
        os.unlink(path)
        
    # Check spam
    spam_result = check_spam(text)
    if not spam_result.get("is_valid", True):
        db.record_spam_strike(account_number)
        raise HTTPException(
            status_code=400, 
            detail={
                "message": f"Please input a valid banking complaint. Spam filter reason: {spam_result.get('reason')}",
                "transcribed_text": text
            }
        )
        
    payload = {
        "complaint_text": text,
        "customer_name": customer_name,
        "channel": "portal",
        "language": "english", # Intake will fix if it's Hindi/Marathi natively spoken
        "account_type": account_type or "savings",
        "location": location,
        "amount_involved": amount_involved,
    }
    
    new_id = ingest_new_complaint(payload)
    try:
        result = process_one_streaming(new_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
        
    return {"id": new_id, "transcribed_text": text, **result}




@app.get("/complaints")
def list_complaints(
    category: Optional[str] = Query(None, description="Filter by category"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    channel: Optional[str] = Query(None, description="Filter by channel"),
    customer: Optional[str] = Query(None, description="Filter by customer name"),
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    """List complaints with optional filters."""
    db.init_db()
    clauses, params = [], []
    if category: clauses.append("category = ?"); params.append(category)
    if severity: clauses.append("severity = ?"); params.append(severity)
    if channel:  clauses.append("channel = ?");  params.append(channel)
    if customer: clauses.append("customer_name = ?"); params.append(customer)
    where = " AND ".join(clauses) if clauses else None
    rows = db.list_complaints(limit=limit, where=where, params=params)
    return {"count": len(rows), "complaints": rows}


@app.get("/stats")
def stats() -> dict[str, Any]:
    """Dashboard KPI stats: total / processed / at-risk / duplicates / by-category."""
    db.init_db()
    rows = db.list_complaints()
    df = pd.DataFrame(rows)
    if df.empty:
        return {"total": 0, "processed": 0, "at_risk": 0, "duplicates": 0,
                "by_category": {}, "by_severity": {}, "by_channel": {}}
    processed = int(df["processed_at"].notna().sum())
    at_risk = int((df["sla_breach_prob"].fillna(0) >= 0.5).sum())
    duplicates = int(df["duplicate_of"].notna().sum())
    avg_breach = float(df["sla_breach_prob"].dropna().mean() or 0.0)
    avg_risk = float(df["risk_score"].dropna().mean() or 0.0)
    return {
        "total": len(df),
        "processed": processed,
        "pending": len(df) - processed,
        "at_risk": at_risk,
        "duplicates": duplicates,
        "avg_breach_probability": round(avg_breach, 3),
        "avg_risk_score": round(avg_risk, 1),
        "by_category": df["category"].fillna("Unprocessed").value_counts().to_dict(),
        "by_severity": df["severity"].fillna("Unprocessed").value_counts().to_dict(),
        "by_channel": df["channel"].value_counts().to_dict(),
        "root_cause_alerts": len(db.list_root_cause_alerts()),
    }


@app.get("/report", response_model=None)
def compliance_report(
    format: str = Query("csv", regex="^(csv|json)$"),
    today: Optional[date] = Query(None),
) -> Response | dict[str, Any]:
    """RBI compliance report. `format=csv` (default) returns a downloadable file;
    `format=json` returns the rows + summary inline."""
    db.init_db()
    rows = db.list_complaints()
    if format == "csv":
        csv_bytes = rbi_report.to_csv(rows, today=today)
        filename = f"rbi_compliance_{(today or date.today()).isoformat()}.csv"
        return Response(
            content=csv_bytes, media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    df = rbi_report.build_report(rows, today=today)
    return {
        "summary": rbi_report.summary_stats(rows, today=today),
        "report":  df.astype(object).where(df.notna(), None).to_dict(orient="records"),
    }
