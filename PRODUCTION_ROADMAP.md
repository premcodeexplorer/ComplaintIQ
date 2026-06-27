# ComplaintIQ — Production Readiness Roadmap

> **Context:** Round 3 of PSBs Hackathon Series 2026 / iDEA 2.0 (PS5 — Union Bank of India).  
> Moving from a working hackathon demo to a production-grade complaint intelligence platform.

---

## Gap Analysis — Current State vs Production Need

| Area | Current State | Production Need |
|------|--------------|-----------------|
| Database | SQLite local file | Cloud PostgreSQL + pooling + migrations |
| Channel ingestion | Demo Streamlit form only | Real webhooks for 6 channels |
| Authentication | None | JWT / OAuth2 on all surfaces |
| Processing | Synchronous, blocks on LLM | Async queue (Celery + Redis) |
| Vector store | Local ChromaDB directory | Cloud-hosted (Qdrant / Pinecone) |
| Logging | `print()` statements | Structured JSON logs + error tracking |
| Secrets | `.env` file | Secrets manager / platform env vars |
| Scaling | Single process | Containerized, horizontally scalable |
| ML ops | Manual retrain scripts | Versioned artifacts + drift detection |
| RBI compliance | CSV export only | Full immutable audit trail |

---

## Priority Order

| Priority | Change | Effort | Impact |
|----------|--------|--------|--------|
| P0 | Cloud DB (Supabase PostgreSQL) | 1 day | Unblocks everything |
| P0 | Async queue (Celery + Redis) | 1 day | Handles real load |
| P0 | Basic auth on dashboard + API | 4 hrs | Security baseline |
| P1 | Gmail webhook integration | 1 day | Real channel demo |
| P1 | WhatsApp webhook (Twilio / Meta) | 1 day | Real channel demo |
| P1 | Structured logging + Sentry | 4 hrs | Observability |
| P1 | Docker + docker-compose | 4 hrs | Reproducible deploy |
| P2 | Twitter/X filtered stream | 1 day | Complete channel set |
| P2 | Phone call (Twilio + Whisper ASR) | 2 days | Impressive live demo |
| P2 | Rate limiting + input sanitization | 4 hrs | Security hardening |
| P2 | LLM response caching (Redis) | 4 hrs | Cost + performance |
| P3 | Alembic migrations | 4 hrs | DB ops maturity |
| P3 | MLflow model versioning | 1 day | ML ops maturity |
| P3 | Audit log table | 4 hrs | RBI compliance |
| P3 | RBAC (3 roles) | 2 days | Access control |

---

## 1. Database — SQLite → Cloud PostgreSQL

### Why it must change
SQLite has no concurrent write support, no network access, no backup tooling, and will corrupt under load. A real bank handles hundreds of complaints per hour.

### Recommended stack
- **Supabase** (free tier, built on Postgres, has built-in auth + row-level security)  
- **Neon** (serverless Postgres, also free tier, good for variable load)

### 1.1 Swap the DB layer (`database/db.py`)

```python
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool

DATABASE_URL = os.getenv("DATABASE_URL")  # postgres://user:pass@host/db
_pool = pool.ThreadedConnectionPool(2, 20, DATABASE_URL)

from contextlib import contextmanager

@contextmanager
def connect():
    conn = _pool.getconn()
    conn.cursor_factory = RealDictCursor
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)
```

**SQL compatibility changes:**
- `?` placeholders → `%s`
- `INSERT OR IGNORE` → `INSERT ... ON CONFLICT DO NOTHING`
- `executescript()` → individual `execute()` calls

### 1.2 Add Alembic for migrations (replaces `ensure_schema()`)

```bash
pip install alembic
alembic init alembic
# Each schema change becomes a versioned migration file
# alembic upgrade head  — runs automatically on deploy
```

### 1.3 Move ChromaDB → Qdrant Cloud

ChromaDB is a local directory — it cannot be shared across containers or workers.  
Qdrant Cloud has a free tier and a near-identical Python API.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
```

---

## 2. Real Channel Integrations (6 Channels)

**Architecture pattern:** Each channel has an inbound webhook / poller that normalizes the message and calls `ingest_new_complaint()`, then enqueues the complaint for async processing. The existing pipeline handles the rest.

```
Channel webhook → normalize → ingest_new_complaint() → Celery queue → process_one()
```

### Channel 1 — Gmail

```python
# Gmail API with push notifications via Google Pub/Sub
# Setup: Enable Gmail API → Create Pub/Sub topic → Point subscription to your webhook

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import base64, email

@app.post("/webhook/gmail")
async def gmail_webhook(payload: dict):
    raw = base64.urlsafe_b64decode(payload["message"]["data"])
    msg = email.message_from_bytes(raw)
    body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
    sender_name = email.utils.parseaddr(msg["From"])[0]

    new_id = ingest_new_complaint({
        "complaint_text": body,
        "customer_name": sender_name,
        "channel": "email",
        "language": "english",
    })
    process_complaint_task.delay(new_id)
    return {"status": "queued", "id": new_id}
```

**Library:** `google-api-python-client`, `google-auth`

### Channel 2 — WhatsApp

```python
# Meta WhatsApp Business API (free: 1000 conversations/month)
# OR Twilio WhatsApp (simpler setup)

@app.post("/webhook/whatsapp")
async def whatsapp_webhook(payload: dict):
    entry = payload["entry"][0]["changes"][0]["value"]
    message = entry["messages"][0]
    sender_name = entry["contacts"][0]["profile"]["name"]
    text = message["text"]["body"]

    new_id = ingest_new_complaint({
        "complaint_text": text,
        "customer_name": sender_name,
        "channel": "whatsapp",
    })
    process_complaint_task.delay(new_id)

    # Acknowledge receipt back to customer
    send_whatsapp_reply(message["from"], f"Thank you. Your complaint has been registered: {new_id}")
    return {"status": "ok"}
```

**Library:** `pywa` (Python WhatsApp wrapper) or direct Meta Graph API calls

### Channel 3 — Twitter / X

```python
import tweepy

class ComplaintStream(tweepy.StreamingClient):
    def on_tweet(self, tweet):
        new_id = ingest_new_complaint({
            "complaint_text": tweet.text,
            "customer_name": str(tweet.author_id),
            "channel": "twitter",
        })
        process_complaint_task.delay(new_id)

# Run as a separate long-running process / Celery task
client = ComplaintStream(bearer_token=os.getenv("X_BEARER_TOKEN"))
client.add_rules(tweepy.StreamRule("@UnionBankOfIndia -is:retweet"))
client.filter()
```

**Library:** `tweepy`

### Channel 4 — Phone Calls (Voice → Text)

```python
# Twilio Voice records the call → POSTs audio URL to your webhook
# Groq Whisper transcribes it

from groq import Groq
import requests

@app.post("/webhook/call")
async def call_webhook(RecordingUrl: str = Form(), Caller: str = Form()):
    audio_bytes = requests.get(RecordingUrl + ".mp3",
                               auth=(TWILIO_SID, TWILIO_TOKEN)).content

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    transcript = groq_client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=("call.mp3", audio_bytes),
        language="en",
    )

    new_id = ingest_new_complaint({
        "complaint_text": transcript.text,
        "customer_name": Caller,
        "channel": "phone_call",
    })
    process_complaint_task.delay(new_id)
    return PlainTextResponse('<?xml version="1.0"?><Response><Say>Thank you. Your complaint has been registered.</Say></Response>')
```

**Library:** `twilio`, `groq` (already in requirements)

### Channel 5 — Bank Portal / Branch

Your existing Streamlit form and FastAPI `POST /complaint` already serve this channel.  
For production:
- Replace the Streamlit form with a proper React/Next.js form
- Branch tablets / kiosks POST to `POST /complaint` with `channel="branch"`
- Auto-populate `location` from branch code in the request header

### Channel 6 — Mobile App

`POST /complaint` already handles this with `channel="mobile_app"`.  
Add push notification acknowledgement:

```python
import firebase_admin
from firebase_admin import messaging

def send_push_notification(device_token: str, complaint_id: str):
    message = messaging.Message(
        notification=messaging.Notification(
            title="Complaint Registered",
            body=f"Your complaint {complaint_id} has been registered. We'll respond within SLA.",
        ),
        token=device_token,
    )
    firebase_admin.messaging.send(message)
```

**Library:** `firebase-admin`

---

## 3. Async Processing Queue

### Why it is critical

`POST /complaint` currently blocks for **8–15 seconds** (3 Groq LLM calls + ChromaDB + model inference). A real bank handles hundreds of complaints per hour — this will time out for every channel webhook.

### Stack: Celery + Redis

**`pipeline/tasks.py`** (new file)
```python
from celery import Celery
import os

celery_app = Celery(
    "complaintiq",
    broker=os.getenv("REDIS_URL"),
    backend=os.getenv("REDIS_URL"),
)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_expires = 3600

@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def process_complaint_task(self, complaint_id: str):
    try:
        from pipeline.orchestrator import process_one
        process_one(complaint_id)
    except Exception as exc:
        raise self.retry(exc=exc)
```

**`api/main.py`** — now returns 202 Accepted immediately
```python
@app.post("/complaint", status_code=202)
def submit_complaint(payload: ComplaintIn):
    new_id = ingest_new_complaint(payload.model_dump())
    process_complaint_task.delay(new_id)   # fire and forget
    return {"id": new_id, "status": "queued"}

@app.get("/complaint/{complaint_id}/status")
def complaint_status(complaint_id: str):
    row = db.get_complaint(complaint_id)
    if not row:
        raise HTTPException(404)
    return {
        "id": complaint_id,
        "processed": row["processed_at"] is not None,
        "status": row["status"],
        "category": row.get("category"),
        "sla_due_date": row.get("sla_due_date"),
    }
```

Start worker:
```bash
celery -A pipeline.tasks worker --loglevel=info --concurrency=4
```

---

## 4. Security

> Currently there is **zero authentication** on any surface. This is the most critical gap for a production banking system.

### 4.1 FastAPI — JWT Authentication

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException
import jwt, os

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# Protect all endpoints
@app.post("/complaint")
def submit_complaint(payload: ComplaintIn, user=Depends(verify_token)):
    ...
```

### 4.2 Streamlit Dashboard — Password Gate

```python
# dashboard/app.py — add at the top of main()
def check_password() -> bool:
    def _on_submit():
        if st.session_state["password"] == st.secrets.get("DASHBOARD_PASSWORD", ""):
            st.session_state["auth"] = True
        else:
            st.session_state["auth"] = False
            st.session_state["auth_error"] = True

    if st.session_state.get("auth"):
        return True

    st.text_input("Dashboard password", type="password",
                  on_change=_on_submit, key="password")
    if st.session_state.get("auth_error"):
        st.error("Incorrect password.")
    return False

if not check_password():
    st.stop()
```

### 4.3 Role-Based Access Control (RBAC)

Three roles for a bank deployment:

| Role | Permissions |
|------|------------|
| `agent` | View live feed, submit feedback |
| `manager` | Agent + analytics, model performance |
| `admin` | Everything + trigger retraining, export reports |

```sql
-- Add to schema
CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('agent', 'manager', 'admin')),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

```python
# FastAPI dependency
def require_role(role: str):
    def checker(user=Depends(verify_token)):
        role_order = {"agent": 1, "manager": 2, "admin": 3}
        if role_order.get(user["role"], 0) < role_order.get(role, 99):
            raise HTTPException(403, "Insufficient permissions")
        return user
    return checker

@app.get("/report")
def compliance_report(_=Depends(require_role("manager"))):
    ...
```

### 4.4 Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/complaint")
@limiter.limit("10/minute")
def submit_complaint(request: Request, payload: ComplaintIn):
    ...
```

### 4.5 Input Sanitization

```python
import html, re

def sanitize_complaint_text(text: str) -> str:
    text = text.strip()[:4000]                # max length
    text = html.escape(text)                  # XSS prevention
    return text

def mask_pii_for_logs(text: str) -> str:
    text = re.sub(r'\b\d{10}\b', '[PHONE]', text)
    text = re.sub(r'\b\d{12,16}\b', '[ACCT_NO]', text)
    text = re.sub(r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}',
                  '[EMAIL]', text, flags=re.I)
    return text
```

### 4.6 Secrets Management

Never store API keys in `.env` committed to git. Use:
- **Streamlit Cloud:** `st.secrets` (already done)
- **Render / Railway:** Platform environment variables (injected at runtime)
- **AWS / GCP:** AWS Secrets Manager / GCP Secret Manager for enterprise

---

## 5. Observability & Monitoring

### 5.1 Replace all `print()` with structured logging

```python
# utils/logger.py  (new file)
import logging, json, os

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("complaint_id", "customer_name", "agent", "latency_ms"):
            if hasattr(record, key):
                log[key] = getattr(record, key)
        return json.dumps(log, ensure_ascii=False)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.basicConfig(handlers=[handler], level=logging.INFO)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"complaintiq.{name}")
```

Usage in every agent/module:
```python
from utils.logger import get_logger
logger = get_logger("classifier")
logger.info("classified", extra={"complaint_id": cid, "category": cat, "latency_ms": 234})
```

### 5.2 Sentry for error tracking (free tier)

```python
# At the top of api/main.py and dashboard/app.py
import sentry_sdk
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    traces_sample_rate=0.1,
    environment=os.getenv("ENV", "development"),
)
```

All unhandled exceptions are automatically captured with full stack trace and sent to your Sentry dashboard.

### 5.3 LLM cost + latency tracking

```python
# agents/llm_client.py — wrap the existing chat() function
import time
from utils.logger import get_logger
logger = get_logger("llm_client")

def chat(prompt: str, **kwargs) -> str:
    start = time.monotonic()
    resp = client.chat.completions.create(...)
    latency_ms = round((time.monotonic() - start) * 1000)
    tokens = resp.usage.total_tokens
    cost_usd = tokens * 0.00000059  # llama-3.3-70b on Groq pricing

    logger.info("groq_call", extra={
        "model": kwargs.get("model", DEFAULT_MODEL),
        "latency_ms": latency_ms,
        "total_tokens": tokens,
        "estimated_cost_usd": round(cost_usd, 6),
    })
    return resp.choices[0].message.content or ""
```

### 5.4 Health check endpoint

```python
# api/main.py
@app.get("/health")
def health():
    checks = {}
    try:
        db.list_complaints(limit=1)
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"

    try:
        import redis as _redis
        r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        r.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks, "version": "1.0.0"}
```

---

## 6. Containerization & CI/CD

### 6.1 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps for LightGBM + psycopg2
RUN apt-get update && apt-get install -y \
    libgomp1 libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: dashboard. Override in docker-compose for API / worker.
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
```

### 6.2 docker-compose.yml (local development)

```yaml
version: "3.9"

services:
  dashboard:
    build: .
    ports: ["8501:8501"]
    env_file: .env
    depends_on: [postgres, redis]
    volumes: ["./data:/app/data"]   # mount pre-seeded DB for local dev

  api:
    build: .
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [postgres, redis]

  worker:
    build: .
    command: celery -A pipeline.tasks worker --loglevel=info --concurrency=4
    env_file: .env
    depends_on: [postgres, redis]

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: complaintiq
      POSTGRES_USER: complaintiq
      POSTGRES_PASSWORD: secret
    volumes: ["pgdata:/var/lib/postgresql/data"]
    ports: ["5432:5432"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

volumes:
  pgdata:
```

### 6.3 GitHub Actions CI

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: complaintiq_test
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        ports: ["5432:5432"]

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
      - run: pip install -r requirements.txt pytest pytest-cov
      - run: pytest tests/ -v --cov=. --cov-report=term-missing
        env:
          DATABASE_URL: postgresql://test:test@localhost/complaintiq_test
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
```

---

## 7. Performance Optimizations

### 7.1 LLM Response Caching (Redis)

Same complaint text hitting the API twice should not call Groq twice — saves cost and latency.

```python
# agents/llm_client.py — add caching wrapper
import hashlib
import redis as _redis

_cache: _redis.Redis | None = None

def _get_cache():
    global _cache
    if _cache is None:
        url = os.getenv("REDIS_URL")
        if url:
            _cache = _redis.from_url(url)
    return _cache

def chat_cached(prompt: str, **kwargs) -> str:
    cache = _get_cache()
    if cache:
        key = "llm:" + hashlib.md5(prompt.encode()).hexdigest()
        cached = cache.get(key)
        if cached:
            return cached.decode()
    result = chat(prompt, **kwargs)
    if cache:
        cache.setex(key, 3600, result)  # cache for 1 hour
    return result
```

### 7.2 Parallel LLM Calls with asyncio

The 3 independent LLM calls (intake, classify, draft) currently run sequentially.  
With `asyncio.gather()` they can run in parallel, cutting per-complaint latency by ~60%.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=8)

async def process_parallel(complaint: dict) -> dict:
    loop = asyncio.get_event_loop()

    # Intake and Classify are independent — run in parallel
    intake_fut = loop.run_in_executor(_executor, intake.extract, complaint)
    cls_fut    = loop.run_in_executor(_executor, classifier.classify, complaint)
    intake_out, cls = await asyncio.gather(intake_fut, cls_fut)

    # Draft depends on classify result
    complaint.update(cls)
    draft = await loop.run_in_executor(_executor, response_drafter.draft, complaint)

    return {"intake": intake_out, "classification": cls, "draft": draft}
```

### 7.3 Dashboard Pagination

Currently loads all 1000 rows on every 20-second refresh. At scale this is thousands of rows.

```python
# database/db.py — add offset support
def list_complaints(limit=100, offset=0, where=None, params=()):
    q = "SELECT * FROM complaints"
    if where:
        q += " WHERE " + where
    q += " ORDER BY date DESC, id DESC"
    q += f" LIMIT {int(limit)} OFFSET {int(offset)}"
    ...

# dashboard/app.py
@st.cache_data(ttl=20)
def load_complaints_page(page: int = 0, page_size: int = 100):
    return db.list_complaints(limit=page_size, offset=page * page_size)
```

### 7.4 Preload ML Models at API Startup

Currently each model lazy-loads on the first call, causing cold-start latency spikes.

```python
# api/main.py
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload all ML models at startup (not on first request)
    from agents import ml_category, sla_monitor, priority
    ml_category._get()
    sla_monitor._artefact_cache()
    priority._get()
    print("ML models preloaded.")
    yield

app = FastAPI(lifespan=lifespan, ...)
```

---

## 8. ML Ops

### 8.1 Model Versioning with MLflow

```python
# models/train_sla_model.py — add after training
import mlflow
import mlflow.sklearn

mlflow.set_experiment("sla-breach-predictor")

with mlflow.start_run(run_name=f"xgboost-tuned-{datetime.utcnow().date()}"):
    mlflow.log_param("n_estimators", xgb_best["n_estimators"])
    mlflow.log_param("max_depth", xgb_best["max_depth"])
    mlflow.log_metric("cv_auc", leaderboard[0]["cv_auc_mean"])
    mlflow.log_metric("holdout_auc", holdout_auc)
    mlflow.sklearn.log_model(winner_pipe, "sla_model",
                              registered_model_name="sla_breach_predictor")
```

### 8.2 Scheduled Retraining (Celery Beat)

```python
# pipeline/tasks.py
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    "retrain-sla-weekly": {
        "task": "pipeline.tasks.retrain_sla_model",
        "schedule": crontab(hour=2, minute=0, day_of_week=0),  # Sunday 2AM
    },
}

@celery_app.task
def retrain_sla_model():
    from models.train_sla_model import main as train
    train()
    logger.info("SLA model retrained successfully.")
```

### 8.3 Data Drift Detection (Evidently)

```python
# pipeline/drift_check.py
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd
from database import db

def check_drift():
    all_rows = db.list_complaints(where="processed_at IS NOT NULL")
    df = pd.DataFrame(all_rows)

    # Reference = first 800 rows (training data period)
    # Current   = last 30 days of complaints
    reference = df.head(800)
    current   = df[df["date"] >= (pd.Timestamp.now() - pd.Timedelta(days=30)).date().isoformat()]

    if len(current) < 30:
        return  # not enough data yet

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[["category", "severity", "sentiment", "amount_involved"]],
               current_data=current[["category", "severity", "sentiment", "amount_involved"]])

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    if drift_detected:
        logger.warning("Data drift detected — scheduling model retraining.")
        retrain_sla_model.delay()
```

---

## 9. RBI Compliance & Audit Trail

For a production banking system, RBI requires an **immutable audit log** of every action on every complaint.

### 9.1 Audit log table

```sql
CREATE TABLE audit_log (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    actor           TEXT NOT NULL,          -- user_id or "system/agent_name"
    action          TEXT NOT NULL,          -- "classify" | "update_status" | "feedback" | "view"
    complaint_id    TEXT REFERENCES complaints(id),
    before_state    JSONB,
    after_state     JSONB,
    ip_address      TEXT,
    user_agent      TEXT
);
-- This table is append-only: no UPDATE or DELETE permissions granted on it
```

### 9.2 Automatic audit logging

```python
# database/db.py — wrap update_complaint()
def update_complaint(complaint_id: str, actor: str = "system", **fields):
    before = get_complaint(complaint_id)
    # ... existing update logic ...
    after = get_complaint(complaint_id)
    _write_audit(actor, "update_complaint", complaint_id, before, after)

def _write_audit(actor, action, complaint_id, before, after):
    with connect() as c:
        c.execute(
            "INSERT INTO audit_log (actor, action, complaint_id, before_state, after_state) "
            "VALUES (%s, %s, %s, %s, %s)",
            (actor, action, complaint_id,
             json.dumps(before), json.dumps(after))
        )
```

---

## 10. Recommended Deployment Architecture

```
                        ┌─────────────────────────────────────┐
                        │         Channel Webhooks             │
                        │  Gmail │ WhatsApp │ Twitter │ Twilio │
                        └──────────────┬──────────────────────┘
                                       │ POST /webhook/<channel>
                        ┌──────────────▼──────────────────────┐
                        │         FastAPI (api/main.py)        │
                        │  Auth │ Rate Limiting │ Validation   │
                        └──────────────┬──────────────────────┘
                                       │ enqueue
                        ┌──────────────▼──────────────────────┐
                        │         Redis (message broker)       │
                        └──────────────┬──────────────────────┘
                                       │ consume
                        ┌──────────────▼──────────────────────┐
                        │     Celery Workers (N instances)     │
                        │  6 agents + 4 ML models per task    │
                        └──────────────┬──────────────────────┘
                                       │ read/write
                   ┌───────────────────┼───────────────────────┐
                   │                   │                       │
       ┌───────────▼─────┐  ┌──────────▼───────┐  ┌──────────▼──────┐
       │  PostgreSQL      │  │  Qdrant Cloud    │  │  Groq API       │
       │  (Supabase)      │  │  (embeddings)    │  │  (LLM calls)    │
       └─────────────────┘  └──────────────────┘  └─────────────────┘
                   │
       ┌───────────▼──────────────────────────────────────────────┐
       │         Streamlit Dashboard (dashboard/app.py)            │
       │  9 tabs │ KPI strip │ Alert banners │ India map           │
       └──────────────────────────────────────────────────────────┘
```

---

## 11. New Files to Create

```
ComplaintIQ/
├── pipeline/
│   └── tasks.py                  # Celery task definitions
├── api/
│   └── channels/
│       ├── __init__.py
│       ├── gmail.py              # Gmail webhook handler
│       ├── whatsapp.py           # WhatsApp webhook handler
│       ├── twitter.py            # Twitter stream consumer
│       └── calls.py              # Twilio voice webhook
├── utils/
│   ├── __init__.py
│   └── logger.py                 # Structured JSON logger
├── tests/
│   ├── test_agents.py
│   ├── test_pipeline.py
│   └── test_api.py
├── alembic/                      # DB migration versions
│   └── versions/
├── Dockerfile
├── docker-compose.yml
└── .github/
    └── workflows/
        └── ci.yml
```

---

## 12. New Dependencies to Add to `requirements.txt`

```text
# Cloud DB
psycopg2-binary==2.9.10
alembic==1.14.0

# Async queue
celery==5.4.0
redis==5.2.1
flower==2.0.1                    # Celery monitoring UI

# Security
python-jose[cryptography]==3.3.0  # JWT
passlib[bcrypt]==1.7.4
slowapi==0.1.9                    # Rate limiting

# Channel integrations
google-api-python-client==2.160.0
google-auth-httplib2==0.2.0
tweepy==4.15.0
twilio==9.4.0
pywa==2.6.0
firebase-admin==6.6.0

# Observability
sentry-sdk[fastapi]==2.21.0

# ML ops
mlflow==2.19.0
evidently==0.5.0

# Vector DB (cloud)
qdrant-client==1.13.3
```

---

*Generated for ComplaintIQ — AgentForge | PSBs Hackathon 2026 / iDEA 2.0*
