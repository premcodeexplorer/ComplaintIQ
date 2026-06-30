# ComplaintIQ — Complete Project Overview (for Final-Round Presentation)

> **Use this document to generate the final-round pitch deck.** It contains the
> full, current state of the project: what it does, the architecture, every
> component, the production work done since Round 2, the metrics, and suggested
> slide structure + talking points.

---

## 1. One-line pitch

**ComplaintIQ is an AI-powered, multi-channel customer-complaint intelligence
platform for Indian public-sector banks** — it ingests complaints from every
channel, runs them through a 6-agent AI pipeline + 4 trained ML models, and gives
bank staff a real-time dashboard for SLA tracking, customer-risk scoring, root-cause
detection, and RBI compliance.

- **Team:** AgentForge — Prem Baba, Purva Bhoyar, Pranil Bankar, Adhishree Shiledar
- **Event:** PSBs Hackathon Series 2026 / iDEA 2.0 — **PS5: Unified Customer
  Complaint Communication Dashboard (Union Bank of India)**
- **Status:** Shortlisted R1 → **TOP 30 in R2** → **Offline 24-hour Grand Finale (R3)**

---

## 2. The problem (PS5)

Indian banks receive complaints across many disconnected channels — email,
WhatsApp, Twitter/X, phone calls, branch walk-ins, the net-banking portal, and the
mobile app. Today these are siloed. The result:

- No single view of a customer's complaint history.
- **RBI SLA windows get breached** (UPI/ATM 5 days, Card/NetBanking 7 days, Loan/General 30 days) → regulatory penalties + Ombudsman escalations.
- Duplicate complaints get worked twice; systemic issues (e.g., an ATM network
  outage in one city) stay invisible until they blow up.
- Multilingual complaints (English / Hindi / Marathi) are hard to triage at scale.

**ComplaintIQ unifies all channels into one intelligent dashboard.**

---

## 3. What it does — the 6-agent AI pipeline

Every complaint flows through 9 stages (6 LLM/AI agents + 3 ML second-opinions):

| # | Agent | Technology | Output |
|---|-------|-----------|--------|
| 1 | **Intake** | Groq `llama-3.3-70b-versatile` | Structured fields from raw text (EN / HI / MR) |
| 2 | **Classifier** | Groq LLM | category / severity / sentiment |
| 3 | **Duplicate Detector** | `sentence-transformers all-MiniLM-L6-v2` + vector search | duplicate flag + similarity score (per-customer) |
| 4 | **Response Drafter** | Groq LLM | policy-compliant reply in the customer's language |
| 5 | **SLA Monitor** | XGBoost (tuned) | due date + breach probability |
| 6 | **Root Cause** | KMeans on embeddings | systemic clusters with top-3 city hotspots |

Plus **4 trained ML models** that run alongside the agents as independent
"second opinions" and feed the composite scoring:

| Model | Algorithm | Metric |
|-------|-----------|--------|
| **SLA breach predictor** | Bake-off: RF / GBM / **XGBoost(tuned)** / LightGBM / Stacking, 5-fold StratifiedCV + SMOTE | **CV AUC 0.9233 ± 0.018, Hold-out AUC 0.9378** |
| **Category classifier** | TF-IDF + Logistic Regression | **97% accuracy, 98% LLM agreement** |
| **Sentiment model** | HuggingFace `cardiffnlp/twitter-roberta-base-sentiment-latest` | local inference |
| **Priority scorer** | Gradient Boosting | **R² 0.997, MAE 0.50** |

### Customer Risk Score (0–100) — explainable, 3 sub-scores
- **RBI Ombudsman escalation risk** (45% weight)
- **Customer churn risk** (30%)
- **Social-media blow-up risk** (25%) — Twitter/WhatsApp public-pressure signal

### Auto-resolution logic
If `severity ∈ {Low, Medium}` **and** `sentiment == Polite` **and** the category has
a standard template → the complaint is **Auto-Resolved (Standard Reply Sent)** without
consuming an agent. Duplicates → **Auto-Resolved (Duplicate)**. Together this cleared
**186 / 1000 (18.6%)** of the workload automatically — a direct staff-hours saving.

---

## 4. Architecture (current, production-grade)

```
                        ┌──────────────────────────────────────────────┐
   CUSTOMER CHANNELS    │                INGESTION                      │
                        │                                               │
  ┌───────────────┐     │   ┌────────────────────┐                      │
  │ Public Portal │────────▶│ Netlify serverless │──┐                   │
  │ (React+Vite)  │     │   │  function          │  │  insert            │
  └───────────────┘     │   └────────────────────┘  │                   │
                        │                            ▼                   │
  ┌───────────────┐     │   ┌────────────────────┐  ┌──────────────────┐│
  │ Gmail inbox   │────────▶│ Gmail IMAP poller  │─▶│  Supabase        ││
  │ (real email)  │     │   │  (worker, 30s)     │  │  PostgreSQL      ││
  └───────────────┘     │   └────────────────────┘  │  + pgvector      ││
                        │                            └────────┬─────────┘│
                        └─────────────────────────────────────┼──────────┘
                                                               │
                        ┌──────────────────────────────────────▼─────────┐
                        │   9-STAGE AI PIPELINE (orchestrator.py)         │
                        │   6 Groq/ML agents + 3 ML second-opinions       │
                        └──────────────────────────────────────┬─────────┘
                                                               │
        ┌──────────────────────────────────────────────────────▼─────────┐
        │  BANK STAFF SURFACES                                            │
        │  • Streamlit dashboard (9 tabs, auth-gated)                     │
        │  • FastAPI service (REST API + OpenAPI docs)                    │
        │  • RBI compliance CSV export                                    │
        └────────────────────────────────────────────────────────────────┘
```

### Stack
- **Language:** Python 3.11 (backend), React 19 + Vite (public portal)
- **LLM:** Groq API, `llama-3.3-70b-versatile`
- **Embeddings:** `sentence-transformers all-MiniLM-L6-v2` (local, 384-dim)
- **Vector store:** **pgvector** (in Postgres) — was local ChromaDB
- **Classical ML:** scikit-learn (RF, GBM, LogReg, KMeans, TF-IDF, Stacking) + XGBoost + LightGBM
- **Deep learning:** HuggingFace Transformers + PyTorch (Roberta sentiment)
- **Class balance:** imbalanced-learn (SMOTE inside an imblearn Pipeline)
- **Database:** **Supabase (cloud PostgreSQL)** — code keeps a SQLite fallback via a
  `DATABASE_URL` toggle for offline/local demos
- **Auth:** **Supabase Auth** (email + password, admin role, RLS)
- **Dashboard:** Streamlit + Plotly + Folium + Plotly geo (`scatter_geo` + `choropleth`)
- **API:** FastAPI + uvicorn
- **Public portal:** React + Vite + `@supabase/supabase-js`, deployed on **Netlify**
  (serverless function + IP rate-limiting)
- **Worker deploy:** Railway / Procfile (`worker: python -m channels.gmail_poller`)

---

## 5. What's NEW since Round 2 (the production-readiness story)

> This is the heart of the finale: R3 rewards **product maturity, scalability,
> and production thinking** layered onto the working prototype. Everything below was
> built/added since the TOP-30 round.

### ✅ Cloud database — SQLite → Supabase PostgreSQL
- `database/db.py` now supports **both** SQLite (offline demo) and cloud Postgres,
  switched purely by a `DATABASE_URL` env var. Same code path, dual dialect (`?` vs `%s`).
- Versioned SQL migrations in `migrations/` (`001_initial_schema.sql`,
  `002_auth_schema.sql`) + `apply_migrations.py`. Idempotent `ensure_schema()` for
  additive columns.

### ✅ pgvector — cloud-native vector search
- Embeddings (`vector(384)`) now live **inside Postgres** via the `pgvector`
  extension, replacing the local ChromaDB directory. One database for everything —
  no separate vector infra to host.

### ✅ Authentication & access control
- **Supabase Auth** with email + password, `user_profiles` table, `role='admin'`.
- Streamlit dashboard is now **login-gated** (`render_login_screen()`), with
  **cookie-based session persistence across reloads** (`streamlit-cookies-controller`).
- Admin accounts are created CLI-only (`scripts/create_admin.py`) — never exposed to
  the browser. Service-role key used server-side only.

### ✅ Public Complaint Portal (brand-new customer-facing app)
- A separate **React 19 + Vite** single-page app (`public-portal/`) — "Secure,
  Anonymous, and Direct Communication Channel" — where customers file complaints.
- Submits via a **Netlify serverless function** (`submit-complaint.js`) that writes
  straight into Supabase, with **IP-based rate-limiting (3 complaints / 24h)** and
  server-side validation. Complaints get a `PORTAL-xxxxxxxx` ID and `channel='portal'`.
- This is a **real second live channel** alongside Gmail — proves the multi-channel
  claim end-to-end.

### ✅ Real live channel — Gmail IMAP poller
- `channels/gmail_poller.py` polls a real Gmail inbox every 30s using only the Python
  stdlib (`imaplib`), normalizes each email into a complaint, and runs it through the
  full 9-stage pipeline. Sender-blocklist filters out Google/marketing noise.
- Runs as a **Railway worker** (`Procfile`). Demo: send an email → watch it appear on
  the dashboard within 30 seconds.

### ✅ PII masking for all outbound LLM calls (privacy/RBI compliance)
- `agents/pii.py` — reversible token masking for Indian PII patterns (mobile, account,
  card, Aadhaar, PAN, IFSC, UPI VPA, email, customer names). Enforced centrally in
  `agents/llm_client.py`: mask before sending to Groq, unmask the reply locally.
- `PII_MASKING=0` env toggle to demo before/after live. 7/7 unit tests pass.

### ✅ SLA model upgrade — single RF → full bake-off
- `models/train_sla_model.py` now runs a **5-model bake-off** (RF / GBM / XGBoost-tuned
  / LightGBM / Stacking) with 5-fold StratifiedCV + SMOTE, saves a leaderboard, and the
  dashboard "Model performance" tab visualizes the winner + feature importances.

### ✅ Dataset scaled up — 200 → 1000 complaints
- 1000 synthetic complaints, Jan–May 2026, **7 channels**, 3 languages, 6 categories,
  62 cities with full lat/long, 86 duplicates detected, 914 LLM-drafted replies.

---

## 6. The dashboard — 9 tabs (bank-staff view)

| Tab | Contents |
|-----|----------|
| **Live feed** | 1000 complaints, filter by severity/category/channel, sort by Priority/Date/Risk/Breach prob. `(*)` marker when ML disagrees with the LLM. |
| **Customer** | Pick a customer → emotion-over-time chart + overall risk gauge + 3 sub-score bars + full history. |
| **India map** | Plotly geo — India-only, 36 states + 594 district borders, 62 cities sized by volume, top-4 labelled. |
| **SLA tracker** | Bucket bar chart (Overdue / <1d / <3d / <7d / >7d) + 25 most-urgent rows. |
| **Root cause** | 6 KMeans clusters with top-3 city hotspots, category chip, volume tier. |
| **Drafted replies** | Expandable LLM replies in the customer's language. |
| **Model performance** | SLA leaderboard, feature importance, confusion matrix, live LLM↔ML agreement, human-feedback accuracy. |
| **Analytics** | 8 Plotly charts (per-day, category, channel, sentiment, severity, breach-by-category, top cities, resolution status). |
| **Feedback** | Human-in-the-loop: Correct / Wrong buttons → corrections write back to the DB (retraining signal). |

**Always-visible:** KPI strip (Total / Processed / SLA at risk / Auto-resolved / Avg
breach prob); conditional alert banners (red CRITICAL >85% breach, amber SYSTEMIC
≥20-complaint cluster, sand ESCALATION risk >80); "Submit new complaint" live-pipeline
expander; sidebar with **RBI Compliance Report CSV download**.

---

## 7. FastAPI service (programmatic access)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/complaint` | submit a complaint, returns full pipeline result |
| `GET` | `/complaints` | list with filters (category, severity, channel, customer) |
| `GET` | `/stats` | dashboard KPIs |
| `GET` | `/report` | RBI compliance CSV (or `format=json`) |

OpenAPI docs at `/docs`. This is the integration surface for a real bank's core systems.

---

## 8. Key metrics to put on slides

- **SLA breach prediction:** CV AUC **0.9233 ± 0.018**, Hold-out AUC **0.9378**
- **Category classifier:** **97% accuracy**, **98%** agreement with the LLM
- **Priority scorer:** **R² 0.997**, MAE 0.50
- **Auto-resolution:** **18.6% (186/1000)** of complaints cleared without an agent
- **Scale demonstrated:** 1000 complaints · 7 channels · 3 languages · 6 categories · 62 cities
- **Duplicates caught:** 86 · **Replies drafted:** 914 · **Root-cause clusters:** 6
- **RBI SLA rules enforced:** UPI/ATM 5d, Card/NetBanking 7d, Loan/General 30d, with
  severity multipliers (Critical 0.5× … Low 1.2×)

---

## 9. Production roadmap — what we BUILT vs what we PRESENT

`PRODUCTION_ROADMAP.md` documents 12 production workstreams (gap analysis +
prioritized P0→P3 plan). Strategy for the finale:

**BUILT (live, demoable):**
Cloud Postgres (Supabase), pgvector vector search, auth + RLS + session cookies,
public React portal (real channel), Gmail live channel, PII masking, SLA model
bake-off, FastAPI, RBI CSV export, versioned migrations.

**PRESENTED (roadmap — proves we understand scale):**
Celery + Redis async queue, Qdrant/Pinecone managed vector store, full RBAC (3 roles),
Sentry observability + structured logging, Docker/compose, MLflow model versioning,
Evidently drift detection, immutable RBI audit trail, remaining 4 channel webhooks
(WhatsApp/Twitter/Phone-via-Whisper/Mobile).

This "built a vertical slice, planned the rest" narrative is the maturity signal R3
judges reward.

---

## 10. Why we win (judging criteria → our answer)

| Judging axis | Our story |
|--------------|-----------|
| **Innovation & relevance** | 6-agent AI + 4 ML models on a real RBI pain point; explainable risk score; auto-resolution |
| **Technical implementation** | Groq LLM + local embeddings + XGBoost bake-off + pgvector + FastAPI, all working end-to-end |
| **Product maturity** | Cloud DB, auth, two live channels, migrations, PII masking — not a toy demo |
| **Scalability / feasibility** | Supabase + serverless + worker architecture; documented roadmap to Celery/Qdrant/Docker |
| **UX / design** | 9-tab branded dashboard, India heat-map, live pipeline view, customer-facing portal |
| **Impact / alignment** | 18.6% auto-resolved, SLA breach prevention, Ombudsman-risk early warning, RBI compliance export |
| **Demo & presentation** | Send a live email → see it processed in 30s; submit from the public portal → appears on dashboard |

---

## 11. Suggested slide order (≈12 slides)

1. **Title** — ComplaintIQ, team AgentForge, PS5 Union Bank, "TOP 30 → Grand Finale"
2. **The problem** — siloed channels, RBI SLA breaches, Ombudsman risk, multilingual
3. **The solution** — one-line pitch + the unified-dashboard screenshot
4. **6-agent pipeline** — the table/diagram of agents 1–6
5. **4 ML models + metrics** — AUC 0.9233, 97% category, R² 0.997
6. **Customer Risk Score** — the 3 explainable sub-scores + auto-resolution (18.6%)
7. **Architecture diagram** — channels → ingestion → pipeline → surfaces
8. **What's new since R2** — cloud DB, auth, public portal, Gmail live channel, PII masking
9. **Live demo** — public portal submission + Gmail email → dashboard in 30s
10. **Dashboard tour** — India map, SLA tracker, root cause, model performance
11. **Production roadmap** — built (slice) vs presented (Celery/Qdrant/Docker/RBAC/MLflow)
12. **Impact & ask** — staff-hours saved, RBI compliance, scale path; thank-you + team

---

## 12. Repo layout (for reference)

```
ComplaintIQ/
├── agents/            6 AI agents + 4 ML inference modules + pii.py + llm_client.py
├── models/            trainer scripts + serialized .joblib artefacts + leaderboard
├── pipeline/          orchestrator (9 stages) + ml_backfill + data_remix
├── dashboard/         app.py (9-tab Streamlit, auth-gated) + rbi_report.py
├── api/               main.py (FastAPI, 4 endpoints)
├── auth/              supabase_auth.py (sign-in, profiles, admin creation)
├── channels/          gmail_poller.py (live IMAP channel → Railway worker)
├── database/          db.py (SQLite + Postgres dual-dialect)
├── migrations/        001_initial / 002_auth SQL + apply_migrations.py
├── scripts/           create_admin.py, migrate_to_supabase.py
├── public-portal/     React 19 + Vite app + Netlify serverless function
├── data/              complaints.json (1000), sla_rules.json, geojson, sqlite
├── tests/             test_pii.py (7/7)
├── Procfile           worker: python -m channels.gmail_poller
├── PRODUCTION_ROADMAP.md
├── requirements.txt
└── README.md
```

---

*Generated as a briefing for the final-round presentation. All metrics and components
verified against the current codebase.*
