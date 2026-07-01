# ComplaintIQ

**AI-powered unified customer complaint dashboard for Indian banks.**
Built for the **PSBs Hackathon Series 2026 / iDEA 2.0 — PS5 (Union Bank of India)** by team **AgentForge**.

ComplaintIQ ingests complaints from every channel (email, WhatsApp, Twitter, calls, branch, portal, mobile app), runs them through a **6-agent AI pipeline + 4 trained ML models**, and surfaces the result on a real-time Streamlit dashboard with SLA tracking, customer risk scoring, an India heat-map with state and district borders, and systemic root-cause alerts.

---

## What's inside

### 6 LLM / AI agents

| # | Agent | Tech | Output |
|---|-------|------|--------|
| 1 | **Intake** | Groq gpt-oss-120b | Structured fields from raw text (English / Hindi / Marathi) |
| 2 | **Classifier** | Groq LLM | category / severity / sentiment |
| 3 | **Duplicate Detector** | `sentence-transformers all-MiniLM-L6-v2` + ChromaDB cosine | duplicate flag + similarity (per-customer) |
| 4 | **Response Drafter** | Groq LLM | Policy-compliant reply in customer's language |
| 5 | **SLA Monitor** | XGBoost (tuned) | due date + breach probability |
| 6 | **Root Cause** | KMeans on embeddings | systemic clusters with top-3 city hotspots |

### 4 trained ML models (second opinions + composite scoring)

| Model | Algorithm | Metric |
|---|---|---|
| **SLA breach predictor** | RF / GBM / XGBoost(tuned) / LightGBM / Stacking — bake-off with 5-fold StratifiedCV + SMOTE | **CV AUC 0.9233 ± 0.018, Hold-out AUC 0.9378** |
| **Category classifier** | TF-IDF + Logistic Regression | 97% accuracy, 98% LLM agreement |
| **Sentiment model** | HF `cardiffnlp/twitter-roberta-base-sentiment-latest` | local inference, 48% LLM agreement (3-class vs 4-class) |
| **Priority scorer** | Gradient Boosting | R² 0.997, MAE 0.50 |

### Customer Risk Score (0–100) with explainable breakdown

- **RBI Ombudsman risk** (45% weight) — formal escalation likelihood
- **Customer churn risk** (30%) — likelihood customer leaves
- **Social-media blow-up risk** (25%) — Twitter / WhatsApp public-pressure signal

### Auto-resolution logic
If `severity ∈ {Low, Medium}` AND `sentiment == Polite` AND category has a standard template, the orchestrator marks the complaint **Auto-Resolved (Standard Reply Sent)** without consuming an agent. Duplicates get **Auto-Resolved (Duplicate)**. Together this cleared **186 / 1000 (18.6%)** of the workload.

---

## Stack

- Python 3.11
- **LLM** — Groq API, `gpt-oss-120b`
- **Embeddings** — `sentence-transformers all-MiniLM-L6-v2`
- **Vector store** — ChromaDB (persistent, local)
- **Classical ML** — scikit-learn (RF, GBM, LogReg, KMeans, TF-IDF, Stacking) + XGBoost + LightGBM
- **Deep learning** — HuggingFace `transformers` + PyTorch (Roberta sentiment)
- **Class balance** — imbalanced-learn (SMOTE inside imblearn Pipeline)
- **DB** — SQLite
- **Dashboard** — Streamlit + Plotly + Folium + Plotly geo (`scatter_geo` + `choropleth`)
- **API** — FastAPI + uvicorn
- **Deploys on** — Streamlit Cloud (free tier)

---

## Project layout

```
ComplaintIQ/
├── agents/                              The 6 AI agents + 4 ML inference modules
│   ├── intake.py                        Agent 1 (Groq)
│   ├── classifier.py                    Agent 2 (Groq, LLM)
│   ├── duplicate_detector.py            Agent 3 (sentence-transformers + ChromaDB)
│   ├── response_drafter.py              Agent 4 (Groq)
│   ├── sla_monitor.py                   Agent 5 (XGBoost tuned)
│   ├── root_cause.py                    Agent 6 (KMeans, top-3 city hotspots)
│   ├── risk_score.py                    Ombudsman / Churn / Social sub-scores
│   ├── ml_category.py                   TF-IDF + LogReg inference
│   ├── sentiment_ml.py                  HF Roberta inference
│   ├── priority.py                      GBM priority inference
│   └── llm_client.py                    Shared Groq client (retry + fallback)
├── models/                              Trainer scripts + serialized artefacts
│   ├── train_sla_model.py               Bake-off: RF / GBM / XGB(tuned) / LGBM / Stacking
│   ├── train_category_classifier.py     TF-IDF + LogReg
│   ├── train_priority_model.py          GBM
│   ├── sla_best_model.joblib            Winner (XGBoost tuned, CV AUC 0.9233)
│   ├── sla_rf.joblib                    Same bytes (legacy filename for sla_monitor)
│   ├── sla_leaderboard.json             Leaderboard + grid best params
│   ├── category_clf.joblib              97% accuracy
│   └── priority_gbm.joblib              R² 0.997
├── pipeline/
│   ├── orchestrator.py                  Chains all 9 stages, sync + streaming variants
│   ├── ml_backfill.py                   Apply ML models to LLM-processed rows
│   └── data_remix.py                    Date redistribution + resolution status migration
├── dashboard/
│   ├── app.py                           9-tab Streamlit UI with brand palette
│   └── rbi_report.py                    RBI compliance CSV generator
├── api/
│   └── main.py                          FastAPI: 4 endpoints
├── database/
│   └── db.py                            SQLite schema + helpers + feedback table
├── data/
│   ├── complaints.json                  1000 complaints (live dataset)
│   ├── sla_rules.json                   RBI SLA windows
│   ├── india_states.geojson             36 states/UTs (~1 MB)
│   ├── india_districts.geojson          594 districts (~1.2 MB, Douglas-Peucker simplified)
│   └── complaintiq.sqlite               Pre-processed DB (ships with repo)
├── .streamlit/
│   ├── config.toml                      Brand palette theme
│   └── secrets.toml.example             Sample for Streamlit Cloud secrets
├── .gitignore                           keeps .env + chroma_db + caches out
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Install deps (Python 3.11 recommended)
pip install -r requirements.txt

# 2. Add your Groq API key (free tier works)
#    Create a file named `.env` in the project root with these two lines:
#
#       GROQ_API_KEY=gsk_your_key_here
#       GROQ_MODEL=gpt-oss-120b
```

Get a free Groq key at https://console.groq.com.

---

## Run end-to-end

The repo ships with a pre-seeded `data/complaintiq.sqlite` and trained models, so you can skip straight to the dashboard:

```bash
# Just launch the dashboard
streamlit run dashboard/app.py
# Optional: also launch the FastAPI backend
uvicorn api.main:app --port 8000
```

To reprocess from scratch:

```bash
python -m database.db                     # seed SQLite from complaints.json
python -m models.train_sla_model          # 5-fold bake-off, saves leaderboard
python -m models.train_category_classifier
python -m models.train_priority_model
python -m pipeline.orchestrator           # runs the 6 agents on every complaint
python -m pipeline.ml_backfill            # adds ML category / sentiment / priority
streamlit run dashboard/app.py
```

The dashboard opens at **http://localhost:8501** with nine tabs:

| Tab | Contents |
|---|---|
| **Live feed** | 1000 complaints filterable by severity/category/channel, sortable by Priority / Date / Risk / Breach prob. `(*)` marker when ML disagrees with LLM. |
| **Customer** | Pick any customer → emotion-over-time bar chart + overall risk gauge + 3 sub-score bars (Ombudsman / Churn / Social) + complete history. |
| **India map** | Plotly geo render — India-only with all 36 states and 594 district borders. 62 cities plotted, sized by complaint count, coloured by volume tier. Top-4 cities permanently labelled. |
| **SLA tracker** | Bucket bar chart (Overdue / <1d / <3d / <7d / >7d) + top 25 most-urgent rows. |
| **Root cause** | Six KMeans-surfaced clusters with top-3 city hotspots, category chip, and volume tier (Very high / High / Moderate). |
| **Drafted replies** | Expandable LLM-generated bank replies in the customer's language. |
| **Model performance** | SLA leaderboard (RF / GBM / XGB / LGBM / Stacking / XGB-tuned), feature importance, category confusion matrix, priority GBM importance, live LLM↔ML agreement, human-feedback accuracy. |
| **Analytics** | 8 Plotly charts in a 2-col grid (per-day, by category, by channel, sentiment, severity, breach-by-category, top cities, resolution status). |
| **Feedback** | Human-in-the-loop: pick a complaint → Correct/Wrong buttons → corrections propagate to the DB. |

### Always-visible UI

- **KPI strip** (top): Total / Processed / SLA at risk (>50% breach chance) / Auto-resolved / Avg breach probability (open)
- **Alert banners** (below KPIs, conditional):
  - Red **CRITICAL** when any complaint has P(breach) > 85%
  - Amber **SYSTEMIC ISSUES** when a cluster has ≥ 20 complaints
  - Sand **ESCALATION RISK** when any customer has risk score > 80
- **Submit a new complaint** expander — runs all 9 stages live with per-agent status
- **Sidebar**: stats panel + **RBI Compliance Report CSV download** button

---

## FastAPI service

`uvicorn api.main:app --port 8000` exposes:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET`  | `/`         | service info |
| `POST` | `/complaint` | submit a complaint, returns full pipeline result |
| `GET`  | `/complaints` | list with filters (category, severity, channel, customer) |
| `GET`  | `/stats`     | dashboard KPIs |
| `GET`  | `/report`    | RBI compliance CSV (or `format=json`) |

OpenAPI docs at http://localhost:8000/docs.

---

## Per-agent smoke tests

```bash
python -m agents.intake               # extract structured fields from a sample row
python -m agents.classifier           # classify a few rows
python -m agents.duplicate_detector   # build the vector index, find a duplicate
python -m agents.response_drafter     # draft a Hindi reply
python -m agents.sla_monitor          # due date + breach probability
python -m agents.root_cause           # detect systemic clusters
```

---

## Deploy to Streamlit Cloud

1. Push this repo to GitHub (the pre-seeded `complaintiq.sqlite` and trained `.joblib` files **are tracked** so the app works on first boot).
2. On https://share.streamlit.io create a new app pointing at `dashboard/app.py`.
3. In the app's **Secrets** add:
   ```toml
   GROQ_API_KEY = "gsk_..."
   GROQ_MODEL = "gpt-oss-120b"
   ```
4. Streamlit Cloud will install `requirements.txt`, then start the app.

---

## RBI SLA rules (from `data/sla_rules.json`)

| Category | Base SLA |
|----------|----------|
| UPI / ATM | 5 days |
| Card / NetBanking | 7 days |
| Loan / General | 30 days |

Severity multipliers tighten the window — Critical 0.5×, High 0.7×, Medium 1.0×, Low 1.2×.

---

## Customer Risk Score (0–100)

The single overall score combines three explainable sub-scores:

| Sub-score | Weight | Inputs |
|---|---|---|
| **Ombudsman escalation** | 45% | breach probability, severity, complaint count, amount stuck, angry-history |
| **Churn** | 30% | total complaints, unresolved count, category breadth, current sentiment + severity |
| **Social-media blow-up** | 25% | Twitter / WhatsApp channel use, angry sentiment, severity |

---

## Final dataset shape

- **1000 synthetic complaints** spanning January – May 2026
- **7 channels** (email / WhatsApp / Twitter / calls / branch / portal / mobile_app)
- **3 languages** (English 640, Hindi 247, Marathi 113)
- **6 categories** (General 362, UPI 175, Loan 133, NetBanking 130, Card 105, ATM 95)
- **62 cities** with full lat/long coverage
- **86 duplicates** detected · **914 LLM-drafted replies** · **6 root-cause clusters** · **200 Resolved + 186 Auto-Resolved**

---

## Team

**AgentForge** — iDEA 2.0 / PSBs Hackathon Series 2026 (PS5: Unified Customer Complaint Communication Dashboard)

- Prem Baba
- Purva Bhoyar 
- Pranil Bankar 
- Adhishree Shiledar 
