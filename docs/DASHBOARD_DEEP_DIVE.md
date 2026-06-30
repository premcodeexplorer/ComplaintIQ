# ComplaintIQ — The Dashboard, Explained in Depth

> Study guide for whoever presents the **dashboard / front-end** at the finale.
> Covers **every screen, every tab, what it shows, how it's built, and the technical
> concepts behind it.** Read it once top-to-bottom; after that you can defend any
> "what does this screen do / how did you build it" question.
>
> File this all lives in: **`dashboard/app.py`** (~1,475 lines) plus
> **`dashboard/rbi_report.py`** (the CSV report generator).

---

## 0. The big picture (say this first)

ComplaintIQ's dashboard is a **single-page web app built with Streamlit**. It's the
"face" of the system — bank admins log in and see everything the 6 agents produced:
live complaints, risk scores, maps, SLA deadlines, systemic alerts, and model health.

> **Concept — Streamlit.** A Python framework that turns a plain Python script into a
> web app. You write `st.metric(...)`, `st.dataframe(...)`, `st.plotly_chart(...)` and
> Streamlit renders them as UI. **It re-runs the whole script top-to-bottom on every
> interaction** (every click/filter change) — that's the key mental model. State that
> must survive a re-run lives in `st.session_state`.

**What the page is made of, top to bottom:**

```
┌──────────────────────────────────────────────────────────┐
│  LOGIN SCREEN  (if not authenticated)                    │
├──────────────────────────────────────────────────────────┤
│  SIDEBAR: RBI Compliance report + CSV download           │
│  ────────────────────────────────────────────────────    │
│  HEADER: "ComplaintIQ" + tagline                         │
│  KPI ROW: 5 metric cards (totals, SLA at risk, ...)      │
│  ALERT BANNERS: red / orange / yellow (conditional)      │
│  LIVE SUBMIT: form to run all 6 agents in real time      │
│  "Process N pending complaints" button                   │
│  ──────────────────────────────────────────────────────  │
│  9 TABS:                                                  │
│  Live feed | Customer | India map | SLA tracker |        │
│  Root cause | Drafted replies | Model performance |      │
│  Analytics | Feedback                                    │
└──────────────────────────────────────────────────────────┘
```

---

## 1. Login & session (the gate)

Before anything renders, `main()` checks if the user is logged in.

- **Auth backend:** Supabase (`auth/supabase_auth.py` → `sign_in`, `sign_out`). The
  login form (`render_login_screen`) takes email + password and calls `sign_in`.
- **Stay-logged-in across refreshes — cookies.** A browser **cookie**
  (`complaintiq_admin_session`, 7-day expiry = `max_age=604800`) stores the session.
  On load, `main()` reads the cookie → restores `st.session_state["admin_session"]`,
  so a page refresh doesn't kick you back to login.
- **Sign out** clears both the session state and the cookie.

> **Concept — Session state vs cookie.** `st.session_state` lives only while the tab is
> open (lost on refresh). A **cookie** persists in the browser, so we copy the session
> into a cookie to survive refreshes. This dance is in `main()` (steps 1–4).

> **Talking point:** "It's a real authenticated admin portal, not an open demo — login
> is enforced and the session persists for 7 days via a secure cookie."

---

## 2. Data loading & caching (why it's fast)

Two cached loaders feed the whole page:
- `load_complaints()` → reads all complaints from the DB into a **pandas DataFrame**.
- `load_alerts()` → reads the root-cause alerts.

Both are decorated with `@st.cache_data(ttl=20)`.

> **Concept — Caching (`@st.cache_data`).** Because Streamlit re-runs the whole script
> on every click, without caching we'd hit the database dozens of times per second.
> `cache_data` stores the result and reuses it; `ttl=20` means "refresh at most every
> 20 seconds." After a new complaint is processed we call `load_complaints.clear()` to
> force a refresh.

> **Concept — pandas DataFrame.** A table in memory (rows = complaints, columns =
> fields). Almost every screen is "filter/group/sort this DataFrame, then chart it."

---

## 3. Header zone (above the tabs) — shown on every screen

### 3a. KPI row (`render_kpis`) — 5 metric cards
1. **Total complaints**
2. **Processed** (how many the agents have run on; shows pending count as delta)
3. **SLA at risk** — open complaints with breach probability ≥ 50%
4. **Auto-resolved** — complaints closed automatically (duplicate or standard reply)
5. **Avg breach probability (open)** — mean predicted breach risk across open cases

> These are the "executive summary" numbers a manager glances at first.

### 3b. Alert banners (`render_alert_banners`) — conditional, color-coded
Only appear when their trigger fires:
- **RED — CRITICAL:** open complaints with breach probability **> 85%**.
- **ORANGE — SYSTEMIC ISSUES:** root-cause clusters with ≥ 20 complaints (collapsed
  into one summary banner naming the top 3).
- **YELLOW — ESCALATION RISK:** customers with risk score **> 80** (proactive outreach).

> **Talking point:** "The dashboard doesn't just display data — it *pushes the urgent
> things to the top* so a manager can't miss them."

### 3c. Live submit (`render_live_submit` + `_run_live_pipeline`) — the showpiece
A form where you type a complaint (English/Hindi/Marathi), pick channel/language/
account/amount, and hit **Run pipeline**. It then:
1. Inserts the complaint, then **streams agent-by-agent progress live** ("Intake:
   working… → done", a progress bar to 6/6) via the orchestrator's `on_step` callback.
2. Shows the final result: category, severity, sentiment, risk score, SLA due date,
   breach probability, duplicate?, and the drafted reply.
3. **Privacy panel (`_render_pii_panel`)** — shows **raw vs PII-masked** text side by
   side, proving customer identifiers are stripped before going to the LLM. This is a
   killer live demo for judges who care about data privacy.

### 3d. "Process N pending complaints" button
Batch-runs the pipeline over any unprocessed complaints, then refreshes.

---

## 4. Sidebar — RBI Compliance Report (`render_sidebar` + `rbi_report.py`)

The left sidebar is the **regulator-facing** view:
- Shows the logged-in admin's name/email + Sign Out.
- Summary tiles: Total / Resolved (manual) / Auto-resolved / Pending / Breached, plus
  a **by-category** table.
- **Download button → CSV** in the exact schema of the **RBI Master Circular on Customer
  Service in Banks (2024)**: Complaint ID, Customer, Type, Severity, Channel, Date
  Filed, SLA Deadline, Days Remaining, Breach Status, Resolution Status.

> **Details worth mentioning:** the CSV is encoded `utf-8-sig` (a BOM) so **Excel opens
> Hindi names correctly**; breach status is computed (Breached / At Risk / On Track),
> and resolution status maps internal codes to readable labels.

> **Talking point:** "Compliance isn't an afterthought — one click exports a
> regulator-ready RBI report."

---

## 5. The 9 tabs (the heart of the dashboard)

> Defined in `main()`: `st.tabs([...])` creates them; each `with tab_x:` block calls a
> `render_*` function.

### Tab 1 — Live feed (`render_live_feed`)
**What:** a filterable, sortable table of complaints (top 60).
- **Filters:** severity, category, channel, "only at-risk (>50%)" checkbox.
- **Sort by:** Priority / Date / Risk score / Breach probability.
- **The clever bit — agreement badges:** a category/sentiment shown as `UPI` means the
  **LLM and the ML second-opinion model agreed** (high confidence); `UPI (*)` means
  they **disagreed** and the row is flagged for human review. (This is where Agents
  `ml_category` / `sentiment_ml` surface on the UI.)

> **Talking point:** "Every classification is cross-checked by a second model; the `(*)`
> tells a human exactly which rows to double-check."

### Tab 2 — Customer (`render_customer_view`)
**What:** drill into one customer's full story.
- **Emotion timeline:** a Plotly bar chart of the customer's sentiment over time
  (Polite→Angry on the y-axis), colored by severity — you literally *see* a customer
  getting angrier.
- **Risk score card (0–100)** + the three explainable sub-scores with progress bars:
  **Ombudsman escalation (45%) + Churn (30%) + Social-media (25%)** — straight from the
  `risk_score` agent.
- Full complaint history table for that customer.

> **Talking point:** "This is the 360° customer view — emotion trend + why they're a
> retention/escalation risk, broken into explainable components."

### Tab 3 — India map (`render_india_map`)
**What:** a geographic **choropleth + bubble map** of complaint hotspots across India.
- Built with **Plotly `graph_objects`** + locally cached **GeoJSON** files
  (`india_states.geojson`, `india_districts.geojson`) — so it draws India outlined by
  states/districts, no external tiles needed.
- City bubbles sized & colored by **complaint volume** (Very high 30+ / High / Moderate
  / Other), with permanent labels on the top 4 cities.

> **Concept — Choropleth / GeoJSON.** GeoJSON is a file describing geographic shapes
> (state/district borders) as coordinates. A **choropleth** map shades/draws regions
> from that file. We overlay **scatter-geo** bubbles for per-city counts.

> **Talking point:** "Geographic clustering is visible at a glance — if Nagpur lights
> up red, that's a regional outage."

### Tab 4 — SLA tracker (`render_sla_tracker`)
**What:** complaints sorted by **days until deadline**.
- Bucketed: Overdue / <1 day / <3 days / <7 days / >7 days, with a bar chart
  (Overdue = Critical red … >7 days = calm).
- Table of the 25 most urgent, with breach probability and risk score.

> **Talking point:** "Turns the abstract SLA promise into an actionable 'do these
> first' worklist."

### Tab 5 — Root cause (`render_alerts`)
**What:** the systemic clusters from **Agent 6 (KMeans)**.
- Each cluster shown as a colored card: cluster #, dominant category, volume tier, count,
  and a plain-English summary ("47 UPI complaints concentrated in Nagpur…").
- Caption explains it groups complaints by **semantic similarity of the text** (so two
  cities can join one cluster if the issue is the same).

> **Talking point:** "This is the agent that finds problems no single complaint reveals
> — a payment-gateway outage hiding inside 47 separate tickets."

### Tab 6 — Drafted replies (`tab_draft` block in `main()`)
**What:** the LLM-drafted bank replies from **Agent 4**, newest 20.
- Each is an expander: original complaint + the drafted reply, with channel/language/
  SLA metadata.

> **Talking point:** "Ready-to-send, language-matched, RBI-compliant drafts — the agent
> writes, a human approves."

### Tab 7 — Model performance (`render_model_performance`) — the "we did real ML" tab
**What:** proof the models are trained and measured. Six sections:
1. **SLA breach predictor** — an **algorithm bake-off** leaderboard (which model won),
   **Hold-out AUC**, **5-fold CV AUC**, training rows, XGBoost best params, and a
   **feature-importance** bar chart (top 15).
2. **Category classifier (TF-IDF + Logistic Regression)** — accuracy + a **confusion
   matrix** heatmap.
3. **Priority scorer (Gradient Boosting)** — **R²** and **MAE**, feature importances.
4. **Sentiment model (RoBERTa)** — described (no training; runs locally).
5. **Live LLM ↔ ML agreement** — real % agreement (category & sentiment) from the DB.
6. **Human feedback accuracy** — from the Feedback tab (see below).

> **Concept — AUC (Area Under ROC Curve).** A score 0.5–1.0 for how well a classifier
> ranks positives above negatives. 0.5 = random guessing, 1.0 = perfect. Better than
> raw accuracy when classes are imbalanced (few breaches vs many on-time).

> **Concept — Cross-validation (5-fold).** Split data into 5 parts, train on 4 / test
> on 1, rotate 5 times, average. Gives a more trustworthy score than a single split.

> **Concept — Confusion matrix.** A grid of actual vs predicted labels. The diagonal =
> correct; off-diagonal = mistakes (e.g. "Card" misread as "UPI"). Shows *where* the
> model errs, not just how often.

> **Concept — R² and MAE (for the priority *regressor*).** R² (0–1) = fraction of
> variance the model explains; MAE = average absolute error in points. Used because
> priority is a 0–100 *number*, not a category.

> **Talking point:** "We didn't just call an LLM — we trained classical models, ran a
> bake-off, and we show the AUC, confusion matrix and feature importance live."

### Tab 8 — Analytics (`render_analytics`)
**What:** 8 Plotly charts in a 2-column grid:
1. Complaints per day (line) · 2. Resolution status (pie) · 3. By category (pie) ·
4. Sentiment distribution (donut) · 5. Severity (bar) · 6. By channel (bar) ·
7. Top 10 cities (bar) · 8. Avg breach probability by category (bar).

> **Talking point:** "The classic BI/analytics view — trends, distributions, and where
> risk concentrates."

### Tab 9 — Feedback (`render_feedback`) — human-in-the-loop
**What:** a reviewer picks a complaint and marks each label **Correct** or **Wrong**;
if Wrong, they pick the right value. Saved via `db.record_feedback`.
- Stats up top: total reviews, marked correct, corrections, accuracy rate, and
  corrections-by-field.

> **Concept — Human-in-the-loop (HITL).** Letting humans validate/correct AI output and
> capturing that signal. It (a) measures real-world accuracy and (b) builds a labeled
> dataset to retrain/improve the models later.

> **Talking point:** "The system learns from its reviewers — every correction is logged
> and feeds the accuracy metrics on the Model Performance tab. It closes the loop."

---

## 6. Design / polish details (good for "attention to detail" credit)

- **Consistent brand palette** (cream / sand / blue / ink) with **fixed color meanings**
  everywhere: Critical=red, High=orange, Medium=yellow, Low=slate. A judge sees the
  same color = same severity on every chart.
- **Custom CSS** injected (`_GLOBAL_CSS`) styles tabs, buttons, metric cards, tables.
- **Windows DLL hygiene** at the very top (preload `torch`, set `KMP_DUPLICATE_LIB_OK`)
  so sentence-transformers/RoBERTa don't crash on Windows — that's why the app starts
  reliably on a fresh laptop.

---

## 7. One-screen cheat sheet (the 9 tabs)

| Tab | Shows | Built with | Source agent(s) |
|-----|-------|-----------|-----------------|
| Live feed | filterable complaint table + agreement badges | st.dataframe | Classifier + ML second-opinions |
| Customer | emotion timeline + risk sub-scores | Plotly bar + progress bars | risk_score |
| India map | geographic hotspots | Plotly geo + GeoJSON | (location data) |
| SLA tracker | deadline buckets + urgent list | Plotly bar | SLA Monitor |
| Root cause | systemic clusters | colored cards | Root Cause (KMeans) |
| Drafted replies | LLM bank replies | expanders | Response Drafter |
| Model performance | AUC, confusion matrix, importances | Plotly | all trained models |
| Analytics | 8 BI charts | Plotly | all |
| Feedback | human correction + accuracy | buttons/forms | human-in-the-loop |

**Cross-cutting strengths (your closing points):**
1. **Real auth + 7-day cookie session** — it's a product, not a toy demo.
2. **Pushes urgency up** — KPI cards + color-coded alert banners.
3. **Live pipeline demo** — submit a complaint, watch all 6 agents run, see PII masked.
4. **Regulator-ready** — one-click RBI compliance CSV.
5. **Proves the ML** — Model Performance tab shows AUC/confusion matrix/importances.
6. **Closes the loop** — human feedback feeds accuracy tracking.

---

## 8. Likely judge questions & one-liners

- *"Is this just static charts?"* → No — it's live on the DB with 20s caching, a
  real-time submit pipeline, and human feedback writing back to the database.
- *"How do you keep users logged in?"* → Supabase auth + a 7-day browser cookie synced
  to Streamlit session state.
- *"What makes a complaint show `(*)`?"* → The ML second-opinion disagreed with the LLM
  — it's flagged for human review.
- *"How is the India map drawn without internet/tiles?"* → Local GeoJSON of Indian
  states/districts rendered as a Plotly choropleth + city bubbles.
- *"How do you prove the models are good?"* → Model Performance tab: hold-out + 5-fold
  CV AUC, confusion matrix, R²/MAE, and live LLM↔ML agreement %.
- *"Where's the privacy safeguard?"* → The submit flow shows raw vs PII-masked text;
  identifiers never reach the LLM.
- *"Does it meet RBI requirements?"* → One-click CSV in the RBI Master Circular (2024)
  schema, Excel-safe for Hindi names.
