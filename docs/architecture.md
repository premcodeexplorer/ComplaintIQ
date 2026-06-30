# ComplaintIQ — System Architecture Diagrams

Two Mermaid diagrams for the final-round presentation:

- **Part A — Full System:** the whole platform end-to-end (the "this is a real product" slide).
- **Part B — Pipeline Zoom:** inside the AI brain (the "this is how it thinks" slide).

> **Export tip:** paste a diagram into <https://mermaid.live> and export PNG/SVG.
> Start copying from the `flowchart` line — do **not** include the ```` ```mermaid ```` fence.

## Colour legend

| Colour | Meaning |
|--------|---------|
| 🟣 Purple | **LLM** — Groq `llama-3.3-70b` (language tasks: intake, classify, draft) |
| 🔵 Cyan | **Trained ML** — prediction tasks (duplicate, SLA, root-cause, priority) |
| 🟢 Teal | **ML cross-check** — independently audits the LLM's category & sentiment |
| 🟠 Orange | **Business logic** — risk score + auto-resolution |
| 🔴 Red | **Security** — PII masking + admin auth |
| 🟢 Green | **Data** — Supabase PostgreSQL + pgvector |

---

## Part A — Full System Architecture

```mermaid
flowchart TB
    subgraph CLIENT["👥 CLIENT INTERFACES"]
        direction LR
        EMAIL["📧 Email"]
        CALLS["📞 Calls"]
        PORTAL["🌐 Public Portal"]
        SOCIAL["📱 Social Media"]
        BRANCH["🏦 Branch"]
    end

    subgraph INGEST["⚙️ INGESTION & API"]
        direction LR
        POLLER["Email Poller<br/>IMAP · Railway"]
        API["FastAPI · Uvicorn<br/>REST Backend"]
    end

    ORCH["🧠 PIPELINE ORCHESTRATOR · sync + streaming"]:::orch

    subgraph PIPELINE["🤖 6-AGENT PIPELINE + ML SECOND-OPINIONS"]
        direction TB
        A1["1 · Intake<br/>Groq LLM"]:::llm
        A2["2 · Classifier<br/>Groq LLM"]:::llm
        XC["🔍 ML Cross-Check<br/>Category TF-IDF · Sentiment RoBERTa"]:::chk
        A3["3 · Duplicate Detector<br/>MiniLM + pgvector"]:::ml
        A4["4 · Response Drafter<br/>Groq LLM"]:::llm
        A5["5 · SLA Monitor<br/>XGBoost"]:::ml
        PR["Priority · GBM"]:::ml
        RISK["💯 Risk Score<br/>Ombudsman·Churn·Social"]:::logic
        AUTO["⚡ Auto-Resolution"]:::logic
        A6["6 · Root Cause<br/>KMeans"]:::ml
        A1 --> A2 --> A3 --> A4 --> A5 --> PR --> RISK --> AUTO --> A6
        A2 -.->|agree?| XC
    end

    PII["🔒 PII Masking · before every LLM call"]:::sec

    subgraph DATA["🗄️ DATA PERSISTENCE · Supabase"]
        direction LR
        PG[("PostgreSQL<br/>complaints + feedback")]:::db
        VEC[("pgvector<br/>384-dim")]:::db
    end

    subgraph STAFF["🛠️ INTERNAL STAFF TOOLS"]
        direction LR
        DASH["Streamlit Dashboard<br/>9 tabs · KPIs"]
        RBI["RBI Report<br/>CSV export"]
    end

    AUTH["🔑 Admin Auth · Supabase"]:::sec
    FB["🔁 Human-in-the-Loop Feedback"]:::sec

    subgraph LEGEND["🎨 LEGEND"]
        direction LR
        L1["LLM"]:::llm
        L2["ML"]:::ml
        L3["Cross-Check"]:::chk
        L4["Logic"]:::logic
        L5["Security"]:::sec
        L6["Data"]:::db
    end

    EMAIL --> POLLER
    CALLS --> POLLER
    SOCIAL --> PORTAL
    BRANCH --> PORTAL
    PORTAL --> API
    POLLER --> ORCH
    API <--> PG
    ORCH -->|triggers| A1
    A1 & A2 & A4 -.- PII
    AUTO --> PG
    A3 <--> VEC
    PG --> DASH --> RBI
    AUTH -->|secures| DASH
    DASH -.->|corrections| FB
    FB -.-> PG

    classDef orch fill:#7c3aed,stroke:#4c1d95,color:#fff
    classDef llm fill:#7c3aed,stroke:#4c1d95,color:#fff
    classDef ml fill:#0891b2,stroke:#155e75,color:#fff
    classDef chk fill:#0d9488,stroke:#134e4a,color:#fff
    classDef logic fill:#ea580c,stroke:#9a3412,color:#fff
    classDef db fill:#16a34a,stroke:#14532d,color:#fff
    classDef sec fill:#dc2626,stroke:#7f1d1d,color:#fff
```

---

## Part B — Inside the Pipeline (AI brain zoom)

```mermaid
flowchart TB
    RAW(["📨 Raw Complaint"]):::io
    A1["1 · Intake — Groq LLM<br/>structured extraction"]:::llm
    A2["2 · Classifier — Groq LLM<br/>category · severity · sentiment"]:::llm
    XC["🔍 ML Cross-Check<br/>Category TF-IDF · Sentiment RoBERTa<br/>flag if disagree"]:::chk
    A3["3 · Duplicate Detector<br/>MiniLM + pgvector cosine"]:::ml
    A4["4 · Response Drafter — Groq LLM<br/>reply in customer language"]:::llm
    A5["5 · SLA Monitor — XGBoost<br/>due date + breach probability"]:::ml
    PR["Priority Score — GBM"]:::ml
    RISK["💯 Risk Score 0-100<br/>Ombudsman · Churn · Social"]:::logic
    AUTO{"⚡ Auto-Resolution?<br/>Low/Med+Polite or Duplicate"}:::logic
    OUT(["✅ Enriched row → Postgres"]):::io
    A6["6 · Root Cause — KMeans<br/>systemic clusters + hotspots"]:::ml
    PII["🔒 PII Masking · every LLM call"]:::sec

    subgraph LEGEND["🎨 LEGEND"]
        direction LR
        L1["LLM"]:::llm
        L2["ML"]:::ml
        L3["Cross-Check"]:::chk
        L4["Logic"]:::logic
        L5["Security"]:::sec
    end

    RAW --> A1 --> A2 --> A3 --> A4 --> A5 --> PR --> RISK --> AUTO --> OUT
    A2 -.->|agree?| XC
    A1 & A2 & A4 -.- PII
    OUT -.->|batch| A6

    classDef llm fill:#7c3aed,stroke:#4c1d95,color:#fff
    classDef ml fill:#0891b2,stroke:#155e75,color:#fff
    classDef chk fill:#0d9488,stroke:#134e4a,color:#fff
    classDef logic fill:#ea580c,stroke:#9a3412,color:#fff
    classDef sec fill:#dc2626,stroke:#7f1d1d,color:#fff
    classDef io fill:#334155,stroke:#0f172a,color:#fff
```

---

## Talking points

- **Part A:** *"This isn't a notebook demo — it's a deployed platform: a public portal, a live email channel, cloud Postgres with vector search, an auth-gated dashboard, and a human-in-the-loop feedback loop."*
- **Part B:** *"We use the right engine for each job — LLM for language, classical ML for prediction — and on the two subjective calls (category & sentiment) a second ML model audits the LLM. Disagreement triggers human review, which feeds retraining."*
