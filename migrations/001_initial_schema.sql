-- =============================================================
--  ComplaintIQ — Initial Schema Migration (001)
--  Target: Supabase / PostgreSQL 15+
--  Translated from: database/db.py (SQLite)
--
--  Run via:  python migrations/apply_migrations.py
--  Or paste into: Supabase Dashboard → SQL Editor → Run
-- =============================================================

-- ── complaints ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS complaints (
    id               TEXT        PRIMARY KEY,
    customer_name    TEXT        NOT NULL,
    channel          TEXT        NOT NULL,
    complaint_text   TEXT        NOT NULL,
    language         TEXT,
    date             TEXT        NOT NULL,
    location         TEXT,
    account_type     TEXT,
    amount_involved  DOUBLE PRECISION,

    -- Agent enrichments (nullable until pipeline runs)
    category         TEXT,                   -- UPI / ATM / Card / Loan / NetBanking / General
    severity         TEXT,                   -- Critical / High / Medium / Low
    sentiment        TEXT,                   -- Angry / Frustrated / Neutral / Polite
    intake_json      TEXT,                   -- raw structured-intake JSON (Agent 1)
    duplicate_of     TEXT REFERENCES complaints(id),
    similarity       DOUBLE PRECISION,       -- duplicate similarity score (Agent 3)
    draft_response   TEXT,                   -- generated reply (Agent 4)
    sla_due_date     TEXT,                   -- ISO date by which complaint must resolve
    sla_breach_prob  DOUBLE PRECISION,       -- 0..1 SLA breach probability (Agent 5)
    risk_score       INTEGER,                -- 0..100 overall customer risk score
    risk_ombudsman   INTEGER,                -- 0..100 RBI Ombudsman escalation sub-score
    risk_churn       INTEGER,                -- 0..100 churn sub-score
    risk_social      INTEGER,                -- 0..100 public / social-media blow-up sub-score
    cluster_id       INTEGER,                -- KMeans cluster (Agent 6)
    status           TEXT        NOT NULL DEFAULT 'open',  -- open / resolved / escalated

    -- ML second-opinion columns
    ml_category          TEXT,              -- XGBoost category prediction
    ml_category_prob     DOUBLE PRECISION,  -- confidence score 0..1
    category_confidence  TEXT,              -- 'High Confidence' / 'Needs Review'
    ml_sentiment         TEXT,              -- positive / neutral / negative
    ml_sentiment_prob    DOUBLE PRECISION,  -- confidence score 0..1
    sentiment_confidence TEXT,              -- 'High Confidence' / 'Needs Review'
    priority_score       INTEGER,           -- 0..100 composite priority

    -- Timestamps
    resolved_at   TEXT,                     -- ISO timestamp when complaint was resolved
    processed_at  TEXT                      -- ISO timestamp when pipeline finished
);

CREATE INDEX IF NOT EXISTS idx_complaints_customer  ON complaints(customer_name);
CREATE INDEX IF NOT EXISTS idx_complaints_date      ON complaints(date);
CREATE INDEX IF NOT EXISTS idx_complaints_category  ON complaints(category);
CREATE INDEX IF NOT EXISTS idx_complaints_location  ON complaints(location);
CREATE INDEX IF NOT EXISTS idx_complaints_status    ON complaints(status);
CREATE INDEX IF NOT EXISTS idx_complaints_severity  ON complaints(severity);

-- ── root_cause_alerts ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS root_cause_alerts (
    id          SERIAL      PRIMARY KEY,
    cluster_id  INTEGER,
    category    TEXT,
    location    TEXT,
    count       INTEGER,
    summary     TEXT,
    created_at  TEXT
);

-- ── feedback ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feedback (
    id              SERIAL   PRIMARY KEY,
    complaint_id    TEXT     NOT NULL REFERENCES complaints(id),
    field           TEXT     NOT NULL,
    original_value  TEXT,
    corrected_value TEXT,
    is_correct      BOOLEAN  NOT NULL,
    created_at      TEXT     NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feedback_complaint ON feedback(complaint_id);
