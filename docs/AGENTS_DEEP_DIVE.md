# ComplaintIQ — The 6 Agents, Explained in Depth

> Your study guide for the finale. This covers **how each agent works internally**,
> **every technical concept it uses**, and **why we built it that way**. Read it top
> to bottom once; after that you can defend any question a judge throws at you.

---

## 0. The mental model (read this first)

A complaint is a piece of text that arrives from some channel (email, WhatsApp,
Twitter, branch, phone). On its own it's unstructured and useless to a dashboard.
Our job is to turn that raw text into **structured, actionable intelligence**.

We do that with an **assembly line of 6 specialist agents**. Each agent does ONE
job well, then hands its output to the next:

```
 raw complaint
      │
      ▼
┌─────────────┐   ┌─────────────┐   ┌──────────────────┐
│ 1. INTAKE   │──▶│ 2.CLASSIFIER│──▶│ 3. DUPLICATE     │
│  clean it   │   │  label it   │   │    DETECTOR       │
└─────────────┘   └─────────────┘   └──────────────────┘
                                            │
      ┌─────────────────────────────────────┘
      ▼
┌─────────────┐   ┌─────────────┐   ┌──────────────────┐
│ 4. RESPONSE │──▶│ 5. SLA      │──▶│ 6. ROOT CAUSE    │
│   DRAFTER   │   │   MONITOR   │   │    DETECTOR       │
└─────────────┘   └─────────────┘   └──────────────────┘
                                            │
                                            ▼
                                    dashboard + alerts
```

**Two families of agents:**

| Family | Agents | Engine | Good at |
|--------|--------|--------|---------|
| **Language agents** | Intake, Classifier, Response Drafter | LLM (Groq, Llama-3.3-70B) | understanding & generating human text |
| **Prediction agents** | Duplicate Detector, SLA Monitor, Root Cause | ML / vector math | numbers, similarity, patterns |

**The golden rule of the whole system: every agent has a deterministic fallback.**
If the LLM is down or the model file is missing, the agent still returns a valid
answer using rules/heuristics. The pipeline **never crashes**. This is your single
strongest talking point — say it early and often.

---

## The foundation layer (what all language agents stand on)

Before the 6 agents, two shared files do critical work. You should understand these
because judges love asking "isn't sending customer data to an external API a privacy
risk?" — and we have a real answer.

### `llm_client.py` — the single door to the LLM

Every language agent calls one of two functions here:

- **`chat(prompt, system=...)`** → returns plain text (used by Response Drafter).
- **`chat_json(prompt, system=...)`** → returns a parsed Python dict (used by Intake
  & Classifier, which need structured fields).

What this layer gives us for free:

1. **One place for the API key** — read from env var, `.env`, or Streamlit secrets.
2. **Retries with backoff** — if a call fails (rate limit, network blip), it retries
   up to 2 more times with a growing delay (`1.5s`, `3s`). Only after all attempts
   fail does it raise — and that's what triggers each agent's fallback.
3. **Robust JSON parsing** — LLMs sometimes wrap JSON in ```` ```json ```` fences or
   add chatter. `chat_json` strips fences and, as a last resort, regex-extracts the
   first `{...}` block. So a slightly messy model reply still parses.
4. **Temperature control** — `temperature=0.0` for Intake/Classifier (we want the
   *same* answer every time = deterministic), `0.4` for the Drafter (we want a bit
   of natural variety in wording).

> **Concept — Temperature.** A number (0–~1) controlling randomness in LLM output.
> `0` = always pick the most likely next word (deterministic, repeatable). Higher =
> more creative/varied but less predictable. We use low temp for classification
> (consistency matters) and medium for writing replies (sound human).

> **Concept — System vs User prompt.** The *system* prompt sets the role and rules
> ("You are a classification agent… respond with JSON only"). The *user* prompt is
> the actual task + data. Separating them makes the model follow rules more reliably.

### `pii.py` — privacy guardrail (PII masking)

> **Concept — PII (Personally Identifiable Information).** Any data that identifies a
> real person: name, mobile, account number, card number, Aadhaar, PAN, email, UPI ID.

**The problem:** Groq is an external API. We must not leak a real customer's
identifiers outside the bank.

**Our solution — reversible token masking:**

1. **Before** the text leaves, `PIIMasker.mask()` finds identifiers and replaces each
   with a stable placeholder token like `PII_NAME_1`, `PII_PHONE_1`, `PII_ACCOUNT_1`.
2. The LLM only ever sees the masked text. It does its job referring to `PII_NAME_1`.
3. **After** the reply comes back, `unmask()` swaps the tokens back to the real values
   *locally*. So the draft reply can still greet "Dear Rahul Sharma" even though Groq
   never saw that name.

How it finds identifiers:

- **Regex patterns** for structured IDs — Indian mobile (`+91` + 10 digits starting
  6–9), 16-digit card, 12-digit Aadhaar, PAN (`ABCDE1234F`), IFSC, UPI VPA, email,
  generic 9–18 digit account numbers. **Order matters**: email is matched before UPI,
  card/Aadhaar before the generic digit-run, so a specific pattern isn't swallowed by
  a generic one.
- **Literal masking** for the name — names aren't regex-detectable, so we pass the
  known customer name in and replace it explicitly (longest-first, so "Rahul Sharma"
  is masked before a lone "Rahul").

Smart details worth mentioning:
- Tokens are `PII_NAME_1`, **not** `[NAME_1]` — LLMs tend to strip brackets as
  "formatting" but preserve a solid alphanumeric token verbatim. `unmask()` is still
  tolerant: it restores `[PII_NAME_1]`, `pii_name_1`, etc.
- **Amounts and locations are deliberately NOT masked** — they aren't identifying, and
  the Classifier/SLA logic *needs* them.
- It can be toggled (`PII_MASKING=0`) so we can show judges the raw-vs-masked diff live.

> **Concept — Regex (regular expression).** A pattern language for matching text. E.g.
> `[6-9]\d{9}` means "a digit 6–9, then any 9 digits" = an Indian mobile number.

---

## Agent 1 — Intake (`intake.py`)  ·  *Language agent*

### What it does
Takes a **raw, messy** complaint (any language, often code-mixed Hindi/English/Marathi)
and produces **one clean structured record** with normalized fields.

### How it works (step by step)
1. Build a prompt embedding the raw text + a **strict JSON schema** listing the exact
   keys we want: `customer_name`, `issue_summary` (one-line English summary),
   `account_type`, `amount_involved`, `transaction_id`, `location_mentioned`,
   `urgency_keywords`, `detected_language`.
2. Call `chat_json` at `temperature=0.0`.
3. **Merge with caller-known data, caller wins.** If the channel already gave us a
   field (e.g. the account type from a verified portal login), we keep it; the LLM
   only fills *gaps*. (`intake.py:64–73`)
4. **Coerce types** — `urgency_keywords` is always forced into a clean `list[str]`.

### The key design decisions (defend these)
- **"Be conservative — if a field isn't present, return null. Do NOT invent values."**
  This is in the system prompt. It stops the LLM hallucinating a fake transaction ID.
- **Source-of-truth wins.** Trusted channel metadata beats the LLM's guess.
- **Fallback:** if the LLM call fails, `_fallback()` returns a minimal record (first
  140 chars as the summary, plus whatever the caller already knew). Downstream agents
  still run.

### Concepts in this agent
- **Schema-constrained extraction** — forcing the LLM to fill a fixed set of fields
  instead of free-form text. Makes the output machine-readable.
- **Normalization** — turning many input forms ("Rs. 5,000", "5000 rupees") into one
  canonical form (`5000` as a number).
- **Code-mixing** — when one sentence blends languages ("UPI payment fail ho gaya").
  The LLM handles this far better than rule-based parsing could.

---

## Agent 2 — Classifier (`classifier.py`)  ·  *Language agent + rule fallback*

### What it does
Assigns **three labels** to each complaint:
- **Category** — `UPI, ATM, Card, Loan, NetBanking, General`
- **Severity** — `Critical, High, Medium, Low`
- **Sentiment** — `Angry, Frustrated, Neutral, Polite`

…plus a one-sentence **rationale** (used on the dashboard for explainability).

### How it works
1. The prompt embeds a **rubric** so the model judges consistently, e.g.
   *Critical = fraud / unauthorized debit / locked out / amount ≥ ₹50,000.*
2. `chat_json` at `temperature=0.0`.
3. **Validation snap-back via `_pick()`** — the model's answer is forced onto an
   allowed value. Exact match first, then partial ("Credit Card" → `Card`), else a
   safe default. The model **cannot** inject an invalid label. (`classifier.py:75–86`)

### Fallback — a real second engine
If the LLM is down, `_fallback()` reproduces all three labels with **pure rules**:
- **Category** via regex keyword patterns (`\bUPI\b`, `credit.?card`, `\bloan\b`…).
- **Severity** via amount thresholds + fraud keywords (`unauthor|fraud|stuck`).
- **Sentiment** via word lists (angry: `unacceptable|worst|!!`; polite: `kindly|please|कृपया`).

This is why we can claim "works even fully offline."

### Concepts in this agent
- **Classification** — assigning an item to one of a fixed set of categories.
- **Multi-label vs multi-class** — here it's 3 separate **multi-class** decisions
  (one category, one severity, one sentiment), not free tags.
- **Output validation / guardrails** — never trust raw LLM output for a controlled
  field; constrain it to your enum.
- **Sentiment analysis** — detecting the emotional tone of text. We use it to
  prioritize angry customers (escalation risk).

---

## Agent 3 — Duplicate Detector (`duplicate_detector.py`)  ·  *Prediction agent*

### What it does
Detects when the **same customer** reports the **same issue more than once** (e.g.
files on WhatsApp, gets impatient, files again by email). Without this, one problem
becomes three tickets and inflates the backlog.

### How it works
1. **Embed** the complaint text into a 384-dimensional vector using
   **sentence-transformers `all-MiniLM-L6-v2`** (runs locally, free, no API).
2. **Store** that vector. In production: a **pgvector** column in Postgres. Locally:
   **ChromaDB**. (Dual-mode — see note below.)
3. For a new complaint, **search** the nearest stored vectors *from the same customer*
   and compute **cosine similarity**.
4. If the best match's similarity ≥ **0.78**, flag it as a duplicate and return which
   complaint it duplicates, plus the neighbour list.

### Why embeddings instead of keyword matching (the core insight)
Two complaints can mean the same thing with **zero shared words**:
- "I bought groceries and the UPI payment failed"
- "my transfer to the store didn't go through"

Keyword search sees no overlap. **Embeddings capture meaning**, so these land close
together in vector space. That's the whole point.

> **Concept — Embedding.** A neural model converts text into a list of numbers (a
> *vector*) — here 384 numbers. Texts with similar *meaning* get similar vectors.
> Think of it as a coordinate in a 384-dimensional "meaning space."

> **Concept — Cosine similarity.** Measures the **angle** between two vectors,
> ignoring their length. Result ranges −1 → 1; **1 = identical direction = same
> meaning**, 0 = unrelated. We use `similarity = 1 − cosine_distance`. We normalize
> the vectors (`normalize_embeddings=True`) so this is fast and well-behaved.

> **Concept — Vector database / pgvector.** A store that can do "find the nearest
> vectors to this one" efficiently. **pgvector** adds this to Postgres; its `<=>`
> operator computes vector distance directly in SQL. **ChromaDB** is a standalone
> vector store we use for local dev. Both use **HNSW**-style indexing under the hood.

### Defend these
- **Scoped to one customer** (`WHERE customer_name = %s`) — we never merge two
  different people's complaints. Safety first.
- **Threshold 0.78** — tuned: high enough to avoid false merges, low enough to catch
  paraphrases.
- **Dual-mode caveat (know this!):** Postgres path uses pgvector's `<=>`; SQLite path
  uses ChromaDB. Say "production runs on **Postgres + pgvector**; ChromaDB is the
  local fallback." The embeddings this agent stores are **reused by Agent 6** — we
  embed once, use twice.

---

## Agent 4 — Response Drafter (`response_drafter.py`)  ·  *Language agent*

### What it does
Writes a **ready-to-send, empathetic, RBI-compliant reply in the customer's own
language**. Output is plain text so an agent can read, tweak, and send it.

### How it works
1. Feeds the prompt the **classifier's outputs** (category/severity/sentiment) + the
   SLA window + a reference number (`UBI/<id>`).
2. The **system prompt encodes bank policy** (the guardrails):
   - Never promise a refund before investigation.
   - Never ask for OTP / credentials.
   - If fraud is involved → advise blocking the card + give the 24×7 helpline.
   - Keep it under 130 words; **match tone to channel** (a tweet reply is shorter
     than an email).
3. `chat` at `temperature=0.4` (a little natural variety).
4. **Fallback:** a safe template letter with the reference number and helpline — so
   even with no LLM, the customer gets a valid, compliant acknowledgment. The error is
   logged server-side but **never shown to the customer** (the reply reads like any
   normal template).

### Defend these
- **It's a *draft for a human to approve*, not auto-send.** This kills the "you let an
  AI talk to customers unsupervised?" objection.
- **Guardrails live in the system prompt** + PII is masked, so no credentials leak and
  no unbacked promises are made.
- **Language matching** — replies go out in Hindi/Marathi/English to match the
  customer, which is a real UX win for an Indian bank.

### Concepts in this agent
- **Conditional / context-grounded generation** — the reply is *conditioned* on
  structured inputs (category, severity, SLA), not written blind.
- **Prompt-based policy guardrails** — encoding compliance rules in the system prompt.
- **Graceful degradation** — the fallback is intentionally indistinguishable from a
  normal template.

---

## Agent 5 — SLA Monitor (`sla_monitor.py`)  ·  *Prediction agent (Random Forest)*

> **Concept — SLA (Service Level Agreement).** The promised deadline to resolve a
> complaint. RBI mandates these for banks. Ours: UPI 5 days, ATM 5, Card 7, Loan 30,
> General 30 (in `sla_rules.json`), adjusted by a severity multiplier.

### What it does — two jobs
1. **Compute the due date** = base SLA for the category × severity multiplier, added to
   the filing date.
2. **Predict the probability the complaint will *breach* its SLA** — so managers can
   act on the risky ones *before* they go late.

### How the prediction works
- A **Random Forest** classifier is trained **offline** (`models/train_sla_model.py`)
  on historical complaints labelled breached / not-breached, and saved as a **joblib**
  file. This agent just **loads and applies** it.
- For a live complaint, `_row_for_model()` builds a **~25-feature vector** that must
  exactly mirror what the model was trained on. Features include:
  - **Raw:** amount, text length, word count, channel, language, account type.
  - **Time:** day of week, is-weekend, hours since filed, % of SLA elapsed.
  - **Risk flags:** fraud-keyword present, is-high-amount (≥25k), is-high-value (>50k),
    is-duplicate, is-repeat-customer, channel-risk (twitter/whatsapp = public = risky).
  - **Encoded ordinals:** severity_score (Critical=4…Low=1), sentiment_score
    (Angry=4…Polite=1).
- `predict_proba(X)[0, 1]` returns the probability of the "breach" class.

### Why ML instead of a simple rule?
Breach risk is **multi-factor and non-linear**. A high amount alone isn't decisive,
but *high amount + angry + filed on a weekend + repeat customer* together is. A Random
Forest **learns these interactions** from data; hand-written rules can't cover them all.

> **Concept — Random Forest.** An ensemble of many **decision trees**. Each tree is
> trained on a random subset of the data/features and votes; the forest averages the
> votes. This beats a single tree (less overfitting) and naturally handles mixed
> numeric/categorical features and non-linear interactions.

> **Concept — Feature engineering.** Turning raw data into informative numeric inputs
> ("features") for a model. E.g. converting a date into `day_of_week`, `is_weekend`,
> `hours_since_filed`. The model is only as good as its features.

> **Concept — `predict_proba`.** Instead of a hard yes/no, the model outputs a
> probability (0–1). We surface this as `breach_probability` so the dashboard can rank
> complaints by risk rather than just flag them.

> **Concept — Train/serve consistency.** The feature vector at prediction time must be
> built **identically** to training time, or the model gets garbage. That's exactly
> why `_row_for_model()` mirrors `engineer_features()` from the training script —
> a great detail to mention; it shows ML maturity.

### Fallback
If the `.joblib` file is missing, `_rule_based_prob()` returns a sensible heuristic
probability (base 0.18, +0.35 if Critical, +0.10 if amount ≥25k, etc.). Pipeline still
runs.

---

## Agent 6 — Root Cause Detector (`root_cause.py`)  ·  *Prediction agent (KMeans)*

### What it does
Finds **systemic problems** that no single complaint reveals — e.g. *"47 UPI failures
from Nagpur this week = a payment-gateway outage."* This is the agent that turns a pile
of tickets into a **management insight**.

### How it works
1. **Reuses the embeddings from Agent 3** — it does **not** re-embed. (Efficiency win:
   embed once in Agent 3, cluster here.)
2. Runs **KMeans** clustering on all the complaint vectors to group semantically
   similar complaints together.
3. For each cluster, it checks two conditions:
   - **size ≥ 5** complaints, AND
   - the cluster is **≥ 60% dominated by a single category**.
4. If both hold, it raises an **alert** and names the **top-3 hotspot cities** so the
   summary is actionable ("concentrated in Nagpur" vs vague "multiple cities").
5. Alerts are sorted (biggest/most-dominant first) and stored in the DB.

### Duplicate Detector vs Root Cause — the difference (judges will ask)
| | Duplicate Detector (3) | Root Cause (6) |
|---|---|---|
| Question | "Is THIS the same as one earlier complaint?" | "Are MANY complaints secretly the same problem?" |
| Scope | one customer, pairwise | all customers, group-wise |
| Method | nearest-neighbour similarity | clustering (KMeans) |
| Output | duplicate flag | systemic alert |

Same embeddings, **opposite zoom level**: one zooms in on a pair, the other zooms out
on the whole population.

> **Concept — Clustering (unsupervised learning).** Grouping data points by similarity
> **without** pre-defined labels. The algorithm discovers the groups itself — perfect
> for *emerging* issues we didn't know to look for.

> **Concept — KMeans.** Picks `k` cluster centers, assigns each point to its nearest
> center, recomputes centers as the mean of their points, repeats until stable. We set
> `k` adaptively (`len(vectors) // 5`, min 2) and use `random_state=42` so results are
> **reproducible**. `n_init=10` runs it 10 times and keeps the best to avoid bad random
> starts.

> **Concept — Supervised vs Unsupervised.** SLA Monitor (Agent 5) is **supervised** —
> it learned from labelled breach/no-breach examples. Root Cause (Agent 6) is
> **unsupervised** — no labels; it discovers structure on its own. Good to contrast
> these two on stage.

> **Concept — Dominance threshold.** A cluster only becomes an alert if one category
> makes up ≥60% of it. This filters out vague mixed clusters and keeps alerts crisp
> and actionable.

---

## The 4 supporting models (beyond the 6-agent spine)

These live in `agents/` too, but they are **not** part of the main pipeline — they are
*helpers* that make the pipeline's output more trustworthy and actionable on the
dashboard. Know them at a high level so a judge browsing the folder can't surprise you.

### A. `ml_category.py` — ML second opinion on category
- **What:** a **TF-IDF + Logistic Regression** model trained on past complaints. Given
  the text, it independently predicts the category (with a probability).
- **Why:** it cross-checks Agent 2 (the LLM classifier). If LLM and ML **agree**, we
  show **"High Confidence"**; if they disagree, **"Needs Review"**. Two independent
  methods agreeing is stronger evidence than one.
- **Robustness:** wrapped in try/except so a scikit-learn version mismatch can never
  crash the pipeline — the LLM answer still stands; the second opinion is just skipped.

> **Concept — TF-IDF (Term Frequency–Inverse Document Frequency).** A way to turn text
> into numbers by weighting words: frequent-in-this-complaint but rare-across-all-
> complaints words score high (they're distinctive, e.g. "ATM", "EMI"). Common words
> like "the" score low. It's a classic, lightweight alternative to neural embeddings.

> **Concept — Logistic Regression.** A simple, fast linear classifier that outputs a
> probability per class. Pairs naturally with TF-IDF for text classification.

> **Concept — Ensemble / cross-validation of methods.** Using two different models
> (LLM + classical ML) and trusting the result more when they agree. This is the
> "High Confidence / Needs Review" badge on the dashboard.

### B. `sentiment_ml.py` — ML second opinion on sentiment
- **What:** a **HuggingFace RoBERTa** model
  (`cardiffnlp/twitter-roberta-base-sentiment-latest`) that returns
  Positive/Neutral/Negative. We map those to the LLM's labels (Polite→positive,
  Angry/Frustrated→negative) so "agreement" is well-defined.
- **Why:** same idea as above — a transformer model double-checks the LLM's sentiment.
- **Robustness:** returns `None` if torch/the model can't load (no internet); the LLM
  sentiment still stands. (This is the model behind the Windows DLL / `KMP_DUPLICATE_LIB_OK`
  hygiene — it sets those env vars at import to avoid the onnxruntime crash.)

> **Concept — Transformer / RoBERTa.** A neural network architecture (the same family
> as the LLM) fine-tuned specifically for sentiment on tweets. Smaller and specialized
> compared to the 70B general LLM.

### C. `priority.py` — priority score (0–100)
- **What:** a **Gradient Boosting** model (`priority_gbm.joblib`) that fuses severity,
  sentiment, amount, repeat-customer count, days since filed, is-duplicate, and the
  SLA breach probability into a single **0–100 priority number** for queue ordering.
- **Why:** managers need one sortable number to decide what to handle first.

> **Concept — Gradient Boosting.** Like Random Forest it's an ensemble of decision
> trees, but trees are built **sequentially**, each correcting the previous one's
> errors. Often slightly more accurate than a Random Forest for tabular data.

> **Note — it consumes other agents' output.** Its `breach_probability` feature comes
> from Agent 5. This shows how the agents *compose*: SLA risk feeds the priority score.

### D. `risk_score.py` — explainable customer risk (0–100)
- **What:** a **rule-based** score (not ML) split into three explainable components:
  - **Ombudsman risk** (0.45 weight) — chance the customer files a formal RBI complaint.
  - **Churn risk** (0.30) — chance the customer leaves the bank.
  - **Social media risk** (0.25) — chance of a public Twitter/WhatsApp blow-up.
  - **Overall** = weighted average of the three.
- **Why rule-based here?** Because the bank needs to *explain* a risk score to
  regulators ("why is this customer high-risk?"). Transparent weighted rules are
  auditable; a black-box model isn't. It draws on the customer's **full history**
  (repeat complaints, unresolved tickets, angry tone, public channels).

> **Concept — Explainability vs accuracy trade-off.** Sometimes a transparent rule is
> preferable to a more accurate black box — especially in regulated banking, where you
> must justify a decision. Agent 5 (Random Forest) optimizes accuracy; risk_score
> optimizes explainability. Good to contrast on stage.

---

## One-screen cheat sheet

| # | Agent | Type | Engine | Key concept | Fallback |
|---|-------|------|--------|-------------|----------|
| 1 | Intake | Language | LLM | schema-constrained extraction | minimal record |
| 2 | Classifier | Language | LLM | classification + enum guardrail | regex keyword rules |
| 3 | Duplicate Detector | Prediction | embeddings + pgvector | cosine similarity | ChromaDB / rule |
| 4 | Response Drafter | Language | LLM | guardrailed generation | template letter |
| 5 | SLA Monitor | Prediction | Random Forest | feature engineering + predict_proba | rule-based prob |
| 6 | Root Cause | Prediction | KMeans | unsupervised clustering | returns no alerts |

**Cross-cutting strengths (your closing points):**
1. **Every agent degrades gracefully** — no LLM, no model file? It still answers.
2. **Privacy by design** — PII is masked before any data leaves the bank.
3. **Compute reuse** — embeddings are computed once (Agent 3) and reused (Agent 6).
4. **Right tool for each job** — LLMs for language, classical ML for prediction.

---

## Likely judge questions & your one-liners

- *"What if the Groq API is down?"* → Every language agent has a deterministic
  fallback; the pipeline completes regardless.
- *"Isn't sending data to an external LLM a privacy risk?"* → We mask all PII to
  reversible tokens before the call and un-mask locally; Groq never sees real IDs.
- *"Why embeddings, not keyword search, for duplicates?"* → Same meaning can have zero
  shared words; embeddings + cosine similarity capture meaning.
- *"Why ML for SLA instead of a rule?"* → Breach risk is non-linear and multi-factor;
  a Random Forest learns feature interactions a rule can't.
- *"How is Root Cause different from Duplicate Detection?"* → Same embeddings, opposite
  zoom: one compares a pair, the other clusters the whole population.
- *"Why KMeans / is it reproducible?"* → Unsupervised clustering finds emerging issues
  with no labels; `random_state=42` + `n_init=10` make it stable and repeatable.
