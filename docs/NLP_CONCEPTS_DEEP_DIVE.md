# ComplaintIQ — All the NLP Concepts, Explained in Depth

> The "theory layer" under the whole project. Every NLP / language-AI concept
> ComplaintIQ uses, explained from scratch in plain English, **with the exact place in
> our project where it appears**. Read this and you can answer any "what is X / why did
> you use X" question a judge asks about the language side of the system.
>
> Pairs with `AGENTS_DEEP_DIVE.md` (the 6 agents) and `DASHBOARD_DEEP_DIVE.md` (the UI).

---

## 0. What is NLP, and where is it in ComplaintIQ?

> **NLP (Natural Language Processing)** = getting computers to understand and generate
> human language (text/speech). A complaint is unstructured human text in English,
> Hindi or Marathi — NLP is how we turn it into structured, actionable data.

ComplaintIQ uses **two generations of NLP** side by side:

| Generation | What it is | Where we use it |
|------------|-----------|-----------------|
| **Classical NLP** | hand-built features + small ML models | TF-IDF category model, regex PII masking |
| **Neural / modern NLP** | deep learning on text | embeddings (MiniLM), RoBERTa sentiment, the gpt-oss-120b LLM |

A core selling point: **we combine both** — the LLM does the heavy understanding, and
classical models cross-check it cheaply.

---

## 1. The starting problem: text isn't numbers

Computers do math, not words. **Every NLP technique is ultimately a way to turn text
into numbers** a model can compute on. The differences are *how* they do it:

- **TF-IDF** → counts words (sparse, no meaning).
- **Embeddings** → a neural net maps text to a dense vector that *captures meaning*.
- **LLM** → predicts the next token given all previous tokens.

Keep this in mind; the rest of the doc is variations on "text → numbers → decision."

---

## 2. Tokenization

> **Tokenization** = splitting text into units ("tokens") a model processes — words,
> sub-words, or pieces. "UPI payment failed" → `["UPI", "payment", "failed"]` (word-
> level) or `["UP", "I", "pay", "ment", ...]` (sub-word, used by transformers/LLMs).

**Where in our project:**
- The LLM and RoBERTa internally tokenize into **sub-words** (so they handle unseen
  words like brand names or typos gracefully).
- RoBERTa is capped at `max_length=512` tokens (`sentiment_ml.py`) — long complaints
  are **truncated**; that's why we slice `text[:1000]` before inference.

> **Concept — Sub-word tokenization (BPE/WordPiece).** Instead of whole words, models
> break rare words into known pieces. Lets a fixed vocabulary cover any input, including
> Hindi/Marathi and code-mixed text.

---

## 3. Text preprocessing & stopwords

> **Preprocessing** = cleaning text before a classical model sees it: lowercasing,
> removing punctuation, stripping **stopwords** (super-common words like "the", "is",
> "और" that carry little meaning).

**Where in our project:**
- The **TF-IDF category model** (`ml_category.py` / `models/train_category_classifier`)
  relies on this kind of normalization so distinctive words dominate.
- **Note:** neural models (LLM, RoBERTa, MiniLM) **don't** want heavy preprocessing —
  they use the raw text because context and casing carry signal. So we preprocess for
  classical models but feed raw text to neural ones. Good nuance to mention.

---

## 4. TF-IDF (Term Frequency–Inverse Document Frequency)

> **TF-IDF** turns a document into a vector of word weights. A word scores high if it's
> **frequent in this complaint** (TF) but **rare across all complaints** (IDF). So "ATM"
> or "EMI" score high (distinctive); "bank" or "the" score low (everywhere).

**Where in our project:** the **category second-opinion model** — `ml_category.py` loads
a `TfidfVectorizer + LogisticRegression` pipeline trained on past complaints.

**Why we use it:** it's tiny, fast, fully explainable, and needs no GPU — a perfect
cheap cross-check on the LLM's category guess.

> **Strength/limit to state honestly:** TF-IDF is **bag-of-words** — it ignores word
> order and meaning ("card blocked" vs "blocked card" look identical; synonyms look
> unrelated). That's exactly why we *also* have embeddings (next section).

---

## 5. Word/sentence embeddings (the heart of our NLP)

> **Embedding** = a neural model converts text into a dense vector of numbers (ours =
> **384 numbers**) such that **texts with similar meaning get similar vectors**. Think of
> each complaint as a point in a 384-dimensional "meaning space."

**Where in our project:** the **Duplicate Detector (Agent 3)** and **Root Cause (Agent
6)** both use `sentence-transformers` model **`all-MiniLM-L6-v2`**, run locally.

**Why it beats TF-IDF here:** two complaints can mean the same thing with **zero shared
words**:
- "I bought groceries and UPI failed"
- "my payment to the store didn't go through"

TF-IDF sees no overlap; embeddings place them **close together** because they *mean* the
same thing. This is the whole reason duplicate detection works.

> **Concept — Dense vs sparse vectors.** TF-IDF is **sparse** (mostly zeros, one slot
> per vocabulary word). Embeddings are **dense** (384 meaningful numbers). Dense vectors
> capture semantics; sparse ones capture surface words.

> **Concept — `all-MiniLM-L6-v2`.** A small, fast sentence-transformer (~80MB) that's a
> distilled BERT. "Sentence" transformer = tuned to embed whole sentences (not just
> words) so similar sentences land near each other. We pick it for speed + it runs free
> on CPU, no API.

---

## 6. Cosine similarity & semantic search

> **Cosine similarity** measures the **angle** between two vectors, ignoring length.
> Range −1 → 1; **1 = same direction = same meaning**, 0 = unrelated. We compute it as
> `similarity = 1 − cosine_distance`.

**Where in our project:** Agent 3 embeds a new complaint, then finds the **nearest
neighbours** (highest cosine similarity) among the same customer's past complaints. If
the best match ≥ **0.78**, it's a duplicate.

> **Concept — Semantic search / nearest-neighbour.** "Find the items whose *meaning* is
> closest to this query." That's vector search. A **vector database** (pgvector in
> Postgres, ChromaDB locally) stores embeddings and does this fast. pgvector's `<=>`
> operator computes vector distance directly in SQL.

> **Why we normalize embeddings** (`normalize_embeddings=True`): it makes cosine
> similarity equivalent to a simple dot product — faster and numerically stable.

---

## 7. Text clustering (unsupervised NLP)

> **Clustering** = grouping texts by similarity **without predefined labels**. The
> algorithm discovers the groups itself — ideal for spotting *emerging* issues we didn't
> know to look for.

**Where in our project:** **Root Cause (Agent 6)** runs **KMeans** on all complaint
embeddings. Complaints describing the same underlying failure cluster together — even
from different cities — revealing a systemic problem (e.g. a payment-gateway outage)
hidden across 47 separate tickets.

> **Concept — KMeans.** Pick `k` cluster centers, assign each point to the nearest
> center, move centers to the average of their points, repeat until stable. We use
> `random_state=42` (reproducible) and `n_init=10` (run 10 times, keep the best).

> **Semantic clustering** = clustering on *embeddings* (meaning), not raw words. This is
> NLP + unsupervised ML working together.

---

## 8. Text classification

> **Text classification** = assign a text to one of a fixed set of categories.

**Where in our project — two complementary ways:**
1. **LLM classifier (Agent 2)** — the gpt-oss-120b model reads the complaint and outputs
   category / severity / sentiment with a rubric in the prompt.
2. **TF-IDF + Logistic Regression (`ml_category.py`)** — a classical model gives a
   **second opinion** on the category. Agree → "High Confidence"; disagree → "Needs
   Review" (the `(*)` flag on the dashboard).

> **Concept — Logistic Regression for text.** A simple linear classifier that takes the
> TF-IDF vector and outputs a probability per category. Fast, interpretable, the classic
> baseline for text classification.

> **Concept — Multi-class vs multi-label.** Ours is **multi-class** (exactly one
> category per complaint), done three times (category, severity, sentiment) — not free
> tagging.

---

## 9. Sentiment analysis

> **Sentiment analysis** = detecting the emotional tone of text. We use it to prioritize
> angry customers (escalation/PR risk).

**Where in our project — again, two ways:**
1. **LLM (Agent 2)** → `Angry / Frustrated / Neutral / Polite`.
2. **`sentiment_ml.py`** → a **HuggingFace RoBERTa** model
   (`cardiffnlp/twitter-roberta-base-sentiment-latest`) → `Positive / Neutral /
   Negative`. We map LLM↔RoBERTa labels into shared buckets and score agreement.

> **Concept — Fine-tuned transformer.** RoBERTa is a BERT-family model **fine-tuned**
> specifically on tweets for sentiment. Smaller and specialized vs the 70B general LLM —
> a good, cheap expert for one job.

---

## 10. Named Entity Recognition (NER) & PII detection

> **NER** = finding and labeling entities in text (names, places, account numbers,
> dates). **PII detection** is NER focused on *identifying* information.

**Where in our project:** `pii.py` detects PII before any text goes to the LLM. We use a
**rule-based / regex** approach (not a neural NER model) for structured identifiers:
Indian mobile, 16-digit card, Aadhaar, PAN, IFSC, UPI VPA, email, account numbers — plus
literal matching of the known customer name.

> **Why regex, not a neural NER model?** Bank identifiers have **strict, predictable
> formats** (PAN = `ABCDE1234F`), so regex is 100% precise, instant, needs no model, and
> is fully auditable for a regulator. A neural NER would be overkill and less certain.
> (Honest trade-off: regex can miss free-form things like names, which is why we also
> pass the known name in explicitly.)

> **Concept — Reversible token masking.** We replace each entity with a stable token
> (`PII_NAME_1`) before the LLM call and swap it back after. The LLM never sees real
> identifiers, but the final reply still greets the real customer. (Full detail in
> `AGENTS_DEEP_DIVE.md`.)

---

## 11. Information extraction (structured extraction)

> **Information extraction** = pulling structured fields out of free text.

**Where in our project:** **Intake (Agent 1)** extracts `issue_summary`, `account_type`,
`amount_involved`, `transaction_id`, `location`, `urgency_keywords`, `detected_language`
from a messy complaint — by asking the LLM to fill a **fixed JSON schema**.

> **Concept — Schema-constrained / structured generation.** Forcing the model to return
> a specific JSON shape instead of free prose, so the output is machine-readable. We then
> **validate** it (e.g. `_pick()` snaps any label to an allowed value) — never trust raw
> LLM output for a controlled field.

---

## 12. Text summarization

> **Summarization** = compressing text into a shorter version that keeps the key meaning.
> **Abstractive** (the model writes a new sentence) vs **extractive** (pick existing
> sentences).

**Where in our project:** Intake's `issue_summary` is a one-line **abstractive** summary
the LLM writes ("Customer's UPI payment of ₹5000 failed but amount was debited"). The
fallback uses a crude **extractive** approach (first 140 characters).

---

## 13. Text generation (NLG)

> **NLG (Natural Language Generation)** = producing fluent human text. Modern NLG = an
> LLM predicting one token at a time, each conditioned on all previous tokens.

**Where in our project:** **Response Drafter (Agent 4)** generates a complete,
empathetic, RBI-compliant bank reply **in the customer's language**.

> **Concept — Conditional / grounded generation.** The reply is *conditioned* on
> structured context (category, severity, sentiment, SLA, reference number) plus policy
> rules in the system prompt — it's not written blind. This keeps it relevant and
> compliant.

---

## 14. Multilingual & code-mixed NLP

> **Multilingual NLP** = handling more than one language. **Code-mixing** = multiple
> languages blended in one sentence ("UPI payment fail ho gaya, please help").

**Where in our project:** complaints arrive in **English, Hindi, Marathi**, often
code-mixed. The LLM handles all three (and the mix) natively — Intake detects the
language, the Classifier understands it, and the Drafter **replies in the same
language**. This is a real edge for an Indian bank and very hard to do with classical
rule-based NLP.

> **Concept — Language detection.** Identifying which language a text is in. Intake
> outputs `detected_language`; we also let the channel's known language win if provided.

---

## 15. Large Language Models (LLMs) & prompting

> **LLM** = a very large transformer trained on huge text corpora to predict the next
> token. With the right prompt it can classify, extract, summarize, translate, and write
> — all without task-specific training (**zero-shot / few-shot**).

**Where in our project:** **Groq-hosted `gpt-oss-120b`** powers Agents 1, 2,
4. One model, many jobs, via different prompts.

Key prompting concepts we use:
- **System vs user prompt** — system sets the role/rules ("You are a classification
  agent… JSON only"); user carries the task + data. Separation improves rule-following.
- **Zero-shot prompting** — we describe the task + rubric without giving labeled
  examples; the model generalizes. (No fine-tuning needed = cheap and fast to build.)
- **Temperature** — randomness knob. `0.0` for Intake/Classifier (deterministic,
  repeatable), `0.4` for the Drafter (a little natural variety). See §16.
- **Prompt-based guardrails** — compliance rules ("never promise refunds, never ask for
  OTP") live in the system prompt.
- **Output parsing & validation** — `chat_json` strips ```` ```json ```` fences and
  regex-extracts the first `{...}`; then we validate against our allowed values.

> **Concept — Hallucination.** When an LLM confidently makes up facts. We fight it with:
> "if a field isn't present return null, don't invent" (Intake), enum validation
> (Classifier), and **deterministic fallbacks** so a bad/failed call can't poison the
> pipeline.

---

## 16. Temperature, determinism & reliability

> **Temperature** controls how random the LLM's next-token choice is. `0` = always the
> most likely token (deterministic — same input gives same output). Higher = more varied
> but less predictable.

**Where in our project:** low temp where consistency matters (classification, extraction
must be repeatable for judges and for the second-opinion agreement to be meaningful),
medium temp where we want the reply to sound human.

> **Reliability layer (cross-cutting NLP-ops):** `llm_client.py` adds **retries with
> backoff** for transient API failures and every language agent has a **rule-based
> fallback**, so the NLP pipeline degrades gracefully instead of crashing.

---

## 17. Evaluating NLP models

How we prove our language models actually work (shown on the **Model Performance** tab):

> **Accuracy** — % of predictions correct. Simple, but misleading on imbalanced data.

> **AUC (Area Under ROC Curve)** — 0.5 (random) → 1.0 (perfect); measures how well a
> classifier *ranks* positives above negatives. Used for the SLA breach predictor.

> **Confusion matrix** — actual vs predicted grid; the diagonal is correct, off-diagonal
> shows *which* categories get confused (e.g. "Card" misread as "UPI"). Shown for the
> TF-IDF category model.

> **Cross-validation (5-fold)** — train/test on rotating splits and average, for a
> trustworthy score that doesn't depend on one lucky split.

> **LLM ↔ ML agreement %** — our own metric: how often the LLM and the classical model
> agree on category/sentiment. High agreement = high confidence; disagreements get human
> review.

> **Human-in-the-loop accuracy** — reviewers mark predictions Correct/Wrong on the
> Feedback tab; this is real-world accuracy and a future training signal.

---

## 18. The big map — every NLP concept → where it lives

| NLP concept | Type | In ComplaintIQ |
|-------------|------|----------------|
| Tokenization / sub-words | foundational | LLM, RoBERTa, MiniLM internals |
| Stopwords / preprocessing | classical | TF-IDF category model |
| TF-IDF | classical vectorization | `ml_category.py` |
| Embeddings (MiniLM, 384-d) | neural vectorization | Agents 3 & 6 |
| Cosine similarity / semantic search | similarity | Agent 3 (duplicates) |
| Vector DB (pgvector / ChromaDB) | retrieval | Agent 3 storage |
| Clustering (KMeans on embeddings) | unsupervised | Agent 6 (root cause) |
| Text classification | supervised | Agent 2 + TF-IDF second opinion |
| Sentiment analysis | supervised | Agent 2 + RoBERTa second opinion |
| NER / PII detection | extraction | `pii.py` (regex) |
| Information extraction (schema) | extraction | Agent 1 (intake) |
| Summarization | generation | Agent 1 `issue_summary` |
| Text generation (NLG) | generation | Agent 4 (drafter) |
| Multilingual / code-mixed | cross-cutting | all LLM agents (EN/HI/MR) |
| Language detection | classification | Agent 1 `detected_language` |
| LLM + prompting (zero-shot) | generative | Agents 1, 2, 4 |
| Temperature / determinism | LLM control | `llm_client.py` |
| Evaluation (AUC, confusion, CV) | NLP-ops | Model Performance tab |

---

## 19. Likely judge questions & one-liners

- *"Difference between TF-IDF and embeddings?"* → TF-IDF counts surface words (no
  meaning, ignores order); embeddings are dense neural vectors that capture meaning, so
  synonyms/paraphrases land close. We use TF-IDF for a cheap cross-check and embeddings
  for duplicate/root-cause where meaning matters.
- *"How does duplicate detection understand paraphrases?"* → It embeds both complaints
  and compares them by cosine similarity in 384-d meaning space, not by shared words.
- *"Why a separate sentiment model if the LLM already does it?"* → Two independent
  methods agreeing is stronger evidence; disagreement flags rows for human review.
- *"How do you detect PII — is it ML?"* → Rule-based regex for bank identifiers (strict
  formats → 100% precise, auditable) plus literal name matching, with reversible token
  masking before the LLM call.
- *"Is the LLM fine-tuned?"* → No — zero-shot prompting with rubrics + validation +
  fallbacks. Cheap, fast, and good enough because we cross-check with classical models.
- *"How do you handle Hindi/Marathi/code-mixing?"* → The LLM is natively multilingual;
  Intake detects language and the Drafter replies in the same language.
- *"How do you stop hallucination?"* → "return null, don't invent" instructions, enum
  validation on outputs, and deterministic fallbacks so a bad call can't corrupt data.
- *"How do you know your NLP works?"* → Model Performance tab: AUC, confusion matrix,
  5-fold CV, live LLM↔ML agreement %, and human-feedback accuracy.
