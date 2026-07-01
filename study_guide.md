# ComplaintIQ - Deep Dive Study Guide

## Table of Contents
1. [The Problem Deep Dive](#1-the-problem-deep-dive)
2. [Why Traditional Solutions Fail](#2-why-traditional-solutions-fail)
3. [The Solution Architecture - Layer by Layer](#3-the-solution-architecture---layer-by-layer)
4. [Deep Dive: Each Agent Explained](#4-deep-dive-each-agent-explained)
5. [ML Models: The Mathematical Intuition](#5-ml-models-the-mathematical-intuition)
6. [The Vector Database: ChromaDB Explained](#6-the-vector-database-chromadb-explained)
7. [Root Cause Detection: KMeans Clustering Deep Dive](#7-root-cause-detection-kmeans-clustering-deep-dive)
8. [Risk Score: The Weighted Formula Explained](#8-risk-score-the-weighted-formula-explained)
9. [Database Schema Deep Dive](#9-database-schema-deep-dive)
10. [Code Flow Analysis](#10-code-flow-analysis)
11. [Why Each Choice Over Alternatives](#11-why-each-choice-over-alternatives)
12. [Hands-On: Tracing One Complaint](#12-hands-on-tracing-one-complaint)

---

## 1. The Problem Deep Dive

### The Business Context

Imagine you work at Union Bank of India. Every day:

- **Thousands of complaints** arrive from different channels
- Each must be categorized, prioritized, and responded to
- RBI mandates strict **SLA (Service Level Agreement)** deadlines
- Failure to meet SLA results in **penalties** and **customer escalation** to RBI Ombudsman
- Customers expect responses in their **own language** (English, Hindi, Marathi)

### The Data Characteristics

```python
# From data/complaints.json
{
    "id": "UBI-0001",
    "customer_name": "Pooja Mishra",
    "channel": "email",           # 7 channels: email, whatsapp, twitter, 
                                   #           branch, bank_portal, phone_call, mobile_app
    "complaint_text": "My UPI payment of Rs.3500 failed but amount got debited...",
    "language": "english",        # 3 languages: english (640), hindi (247), marathi (113)
    "date": "2026-01-15",
    "location": "Nagpur",         # 62 cities across India
    "account_type": "savings",
    "amount_involved": 3500
}
```

### The Scale Challenge

```
Traditional Manual Process:
┌─────────────────────────────────────────────────────────────────┐
│ Complaint arrives                                               │
│     ↓                                                            │
│ Human reads the complaint (2-3 minutes)                         │
│     ↓                                                            │
│ Human classifies: "This is a UPI issue" (30 seconds)            │
│     ↓                                                            │
│ Human determines severity: "High - amount stuck" (30 seconds)   │
│     ↓                                                            │
│ Human checks for duplicates: search database (2 minutes)        │
│     ↓                                                            │
│ Human drafts response: "Dear Customer..." (5 minutes)           │
│     ↓                                                            │
│ Human calculates SLA: check rules, calculate due date (1 min)   │
│     ↓                                                            │
│ Human assesses risk: Will they escalate? (2 minutes)            │
│                                                                 │
│ TOTAL: ~15 minutes per complaint × 1000 complaints = 250 hours │
└─────────────────────────────────────────────────────────────────┘

With ComplaintIQ:
┌─────────────────────────────────────────────────────────────────┐
│ Complaint arrives                                               │
│     ↓                                                            │
│ AI processes: all steps in ~5 seconds                           │
│                                                                 │
│ TOTAL: 5 seconds per complaint × 1000 complaints = 1.4 hours    │
└─────────────────────────────────────────────────────────────────┘

Speed improvement: 178x faster
```

---

## 2. Why Traditional Solutions Fail

### A) Rule-Based Systems Don't Work

Traditional approach: Write rules like "IF text contains 'UPI' THEN category = UPI"

**Why it fails:**
```
Complaint: "My phone pay is not working, I tried to transfer 5000 rupees"

Rule-based: 
- Contains "pay" but not "UPI" → might miss it
- What about "GPay", "PhonePe", "Paytm"? New terms every month
- Can't handle: multilingual, typos, context

AI-based:
- Understands "phone pay" = "UPI"
- Handles variations naturally
- Learns from data
```

### B) Basic Keyword Matching Fails

```
Complaint: "I don't want to complain, but my card got blocked"

Keywords: "complain", "blocked", "card"
Expected: Card issue - HIGH severity

But the customer is being POLITE - they don't want to complain
Keyword matcher would incorrectly flag this as "Angry"
```

### C) Can't Handle Language Diversity

```
English: "UPI payment failed"
Hindi:   "UPI पेमेंट फेल हो गया"
Marathi: "UPI पेमेंट फेल झालं"

Rule-based: Need separate rules for each language
AI: Handles all three naturally
```

### D) No Way to Detect Systemic Issues

```
Bank's internal systems would show:
- 47 separate UPI complaints from Nagpur
- 23 separate UPI complaints from Mumbai
- Each handled individually

But ComplaintIQ's root cause detection finds:
- 70 UPI complaints are actually ONE systemic issue
- 50 are from Nagpur region → local gateway problem
- This is actionable intelligence!
```

---

## 3. The Solution Architecture - Layer by Layer

### Layer 1: Data Ingestion

```
Raw Input Sources                    Normalized Format
─────────────────                   ──────────────────
Email         ──────────────────►
WhatsApp      ──────────────────►    {
Twitter       ──────────────────►     complaint_text: str,
Phone Call    ──────────────────►     channel: str,
Branch        ──────────────────►     language: str,
Bank Portal   ──────────────────►     date: str,
Mobile App   ──────────────────►     location: str,
                                     account_type: str,
                                     amount_involved: float
                                    }
```

**Why normalize?** Each channel has different formats:
- Email: Subject, body, attachments
- Twitter: 280 chars, mentions, hashtags
- WhatsApp: Text, maybe voice note
- Branch: Paper form, handwritten

All get normalized to a common schema.

### Layer 2: AI Agents Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                    THE 6 AI AGENTS                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Agent 1: INTAKE                                                      │
│  ─────────────                                                       │
│  Input:  "मेरा UPI payment fail हो गया है 5000 rupees"                │
│  Output: {                                                            │
│      customer_name: null,                                            │
│      issue_summary: "UPI payment failed, amount debited",            │
│      account_type: "savings",                                        │
│      amount_involved: 5000,                                          │
│      transaction_id: null,                                           │
│      location_mentioned: "Nagpur",                                   │
│      urgency_keywords: ["fail", "stuck"],                            │
│      detected_language: "hindi"                                      │
│  }                                                                    │
│                                                                      │
│  HOW IT WORKS: LLM reads the Hindi/English text and extracts         │
│  structured information using its understanding of language         │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Agent 2: CLASSIFIER                                                 │
│  ─────────────                                                       │
│  Input:  complaint_text + amount + channel + account_type           │
│  Output: {                                                            │
│      category: "UPI",         # UPI | ATM | Card | Loan | NetBanking │
│      severity: "High",       # Critical | High | Medium | Low      │
│      sentiment: "Frustrated",# Angry | Frustrated | Neutral | Polite│
│      rationale: "Repeated failure with amount stuck"               │
│  }                                                                    │
│                                                                      │
│  HOW IT WORKS: Another LLM call with specific instructions           │
│  to classify into mutually exclusive categories                     │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Agent 3: DUPLICATE DETECTOR                                          │
│  ─────────────────────                                               │
│  Input:  complaint_text + customer_name                             │
│  Output: {                                                            │
│      is_duplicate: True,                                             │
│      duplicate_of: "UBI-0001",                                        │
│      similarity: 0.82,                                               │
│      neighbours: [...]                                               │
│  }                                                                    │
│                                                                      │
│  HOW IT WORKS:                                                       │
│  1. Convert text to 384-dimensional vector (embedding)             │
│  2. Search ChromaDB for similar complaints                         │
│  3. Only search within SAME CUSTOMER                               │
│  4. If similarity > 0.78 threshold → duplicate                      │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Agent 4: RESPONSE DRAFTER                                           │
│  ──────────────────                                                  │
│  Input:  category, severity, sentiment, language, channel           │
│  Output: "Dear Customer,                                             │
│          Thank you for contacting Union Bank...                    │
│          We have registered your complaint under UBI/UBI-0002...    │
│          For urgent assistance call 1800-22-2244..."              │
│                                                                      │
│  HOW IT WORKS:                                                       │
│  1. Build prompt with all context                                   │
│  2. LLM generates professional, RBI-compliant response             │
│  3. Response is in customer's language                              │
│  4. Skipped for duplicates (use prior response)                    │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Agent 5: SLA MONITOR                                                │
│  ──────────────                                                      │
│  Input:  category, severity, date, amount, sentiment, channel       │
│  Output: {                                                            │
│      sla_due_date: "2026-06-08",                                    │
│      sla_days: 4,                                                    │
│      breach_probability: 0.35                                        │
│  }                                                                    │
│                                                                      │
│  HOW IT WORKS:                                                       │
│  1. Calculate due date from RBI rules:                             │
│     - UPI base = 5 days                                             │
│     - High severity multiplier = 0.7                                │
│     - Effective SLA = 5 × 0.7 = 3.5 → 4 days                       │
│  2. ML model predicts breach probability                           │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Agent 6: ROOT CAUSE DETECTOR                                        │
│  ──────────────────────                                              │
│  Input:  (Runs on ALL complaints, not one)                          │
│  Output: [                                                            │
│      {                                                                │
│          cluster_id: 3,                                               │
│          category: "UPI",                                             │
│          location: "Nagpur",                                         │
│          top_cities: [("Nagpur", 23), ("Mumbai", 15), ("Pune", 9)], │
│          count: 47,                                                   │
│          summary: "47 UPI complaints concentrated in Nagpur..."    │
│      }                                                                │
│  ]                                                                    │
│                                                                      │
│  HOW IT WORKS:                                                       │
│  1. Get all complaint embeddings from ChromaDB                     │
│  2. Run KMeans clustering (k=12)                                    │
│  3. For each cluster:                                               │
│     - If ≥5 complaints AND ≥60% same category → ALERT             │
│     - Identify geographic hotspots                                  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Layer 3: ML Second Opinions

```
┌──────────────────────────────────────────────────────────────────────┐
│              ML SECOND OPINIONS (FOR CONFIDENCE)                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Why second opinions?                                                │
│  ────────────────────                                                │
│  • LLM makes mistakes (especially on edge cases)                   │
│  • ML models provide independent verification                       │
│  • When LLM and ML disagree → flag for human review                │
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │ ML CATEGORY     │  │ ML SENTIMENT    │  │ ML PRIORITY     │    │
│  │                 │  │                 │  │                 │    │
│  │ TF-IDF + LogReg │  │ RoBERTa         │  │ Gradient Boost  │    │
│  │                 │  │                 │  │                 │    │
│  │ Input: text     │  │ Input: text     │  │ Input: features│    │
│  │ Output: category│  │ Output: sentiment│ │ Output: 0-100  │    │
│  │ Accuracy: 97%  │  │ Runs locally    │  │ R²: 0.997      │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                                                                      │
│  Agreement Checking:                                                │
│  ──────────────────                                                  │
│  • If LLM category == ML category → "High Confidence"               │
│  • If they disagree → "Needs Review"                               │
│  • Dashboard shows (*) marker for disagreements                     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Layer 4: Risk Scoring

```
┌──────────────────────────────────────────────────────────────────────┐
│                    RISK SCORE CALCULATION                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  THREE SUB-SCORES:                                                  │
│                                                                      │
│  1. OMBUDSMAN ESCALATION RISK (45% weight)                         │
│     ────────────────────────────────                                │
│     Factors:                                                        │
│     • breach_probability × 40  (0-40 points)                        │
│     • severity: Critical(+30), High(+20), Medium(+10), Low(+3)     │
│     • complaint_count × 3      (0-15 points)                        │
│     • amount ≥ 100000 (+12), ≥25000 (+6)                           │
│     • angry_history × 2       (0-8 points)                         │
│                                                                      │
│     Example:                                                        │
│     - breach_prob = 0.6 → 0.6 × 40 = 24                            │
│     - severity = High → +20                                        │
│     - complaints = 2 → +6                                          │
│     - amount = 50000 → +6                                          │
│     - angry_count = 1 → +2                                         │
│     ─────────────────────────────────                              │
│     OMBUDSMAN = 58/100                                              │
│                                                                      │
│  2. CHURN RISK (30% weight)                                        │
│     ─────────────────                                              │
│     Factors:                                                        │
│     • total_complaints × 6        (0-30 points)                   │
│     • unresolved × 5             (0-25 points)                    │
│     • categories_touched × 6      (0-20 points)                    │
│     • Angry/Frustrated = +12                                        │
│     • Critical/High = +8                                          │
│                                                                      │
│  3. SOCIAL MEDIA RISK (25% weight)                                │
│     ──────────────────────                                         │
│     Factors:                                                        │
│     • twitter/whatsapp_complaints × 12  (0-45 points)             │
│     • current channel is public = +18                             │
│     • sentiment = Angry (+22), Frustrated (+10)                   │
│     • severity = Critical (+8)                                     │
│                                                                      │
│  OVERALL CALCULATION:                                               │
│  ────────────────────                                               │
│  overall = 0.45 × ombudsman + 0.30 × churn + 0.25 × social        │
│                                                                      │
│  Example:                                                           │
│  overall = 0.45 × 58 + 0.30 × 52 + 0.25 × 45                      │
│          = 26.1 + 15.6 + 11.25                                     │
│          = 52.95 → 53/100                                          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Layer 5: Auto-Resolution Logic

```
┌──────────────────────────────────────────────────────────────────────┐
│                    AUTO-RESOLUTION LOGIC                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  WHY AUTO-RESOLVE?                                                   │
│  ─────────────────                                                   │
│  • 18.6% of complaints are duplicates or simple cases             │
│  • Don't waste human time on these                                 │
│  • Free up agents for complex issues                               │
│                                                                      │
│  RULES:                                                             │
│  ───────                                                            │
│                                                                      │
│  Rule 1: DUPLICATE → auto_resolved_dup                              │
│  ─────────────────────────────────────                              │
│  IF Agent 3 finds is_duplicate = True THEN                         │
│      status = "auto_resolved_dup"                                  │
│      Don't generate new response (use prior complaint's response)  │
│                                                                      │
│  Rule 2: SIMPLE CASE → auto_resolved_std                           │
│  ─────────────────────────────────────────                          │
│  IF severity IN [Low, Medium]                                       │
│     AND sentiment = Polite                                          │
│     AND category IN [UPI, ATM, Card, NetBanking, Loan, General]   │
│  THEN                                                               │
│      status = "auto_resolved_std"                                   │
│      Generate standard template response                            │
│                                                                      │
│  Rule 3: DEFAULT → open                                            │
│  ──────────────────────                                             │
│  ELSE                                                               │
│      status = "open"                                                │
│      Human needs to review                                          │
│                                                                      │
│  STATISTICS:                                                        │
│  ───────────                                                        │
│  From 1000 complaints:                                              │
│  - 86 duplicates (8.6%)                                             │
│  - 100 auto-resolved_std (10%)                                      │
│  - Total: 186/1000 = 18.6% auto-resolved                            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Deep Dive: Each Agent Explained

### Agent 1: INTAKE - The Information Extractor

#### What it does

Takes raw complaint text and extracts structured information:

```python
# Input
raw_complaint = {
    "complaint_text": "मेरा UPI payment fail हो गया है, 5000 rupees debited हो गए",
    "channel": "whatsapp",
    "language": "hindi"
}

# Output
intake_output = {
    "customer_name": None,           # Not in text
    "issue_summary": "UPI payment failed, amount debited",
    "account_type": "savings",       # Guessed
    "amount_involved": 5000,
    "transaction_id": None,
    "location_mentioned": None,
    "urgency_keywords": ["fail", "stuck", "debited"],
    "detected_language": "hindi"
}
```

#### Why Use LLM for This?

**Traditional approach:**
```python
# Regex-based extraction
import re

def extract_amount(text):
    match = re.search(r'(\d+)\s*(?:rupees|rs\.?|INR)', text, re.IGNORECASE)
    return int(match.group(1)) if match else None
```

**Problems with regex:**
- Can't handle: "five thousand rupees", "₹5000"
- Can't extract: issue_summary (requires understanding)
- Can't detect: urgency_keywords, sentiment
- Language-specific rules needed

**LLM approach:**
- Understands context
- Handles any language
- Extracts multiple fields at once
- Learns from examples

#### The Prompt Engineering

```python
SYSTEM = """
You are an intake agent for an Indian bank's complaint system.
You receive raw customer complaints in English, Hindi or Marathi
(often code-mixed) and return a JSON object with normalized fields.
Be conservative: if a field is not present, return null. Do NOT invent values.
"""

PROMPT_TEMPLATE = """
Extract a structured complaint record from the message below.

Return ONLY a JSON object with these exact keys (use null for unknown):
{{
  "customer_name": string or null,
  "issue_summary": short one-sentence summary in English,
  "account_type": one of ["savings", "current", "credit_card", "loan", "demat", null],
  "amount_involved": number (INR, no commas) or null,
  "transaction_id": string or null,
  "location_mentioned": string or null,
  "urgency_keywords": list of strings (e.g. ["urgent","blocked","unable"]) or [],
  "detected_language": one of ["english","hindi","marathi","mixed"]
}}

Channel: {channel}
Known customer name (may be empty): {customer_name}

Complaint text:
\"\"\"{text}\"\"\"
"""
```

**Key prompt elements:**
1. **Output format specification** - exact JSON keys
2. **Conservative instruction** - don't invent values
3. **Language support** - handles multilingual
4. **Context** - channel, customer name as hints

#### The Code Flow

```python
# agents/intake.py - Simplified flow

def extract(raw: dict[str, Any]) -> dict[str, Any]:
    text = raw.get("complaint_text") or ""
    
    # Build prompt with context
    prompt = PROMPT_TEMPLATE.format(
        channel=raw.get("channel", "unknown"),
        customer_name=raw.get("customer_name", "") or "",
        text=text,
    )
    
    try:
        # Call LLM - returns JSON
        data = chat_json(prompt, system=SYSTEM, temperature=0.0)
    except Exception as e:
        # Fallback if LLM fails
        return _fallback(raw, error=str(e))
    
    # Normalize and merge with caller-provided fields
    # (caller-provided fields win - they're source of truth)
    out = {k: data.get(k) for k in EXPECTED_KEYS}
    if not out.get("customer_name") and raw.get("customer_name"):
        out["customer_name"] = raw["customer_name"]
    
    return out
```

---

### Agent 2: CLASSIFIER - The Categorizer

#### What it does

Assigns three labels to each complaint:

```python
# Input
complaint = {
    "complaint_text": "My UPI payment failed after 5 attempts!",
    "amount_involved": 5000,
    "channel": "twitter",
    "account_type": "savings"
}

# Output
classification = {
    "category": "UPI",        # 6 possible values
    "severity": "High",       # 4 possible values
    "sentiment": "Frustrated",# 4 possible values
    "rationale": "Repeated failures indicate persistent issue"
}
```

#### Category Definitions

```
CATEGORIES = ["UPI", "ATM", "Card", "Loan", "NetBanking", "General"]

UPI:
  - UPI payment failures
  - Failed transfers
  - Wrong UPI handles
  
ATM:
  - Cash not dispensed
  - Card retained
  - Wrong amount dispensed
  
Card:
  - Credit card issues
  - Debit card blocked
  - Card not working
  
Loan:
  - Loan inquiry
  - EMI issues
  - Loan application
  
NetBanking:
  - Login issues
  - Transaction failures
  - Balance errors
  
General:
  - Everything else
  - Customer service
  - General inquiries
```

#### Severity Definitions

```
SEVERITIES = ["Critical", "High", "Medium", "Low"]

Critical:
  - Fraud/unauthorized debit
  - Account locked out
  - Large amount ≥ 50000 INR
  
High:
  - Amount stuck issue
  - Repeated failures
  - Account access issues
  
Medium:
  - Typical service issues
  - Single failed transaction
  - Delays
  
Low:
  - Information requests
  - Minor inconvenience
```

#### Sentiment Definitions

```
SENTIMENTS = ["Angry", "Frustrated", "Neutral", "Polite"]

Angry:
  - Aggressive language
  - Uses CAPS/exclamations
  - Threats to escalate
  - "Unacceptable!", "Worst service!"

Frustrated:
  - Tired tone
  - Repeated attempts mentioned
  - Demands action
  - "I've tried 5 times!"

Neutral:
  - Factual tone
  - No strong emotion
  - Just reporting the issue

Polite:
  - Courteous phrasing
  - Uses "please", "kindly"
  - "Would you please help"
```

#### The Classifier Prompt

```python
SYSTEM = """
You are a complaint classification agent for an Indian bank.
You assign a single category, severity, and sentiment label to each complaint.
Respond with JSON only.
"""

PROMPT = """Classify the complaint below.

Category MUST be one of: {cats}
Severity MUST be one of: {sevs}
  - Critical: fraud, unauthorized debit, locked out, large amounts >= 50000
  - High: amount-stuck issues, repeated failures, account access issues
  - Medium: typical service issues, single failed transaction, delays
  - Low: information requests, minor inconvenience
Sentiment MUST be one of: {sens}
  - Angry: aggressive, uses caps/exclamations, threats to escalate
  - Frustrated: tired, repeated attempts, demands action
  - Neutral: factual, no strong emotion
  - Polite: courteous phrasing

Return ONLY this JSON shape:
{{"category":"...","severity":"...","sentiment":"...","rationale":"one short sentence"}}

Complaint text:
\"\"\"{text}\"\"\"
Amount involved (INR): {amount}
Channel: {channel}
Account type: {account_type}
```

#### Keyword Fallback (When LLM Fails)

```python
# agents/classifier.py - Fallback logic

_CAT_PATTERNS = [
    ("UPI",        re.compile(r"\bUPI\b|upi", re.I)),
    ("ATM",        re.compile(r"\bATM\b|cash.{0,15}machine", re.I)),
    ("Card",       re.compile(r"credit.?card|debit.?card|\bcard\b", re.I)),
    ("Loan",       re.compile(r"\bloan\b|EMI|home.?loan|personal.?loan", re.I)),
    ("NetBanking", re.compile(r"net.?banking|mobile.?banking|internet.?banking", re.I)),
]

_HIGH_KEYWORDS = re.compile(r"unauthor|fraud|stuck|debit.{0,15}fail|debited", re.I)
_ANGRY_KEYWORDS = re.compile(r"unacceptable|disgust|outrage|terrible|worst|!!", re.I)
_POLITE_KEYWORDS = re.compile(r"kindly|please|would you|appreciate|sir/madam|कृपया", re.I)

def _fallback(complaint):
    text = complaint.get("complaint_text", "") or ""
    
    # Category: find first matching pattern
    category = "General"
    for cat, pat in _CAT_PATTERNS:
        if pat.search(text):
            category = cat
            break
    
    # Severity: based on amount and keywords
    amt = complaint.get("amount_involved") or 0
    if amt and amt >= 50000:
        severity = "Critical"
    elif _HIGH_KEYWORDS.search(text):
        severity = "High"
    elif amt and amt > 0:
        severity = "Medium"
    else:
        severity = "Low"
    
    # Sentiment: based on exclamation marks and keywords
    if _ANGRY_KEYWORDS.search(text):
        sentiment = "Angry"
    elif _POLITE_KEYWORDS.search(text):
        sentiment = "Polite"
    elif "!" in text or "?" in text:
        sentiment = "Frustrated"
    else:
        sentiment = "Neutral"
    
    return {"category": category, "severity": severity, "sentiment": sentiment}
```

---

### Agent 3: DUPLICATE DETECTOR - The Semantic Matcher

#### The Problem with Simple Matching

```
Complaint 1: "UPI payment failed"
Complaint 2: "UPI payment not working"

Simple match: Different (no exact match)
Semantic match: Same (meaning is similar)
```

#### How Semantic Search Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SEMANTIC SEARCH PROCESS                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 1: CONVERT TEXT TO VECTOR                                        │
│  ────────────────────────────                                          │
│                                                                         │
│  Text: "UPI payment failed"                                           │
│         │                                                               │
│         ▼ sentence-transformers model                                  │
│         ┌─────────────────────────────────────┐                        │
│         │ all-MiniLM-L6-v2 (384 dimensions)   │                        │
│         │                                     │                        │
│         │ [0.12, -0.34, 0.56, 0.23, ..., 0.78]│                        │
│         │  ↑                                   │                        │
│         │ semantic meaning encoded as         │                        │
│         │ numbers                              │                        │
│         └─────────────────────────────────────┘                        │
│                                                                         │
│  Step 2: STORE IN VECTOR DATABASE                                     │
│  ──────────────────────────────────                                     │
│                                                                         │
│  ChromaDB collection: "complaints"                                     │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ ID         │ Embedding                              │ Metadata   │   │
│  ├────────────┼────────────────────────────────────────┼────────────┤   │
│  │ UBI-0001   │ [0.12, -0.34, 0.56, ...]              │ customer: X│   │
│  │ UBI-0002   │ [0.45, 0.12, -0.23, ...]              │ customer: Y│   │
│  │ UBI-0003   │ [0.78, -0.56, 0.34, ...]              │ customer: X│   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Step 3: QUERY FOR DUPLICATES                                          │
│  ─────────────────────────                                              │
│                                                                         │
│  New complaint from "Pooja Mishra":                                    │
│  "UPI payment failed again"                                            │
│         │                                                               │
│         ▼ embed                                                         │
│  Query vector: [0.11, -0.33, 0.55, ...]                                │
│         │                                                               │
│         ▼ ChromaDB query (cosine similarity)                          │
│  Only search: customer_name = "Pooja Mishra"                           │
│         │                                                               │
│         ▼                                                               │
│  Results:                                                              │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ ID         │ Similarity │ Content                             │    │
│  ├────────────┼────────────┼─────────────────────────────────────┤    │
│  │ UBI-0001   │ 0.82       │ "UPI payment failed"  ← DUPLICATE  │    │
│  │ UBI-0003   │ 0.65       │ "My UPI is not working"            │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Since 0.82 > 0.78 threshold: is_duplicate = True                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Why Use Sentence Transformers?

```python
# The model: all-MiniLM-L6-v2
# 
# Why this model?
# 1. Fast: Optimized for speed
# 2. Small: 384 dimensions vs 768/1024 in larger models
# 3. Good quality: Good enough for duplicate detection
# 4. Pre-trained: Already knows language semantics

# Example embeddings:
text1 = "UPI payment failed"
text2 = "UPI payment not working" 
text3 = "ATM card is blocked"

# These are CLOSE (same category):
print(cosine_similarity(embed(text1), embed(text2)))  # ~0.82

# These are FAR (different category):
print(cosine_similarity(embed(text1), embed(text3)))  # ~0.45
```

#### The ChromaDB Setup

```python
# agents/duplicate_detector.py

CHROMA_DIR = Path("data/chroma_db")
MODEL_NAME = "all-MiniLM-L6-v2"
DUP_THRESHOLD = 0.78  # cosine similarity

def _get_collection():
    import chromadb
    
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Cosine similarity space
    collection = client.get_or_create_collection(
        name="complaints",
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def find_duplicate(complaint):
    # 1. Embed the new complaint
    text = complaint.get("complaint_text", "") or ""
    query_vector = embed(text)
    
    # 2. Search ONLY same customer
    customer = complaint.get("customer_name") or ""
    
    # 3. Query ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=5,
        where={"customer_name": customer}  # Filter by customer
    )
    
    # 4. Find best match
    best_similarity = 0
    best_id = None
    
    for id, distance in zip(results["ids"][0], results["distances"][0]):
        if id == complaint.get("id"):
            continue  # Don't match self
        similarity = 1 - distance  # Convert distance to similarity
        if similarity > best_similarity:
            best_similarity = similarity
            best_id = id
    
    # 5. Return result
    is_duplicate = best_similarity >= DUP_THRESHOLD
    
    return {
        "is_duplicate": is_duplicate,
        "duplicate_of": best_id if is_duplicate else None,
        "similarity": round(best_similarity, 4)
    }
```

---

### Agent 4: RESPONSE DRAFTER - The Professional Writer

#### Why This Agent Exists

Bank responses must be:
- **Professional**: Represent the bank
- **RBI-compliant**: Follow regulations
- **Accurate**: Don't make false promises
- **Empathetic**: Acknowledge customer frustration
- **Actionable**: Give clear next steps

#### The System Prompt (Instructions to LLM)

```python
SYSTEM = (
    "You are a professional customer service writer for Union Bank of India. "
    "You draft empathetic, RBI-compliant replies that acknowledge the issue, "
    "give a clear next step with a reference / SLA window, and end politely. "
    "Never promise refunds without investigation. Never share OTP / credentials. "
    "If complaint involves potential fraud, advise blocking card / freezing account "
    "and call the 24x7 helpline 1800-22-2244. Keep replies under 130 words."
)
```

**Key constraints in the system prompt:**
1. **Acknowledge issue**: Show empathy
2. **RBI-compliant**: Follow regulations
3. **Clear next step**: Actionable advice
4. **Reference number**: Trackable
5. **SLA window**: Set expectations
6. **Never promise refunds**: Investigation first
7. **Never share OTP/credentials**: Security
8. **Fraud advice**: Block card, helpline
9. **Under 130 words**: Concise

#### Example Responses

**English (Twitter, Frustrated):**
```
@rahul_sharma Sorry for the inconvenience. Your UPI issue (UBI/UBI-0002) 
is with our tech team. Call 1800-22-2244 for urgent help. 
We resolve UPI issues in 4 business days.
```

**Hindi (Email, Polite):**
```
Dear Rahul,

Thank you for contacting Union Bank of India regarding your UPI concern.
We have registered your complaint under reference UBI/UBI-0002 and our 
technical team is investigating the issue.

For urgent assistance, please call our 24x7 helpline 1800-22-2244.

We aim to resolve this within 5 business days.

Regards,
Union Bank Customer Care
```

**Marathi (WhatsApp, Neutral):**
```
Namaste Rahul,

Thank you for contacting Union Bank. Your complaint (UBI/UBI-0003) 
regarding UPI is noted. Our team will investigate.

For immediate assistance, call 1800-22-2244.

Regards,
Union Bank Customer Care
```

---

### Agent 5: SLA MONITOR - The Deadline Calculator

#### Two Responsibilities

1. **Calculate due date**: When must this complaint be resolved?
2. **Predict breach probability**: Will it be resolved on time?

#### SLA Rules (from RBI Regulations)

```python
# data/sla_rules.json
{
    "sla_days": {
        "UPI": 5,        # Base: 5 days
        "ATM": 5,        # Base: 5 days
        "Card": 7,      # Base: 7 days
        "NetBanking": 7,# Base: 7 days
        "Loan": 30,     # Base: 30 days
        "General": 30   # Base: 30 days
    },
    "severity_multiplier": {
        "Critical": 0.5,  # Half the time!
        "High": 0.7,
        "Medium": 1.0,    # Normal
        "Low": 1.2        # More lenient
    }
}
```

#### Example Calculation

```python
# Complaint: UPI issue, High severity, filed on June 1

# Step 1: Base SLA
base_sla = 5  # days for UPI

# Step 2: Apply severity multiplier
multiplier = 0.7  # High severity
effective_sla = 5 * 0.7 = 3.5 → 4 days

# Step 3: Calculate due date
filed_date = June 1
due_date = June 1 + 4 days = June 5

# Step 4: ML model predicts breach probability
# Using features: severity, sentiment, amount, channel, etc.
breach_probability = 0.35  # 35% chance of breaching
```

#### ML Model: What Goes In/Out

```python
# Features for SLA breach prediction
features = {
    # Categorical (one-hot encoded)
    "channel": "twitter",
    "language": "english", 
    "account_type": "savings",
    "category": "UPI",
    "severity": "High",
    "sentiment": "Frustrated",
    
    # Numeric
    "amount_involved": 5000,
    "complaint_text_length": 156,
    "complaint_word_count": 23,
    "hours_since_filed": 72.5,
    "day_of_week": 3,  # Wednesday
    "is_weekend_filed": 0,
    "is_high_amount": 1,  # >= 25000
    "is_high_value": 0,   # > 50000
    "is_fraud_keyword": 1,  # contains "fraud" etc
    "is_duplicate": 0,
    "is_repeat_customer": 1,
    "has_amount": 1,
    "channel_risk": 1,  # twitter = high visibility
    "sentiment_score": 3,  # Frustrated = 3
    "severity_score": 2,   # High = 2
    "customer_complaint_count": 2,
    "days_to_sla": 4,
    "pct_sla_elapsed": 0.75  # 72.5 hours / (4*24) = 0.75
}

# Output
prediction = {
    "breach_probability": 0.35,  # 35% chance
    "sla_due_date": "2026-06-05",
    "sla_days": 4
}
```

---

### Agent 6: ROOT CAUSE - The Systemic Issue Detector

#### The Problem

```
Individual complaints:
- "UPI failed in Nagpur"
- "UPI not working in Nagpur"  
- "Nagpur UPI issue"
- ... 47 more

Traditional system sees: 50 separate complaints
ComplaintIQ sees: ONE systemic issue affecting Nagpur
```

#### How KMeans Clustering Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    KMEANS CLUSTERING PROCESS                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 1: GET ALL EMBEDDINGS                                            │
│  ─────────────────────────                                              │
│                                                                         │
│  ┌──────────────────────────────────────────────────┐                 │
│  │ complaint_id │ embedding                          │                 │
│  ├─────────────┼─────────────────────────────────────┤                 │
│  │ UBI-0001    │ [0.12, -0.34, 0.56, ...]          │                 │
│  │ UBI-0002    │ [0.15, -0.31, 0.53, ...]          │                 │
│  │ UBI-0003    │ [0.78, -0.12, 0.23, ...]          │                 │
│  │ ...         │ ...                                │                 │
│  │ UBI-1000    │ [0.11, -0.33, 0.55, ...]          │                 │
│  └──────────────────────────────────────────────────┘                 │
│                  │                                                       │
│                  ▼ (1000 complaints → 1000 vectors)                    │
│                                                                         │
│  Step 2: RUN KMEANS (k=12)                                             │
│  ─────────────────────────                                              │
│                                                                         │
│  KMeans finds 12 groups of similar complaints:                         │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │ Cluster 0: 89 complaints (General, various cities)    │           │
│  │ Cluster 1: 45 complaints (ATM, Mumbai)                │           │
│  │ Cluster 2: 78 complaints (Card, Delhi)               │           │
│  │ Cluster 3: 47 complaints (UPI, Nagpur) ◀ ALERT       │           │
│  │ Cluster 4: 62 complaints (Loan, various)            │           │
│  │ ...                                                   │           │
│  │ Cluster 11: 34 complaints (NetBanking, Mumbai)       │           │
│  └─────────────────────────────────────────────────────────┘           │
│                  │                                                       │
│                  ▼                                                       │
│  Step 3: ANALYZE EACH CLUSTER                                          │
│  ─────────────────────────                                              │
│                                                                         │
│  For Cluster 3 (47 UPI complaints):                                   │
│  ┌───────────────────────────────────────────────────────┐             │
│  │ Category analysis:                                   │             │
│  │   UPI: 37/47 = 78% (dominant!)                       │             │
│  │   ATM: 6/47 = 13%                                    │             │
│  │   Other: 4/47 = 9%                                   │             │
│  │                                                       │             │
│  │ Location analysis:                                    │             │
│  │   Nagpur: 23/47 = 49% (dominant!)                   │             │
│  │   Mumbai: 15/47 = 32%                                │             │
│  │   Pune: 9/47 = 19%                                   │             │
│  │                                                       │             │
│  │ Decision: 78% UPI + 49% Nagpur = SYSTEMIC ISSUE      │             │
│  └───────────────────────────────────────────────────────┘             │
│                  │                                                       │
│                  ▼                                                       │
│  Step 4: GENERATE ALERT                                                │
│  ─────────────────────                                                  │
│                                                                         │
│  Alert:                                                                 │
│  ┌────────────────────────────────────────────────────────┐            │
│  │ Cluster #3 - SYSTEMIC ISSUE                           │            │
│  │ Category: UPI (78% of cluster)                        │            │
│  │ Location: Nagpur (49% of cluster)                      │            │
│  │ Count: 47 complaints                                  │            │
│  │ Top cities: Nagpur (23), Mumbai (15), Pune (9)        │            │
│  │ Summary: 47 UPI complaints concentrated in Nagpur     │            │
│  │          - possible local UPI service issue          │            │
│  └────────────────────────────────────────────────────────┘            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Alert Thresholds

```python
# agents/root_cause.py

MIN_CLUSTER_SIZE = 5      # At least 5 complaints
DOMINANCE_THRESHOLD = 0.60  # 60% must be same category

def detect(k=12):
    # Run KMeans
    labels = kmeans.fit_predict(embeddings)
    
    for cluster_id in range(k):
        cluster_indices = where(labels == cluster_id)
        cluster_size = len(cluster_indices)
        
        if cluster_size < MIN_CLUSTER_SIZE:
            continue  # Too small
        
        # Analyze cluster
        categories = [metas[i]["category"] for i in cluster_indices]
        locations = [metas[i]["location"] for i in cluster_indices]
        
        most_common_category = Counter(categories).most_common(1)[0]
        category_share = most_common_category[1] / cluster_size
        
        # If dominant category > 60% → ALERT
        if category_share >= DOMINANCE_THRESHOLD:
            alert = {
                "cluster_id": cluster_id,
                "category": most_common_category[0],
                "count": cluster_size,
                "category_share": category_share,
                # ... location analysis
            }
            alerts.append(alert)
    
    return alerts
```

---

## 5. ML Models: The Mathematical Intuition

### Model 1: SLA Breach Predictor (XGBoost)

#### What is XGBoost?

**XGBoost = eXtreme Gradient Boosting**

Think of it like building a team of decision makers:

```
Individual decision trees:
├── Tree 1: "If severity=Critical AND breach_prob > 0.5 → High risk"
├── Tree 2: "If amount > 25000 AND is_repeat_customer → High risk"  
└── Tree 3: "If channel=twitter AND sentiment=Angry → High risk"

XGBoost combines all trees, weighted by their accuracy
```

#### Gradient Boosting Intuition

```python
# Simplified concept of gradient boosting

# Start with a simple prediction (mean)
prediction = 0.18  # 18% base breach rate

# Calculate error for each sample
errors = actual_breach - prediction  # [-0.18, 0.82, -0.18, ...]

# Train next tree to predict these errors
tree2 = train_tree(features, errors)

# Update prediction
prediction = prediction + learning_rate * tree2.predict(features)

# Repeat this 400 times (n_estimators=400)
# Final prediction = sum of all trees
```

#### Feature Importance (What Matters Most)

From the trained model, the top features are:

```
Top 10 Features for SLA Breach Prediction:
1.  pct_sla_elapsed     (how much of SLA time has passed)
2.  days_to_sla         (total SLA window)
3.  hours_since_filed   (time since complaint)
4.  severity_score      (Critical=4, High=3, Medium=2, Low=1)
5.  customer_complaint_count (repeat customer)
6.  sentiment_score     (Angry=4, Frustrated=3, Neutral=2, Polite=1)
7.  is_duplicate        (duplicates get handled faster)
8.  is_fraud_keyword    (fraud cases get priority)
9.  channel_risk        (Twitter/WhatsApp = public pressure)
10. is_high_amount     (large amounts get attention)
```

**Intuition:**
- The more SLA time elapsed, the more likely to breach
- Repeat customers are more likely to churn/escalate
- Public channels (Twitter) create pressure to resolve faster
- High-severity complaints get prioritized

#### Model Performance

```python
# From training results
{
    "winner": "XGBoost(tuned)",
    "cv_auc_mean": 0.9233,      # 5-fold cross-validation AUC
    "cv_auc_std": 0.018,
    "holdout_auc": 0.9378,       # Held-out test set AUC
    "holdout_accuracy": 0.864    # 86.4% accuracy
}
```

**What do these numbers mean?**
- **AUC = 0.94**: If we pick a random breaching complaint and a random 
  non-breaching complaint, the model scores the breaching one higher 94% of time
- **Accuracy = 86.4%**: Overall correct predictions

---

### Model 2: Category Classifier (TF-IDF + Logistic Regression)

#### What is TF-IDF?

**TF-IDF = Term Frequency - Inverse Document Frequency**

Measures how important a word is to a document in a collection:

```
TF-IDF(word, document, corpus) = 
    TF(word, document) × IDF(word, corpus)

Where:
- TF(word, document) = count of word in document / total words
- IDF(word, corpus) = log(total documents / documents containing word)
```

**Example:**
```
Documents:
1. "UPI payment failed"
2. "UPI not working"
3. "ATM cash not dispensed"

Word "UPI":
- TF in doc1: 1/3 = 0.33
- IDF: log(3/2) = 0.405
- TF-IDF: 0.33 × 0.405 = 0.134

Word "payment":
- TF in doc1: 1/3 = 0.33
- IDF: log(3/1) = 1.099
- TF-IDF: 0.33 × 1.099 = 0.363
```

**So "payment" is more distinctive than "UPI" for categorization!**

#### TF-IDF Process

```python
# Input: complaint text
text = "UPI payment failed, amount got debited"

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform([text])

# Result: sparse matrix of shape (1, 5000)
# Non-zero values at indices corresponding to words
# [
#     0,    # "atm" = 0
#     0,    # "card" = 0
#     0.234, # "failed" = 0.234
#     0.156, # "payment" = 0.156
#     0.412, # "upi" = 0.412
#     ...
# ]
```

#### What is Logistic Regression?

Despite the name, it's a **classifier** (not regression):

```python
# Binary logistic regression
# P(breach=1) = 1 / (1 + e^-(b0 + b1*x1 + b2*x2 + ...))

# For multiclass (6 categories), we train 6 binary classifiers
# or use softmax activation

# Example: Category prediction
logistic = LogisticRegression()
logistic.fit(X_train, y_train)  # Learn weights

# Predict
prediction = logistic.predict(X_new)  # "UPI"
probabilities = logistic.predict_proba(X_new)  # [0.01, 0.94, 0.02, 0.01, 0.01, 0.01]
```

#### Why This Works

```
TF-IDF + Logistic Regression is a classic combination because:

1. TF-IDF converts text → meaningful numbers
2. LogReg learns boundaries between categories
3. Fast to train and predict
4. Interpretable (which words matter for which category)
5. Works well with clear categories (UPI vs ATM vs Card)

Category-specific words:
- UPI: "upi", "payment", "transfer", "failed"
- ATM: "atm", "cash", "machine", "dispense"
- Card: "card", "credit", "debit", "blocked"
- Loan: "loan", "emi", "interest", "principal"
- NetBanking: "login", "internet", "password", "balance"
```

#### Model Performance

```
Category Classifier Results:
- Accuracy: 97%
- Agreement with LLM: 98%

Confusion Matrix:
              Predicted
              UPI  ATM  Card Loan NetBanking General
Actual UPI    168   2    3    1      0         1
       ATM    3   89   2    0      1         0
       Card   2   1  100   1      1         0
       Loan   0   0    1  128     2         1
       NetB   1   1    0    2    124        2
       Gen    1   0    1    2      3      355
```

---

### Model 3: Priority Scorer (Gradient Boosting)

#### What is Gradient Boosting?

Similar to XGBoost but slightly different implementation:

```python
# Gradient Boosting regressor
# Goal: predict priority score (0-100)

# Features
X = {
    "severity_encoded": 2,       # High = 2
    "sentiment_encoded": 3,       # Frustrated = 3  
    "amount_involved": 5000,
    "customer_complaint_count": 2,
    "days_since_filed": 3,
    "is_duplicate": 0,
    "breach_probability": 0.35
}

# Model predicts
priority_score = 72
```

#### Why Gradient Boosting for Priority?

```
Priority is a SCORE (regression), not a category (classification).

Why GBM over linear regression?
- Non-linear relationships
- Handles missing values
- Feature interactions (severity + amount together matter more)
- Robust to outliers

Features interact:
- severity=High + amount=50000 → higher priority than either alone
- sentiment=Angry + channel=twitter → highest priority
```

#### Model Performance

```
Priority Scorer Results:
- R² = 0.997  (explains 99.7% of variance)
- MAE = 0.50  (average error is 0.5 points on 0-100 scale)
```

---

### Model 4: Sentiment Analysis (RoBERTa)

#### What is RoBERTa?

**RoBERTa = Robustly Optimized BERT Pretraining Approach**

A transformer model pre-trained on massive text data:

```python
# Pre-trained model from HuggingFace
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# This model is already trained on millions of tweets
# to classify sentiment: Positive, Neutral, Negative

# Example:
text = "UPI payment failed after 5 attempts!!! So frustrated!"
result = pipeline(text)
# {'label': 'negative', 'score': 0.92}
```

#### Why Use a Pre-trained Model?

```
Training a sentiment model from scratch:
- Need 100,000+ labeled examples
- Need GPU for training
- Takes hours/days
- Need domain expertise

Using pre-trained RoBERTa:
- Already trained on 58M tweets
- Fine-tuned for sentiment
- Just download and use
- Runs on CPU
- Takes 5 minutes to download
```

#### Mapping to LLM Categories

```
RoBERTa outputs:  Positive | Neutral | Negative
LLM outputs:      Polite   | Neutral | Angry/Frustrated

Mapping:
RoBERTa Positive  → LLM Polite
RoBERTa Neutral   → LLM Neutral  
RoBERTa Negative  → LLM Angry/Frustrated
```

---

## 6. The Vector Database: ChromaDB Explained

### What is ChromaDB?

```
Traditional database: Stores exact values
- SQL: "Find customer with name = 'Rahul'"
- Exact match only

Vector database: Stores semantic meaning
- "Find complaints similar to 'UPI failed'"
- Semantic similarity search
```

### How ChromaDB Works

```python
import chromadb

# Initialize persistent database
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection with cosine similarity
collection = client.create_collection(
    name="complaints",
    metadata={"hnsw:space": "cosine"}  # Cosine similarity
)

# Add embeddings
collection.add(
    ids=["UBI-0001", "UBI-0002"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],  # 384-dim vectors
    documents=["UPI failed", "ATM blocked"],
    metadatas=[
        {"customer": "Rahul", "category": "UPI"},
        {"customer": "Priya", "category": "ATM"}
    ]
)

# Query similar documents
results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],
    n_results=2,
    where={"customer": "Rahul"}  # Filter: only same customer
)

# Results contain IDs, distances, documents, metadatas
```

### Why ChromaDB over alternatives?

| Feature | ChromaDB | Pinecone | FAISS |
|---------|----------|----------|-------|
| Local/Persistent | ✅ | ❌ | ✅ |
| Metadata filtering | ✅ | ✅ | ❌ (hard) |
| Free | ✅ | ❌ (paid) | ✅ |
| Easy setup | ✅ | ✅ | ❌ (complex) |
| Python-native | ✅ | ✅ | ❌ (C++) |

---

## 7. Root Cause Detection: KMeans Clustering Deep Dive

### What is KMeans?

```python
# KMeans algorithm
# 1. Pick k random centroids
# 2. Assign each point to nearest centroid
# 3. Recalculate centroids
# 4. Repeat until convergence

from sklearn.cluster import KMeans

# Example: Cluster 2D points
points = [
    [1, 2], [1, 4], [1, 0],    # Cluster A
    [10, 2], [10, 4], [10, 0], # Cluster B
]

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(points)

# labels = [0, 0, 0, 1, 1, 1]  - correctly identified!
```

### Application to Complaint Analysis

```python
# Step 1: Get all complaint embeddings
all_embeddings = []  # 1000 × 384 matrix
all_metadata = []    # 1000 dicts

for complaint in db.list_complaints():
    embedding = embed(complaint["complaint_text"])
    all_embeddings.append(embedding)
    all_metadata.append({
        "customer": complaint["customer_name"],
        "category": complaint["category"],
        "location": complaint["location"]
    })

# Step 2: Run KMeans
kmeans = KMeans(n_clusters=12, random_state=42)
cluster_labels = kmeans.fit_predict(all_embeddings)

# Step 3: Analyze each cluster
for cluster_id in range(12):
    # Get complaints in this cluster
    cluster_mask = cluster_labels == cluster_id
    cluster_complaints = [all_metadata[i] for i in range(len(cluster_labels)) if cluster_mask[i]]
    
    # Analyze category distribution
    categories = [c["category"] for c in cluster_complaints]
    most_common = Counter(categories).most_common(1)[0]
    
    # If dominant category > 60%, trigger alert
    if most_common[1] / len(cluster_complaints) > 0.60:
        generate_alert(cluster_id, most_common[0])
```

### Why k=12?

```python
# Rule of thumb: k ≈ √(n/2)
# For 1000 complaints: k ≈ √500 ≈ 22

# But we use k=12 because:
# - Too many clusters → no dominant categories
# - Too few clusters → mixing unrelated complaints
# - k=12 found empirically to work well

# With k=12 and MIN_CLUSTER_SIZE=5:
# - Minimum useful cluster: 5 complaints
# - Maximum clusters with alerts: 12
# - Expected meaningful alerts: 4-8
```

---

## 8. Risk Score: The Weighted Formula Explained

### The Three Sub-Scores

#### 1. Ombudsman Risk (45%)

```python
def _ombudsman_score(complaint, history, breach_prob):
    score = 0
    
    # Base: breach probability (0-40 points)
    score += breach_prob * 40
    
    # Severity contribution
    severity_map = {"Critical": 30, "High": 20, "Medium": 10, "Low": 3}
    score += severity_map.get(complaint["severity"], 10)
    
    # Repeat customer (0-15 points)
    score += min(len(history) * 3, 15)
    
    # Amount stuck (0-12 points)
    amount = complaint.get("amount_involved", 0)
    if amount >= 100000:
        score += 12
    elif amount >= 25000:
        score += 6
    
    # Angry history (0-8 points)
    angry_count = sum(1 for h in history if h.get("sentiment") == "Angry")
    score += min(angry_count * 2, 8)
    
    return min(100, int(score))
```

**Intuition:**
- High breach probability → customer likely to escalate
- Critical severity → already a big issue
- Repeat customer → already unhappy
- Large amount → more upset
- Angry history → escalating pattern

#### 2. Churn Risk (30%)

```python
def _churn_score(complaint, history):
    score = 0
    
    # Number of complaints (0-30 points)
    score += min(len(history) * 6, 30)
    
    # Unresolved issues (0-25 points)
    unresolved = sum(1 for h in history if h.get("status") == "open")
    score += min(unresolved * 5, 25)
    
    # Category breadth (0-20 points)
    categories = set(h.get("category") for h in history)
    score += min(len(categories) * 6, 20)
    
    # Current sentiment
    if complaint.get("sentiment") in ["Angry", "Frustrated"]:
        score += 12
    
    # Current severity
    if complaint.get("severity") in ["Critical", "High"]:
        score += 8
    
    return min(100, int(score))
```

**Intuition:**
- Multiple complaints → unhappy customer
- Unresolved issues → not getting help
- Many categories → systemic problems
- Angry/frustrated → might leave
- High severity → urgent attention needed

#### 3. Social Media Risk (25%)

```python
def _social_score(complaint, history):
    score = 0
    
    # Past public complaints (0-45 points)
    public_channels = {"twitter", "whatsapp"}
    public_complaints = [h for h in history if h.get("channel") in public_channels]
    score += min(len(public_complaints) * 12, 45)
    
    # Current public channel
    if complaint.get("channel") in public_channels:
        score += 18
    
    # Current sentiment
    sentiment_map = {"Angry": 22, "Frustrated": 10}
    score += sentiment_map.get(complaint.get("sentiment"), 0)
    
    # Current severity
    if complaint.get("severity") == "Critical":
        score += 8
    
    return min(100, int(score))
```

**Intuition:**
- Twitter/WhatsApp complaints are public
- Public complaints damage reputation
- Angry sentiment on public channel → viral risk
- Critical severity → high visibility

### Weighted Overall

```python
def compute_overall(ombudsman, churn, social):
    overall = 0.45 * ombudsman + 0.30 * churn + 0.25 * social
    return int(min(100, max(0, overall)))
```

**Why these weights?**
- **Ombudsman (45%)**: Regulatory risk is highest priority
- **Churn (30%)**: Customer retention is important
- **Social (25%)**: Reputation risk, but less controllable

---

## 9. Database Schema Deep Dive

### SQLite Schema

```sql
-- Main complaints table
CREATE TABLE complaints (
    id              TEXT PRIMARY KEY,        -- UBI-0001, UBI-0002, etc.
    customer_name   TEXT NOT NULL,
    channel         TEXT NOT NULL,          -- email, whatsapp, twitter, etc.
    complaint_text  TEXT NOT NULL,
    language        TEXT,                    -- english, hindi, marathi
    date            TEXT NOT NULL,           -- ISO date
    location        TEXT,                    -- city name
    account_type    TEXT,                    -- savings, current, etc.
    amount_involved REAL,                    -- in INR
    
    -- Enriched by agents (nullable until processed)
    category        TEXT,                    -- UPI, ATM, Card, Loan, NetBanking, General
    severity        TEXT,                    -- Critical, High, Medium, Low
    sentiment       TEXT,                    -- Angry, Frustrated, Neutral, Polite
    intake_json     TEXT,                    -- JSON from Agent 1
    
    duplicate_of    TEXT,                    -- ID of duplicate complaint
    similarity      REAL,                    -- similarity score
    
    draft_response  TEXT,                    -- Generated reply from Agent 4
    
    sla_due_date    TEXT,                    -- ISO date
    sla_breach_prob REAL,                    -- 0-1 probability
    
    risk_score      INTEGER,                 -- 0-100 overall
    risk_ombudsman  INTEGER,                 -- 0-100
    risk_churn      INTEGER,                 -- 0-100
    risk_social     INTEGER,                 -- 0-100
    
    cluster_id      INTEGER,                 -- KMeans cluster
    
    status          TEXT DEFAULT 'open',     -- open, resolved, auto_resolved_*
    
    processed_at    TEXT,                    -- ISO timestamp
    
    -- ML second opinions
    ml_category          TEXT,               -- ML-predicted category
    ml_category_prob    REAL,                -- probability
    category_confidence TEXT,               -- High Confidence / Needs Review
    
    ml_sentiment         TEXT,               -- positive/neutral/negative
    ml_sentiment_prob    REAL,
    sentiment_confidence TEXT,
    
    priority_score       INTEGER             -- 0-100
);

-- Indexes for common queries
CREATE INDEX idx_complaints_customer ON complaints(customer_name);
CREATE INDEX idx_complaints_date ON complaints(date);
CREATE INDEX idx_complaints_category ON complaints(category);
CREATE INDEX idx_complaints_location ON complaints(location);
```

### Root Cause Alerts Table

```sql
CREATE TABLE root_cause_alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id  INTEGER,        -- Which KMeans cluster
    category    TEXT,           -- Dominant category
    location    TEXT,           -- Dominant location (if any)
    count       INTEGER,        -- How many complaints in cluster
    summary     TEXT,           -- Human-readable description
    created_at  TEXT            -- ISO timestamp
);
```

### Feedback Table (Human-in-the-loop)

```sql
CREATE TABLE feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_id    TEXT NOT NULL,
    field           TEXT NOT NULL,         -- category/severity/sentiment
    original_value  TEXT,
    corrected_value TEXT,                   -- NULL = marked correct
    is_correct      INTEGER NOT NULL,       -- 1 = correct, 0 = correction
    created_at      TEXT NOT NULL
);
```

---

## 10. Code Flow Analysis

### One Complaint - Full Trace

Let's trace complaint "UBI-0001" through the entire system:

```python
# Step 1: Raw complaint enters
raw = {
    "id": "UBI-0001",
    "customer_name": "Pooja Mishra",
    "channel": "email",
    "complaint_text": "My UPI payment of Rs.3500 failed but amount got debited. I tried 3 times!",
    "language": "english",
    "date": "2026-01-15",
    "location": "Nagpur",
    "account_type": "savings",
    "amount_involved": 3500
}

# Store in database
db.upsert_raw_complaint(raw)

# Step 2: Run pipeline
result = process_one_streaming("UBI-0001")

# Inside process_one_streaming():
# ──────────────────────────

# Agent 1: Intake
intake_out = intake.extract(raw)
# Returns:
# {
#     "customer_name": "Pooja Mishra",
#     "issue_summary": "UPI payment failed, amount debited",
#     "account_type": "savings",
#     "amount_involved": 3500,
#     "transaction_id": None,
#     "location_mentioned": "Nagpur",
#     "urgency_keywords": ["failed", "stuck", "debited"],
#     "detected_language": "english"
# }

# Agent 2: Classifier
cls = classifier.classify(raw)
# Returns:
# {
#     "category": "UPI",
#     "severity": "High",
#     "sentiment": "Frustrated",
#     "rationale": "Repeated failures with amount stuck"
# }

# Agent 3: Duplicate Detector
dup = duplicate_detector.find_duplicate(raw)
# Returns:
# {
#     "is_duplicate": False,
#     "duplicate_of": None,
#     "similarity": 0.0,
#     "neighbours": []
# }
# (No prior complaints from this customer)

# Agent 4: Response Drafter (since not duplicate)
draft = response_drafter.draft(raw, sla_days=4)
# Returns:
# "Dear Pooja Mishra,\n\nThank you for contacting Union Bank of India 
# regarding your UPI concern. We have registered your complaint under 
# reference UBI/UBI-0001 and our technical team is investigating the issue.\n
# For urgent assistance, please call our 24x7 helpline 1800-22-2244.\n\n
# We aim to resolve this within 4 business days.\n\nRegards,\nUnion Bank Customer Care"

# Agent 5: SLA Monitor
sla = sla_monitor.predict_breach(raw)
# Returns:
# {
#     "sla_due_date": "2026-01-19",
#     "sla_days": 4,
#     "breach_probability": 0.35,
#     "model_used": "xgboost_tuned"
# }

# ML Second Opinions
ml_cat = ml_category.predict(raw["complaint_text"])
# Returns:
# {
#     "category": "UPI",
#     "probability": 0.94,
#     "all_probabilities": {"UPI": 0.94, "ATM": 0.02, ...}
# }
# Agreement: "High Confidence" (LLM=UPI, ML=UPI)

ml_sent = sentiment_ml.predict(raw["complaint_text"])
# Returns:
# {
#     "label": "negative",
#     "score": 0.87,
#     "bucket": "negative"
# }
# Agreement: "Needs Review" (LLM=Frustrated → negative, ML=negative)

# Risk Score
history = db.customer_history("Pooja Mishra")  # Empty - first complaint
risk = risk_module.compute(raw, history, breach_prob=0.35)
# Returns:
# {
#     "ombudsman": 38,
#     "churn": 6,
#     "social": 0,
#     "overall": 19
# }

# Priority Score
priority = priority_module.score(raw)
# Returns:
# 68

# Auto-resolution check
# severity = High → not Low/Medium
# sentiment = Frustrated → not Polite
# → status = "open"

# Update database
db.update_complaint(
    "UBI-0001",
    category="UPI",
    severity="High",
    sentiment="Frustrated",
    intake_json=json.dumps(intake_out),
    draft_response=draft,
    sla_due_date="2026-01-19",
    sla_breach_prob=0.35,
    risk_score=19,
    risk_ombudsman=38,
    risk_churn=6,
    risk_social=0,
    ml_category="UPI",
    ml_category_prob=0.94,
    category_confidence="High Confidence",
    ml_sentiment="negative",
    ml_sentiment_prob=0.87,
    sentiment_confidence="Needs Review",
    priority_score=68,
    status="open",
    processed_at="2026-01-15T10:30:00Z"
)

# Index for future duplicate detection
duplicate_detector.index_complaint(raw)

# ──────────────────────────
# Pipeline complete!

# Final database row:
# {
#     "id": "UBI-0001",
#     "customer_name": "Pooja Mishra",
#     "channel": "email",
#     "complaint_text": "My UPI payment of Rs.3500 failed but amount got debited...",
#     "language": "english",
#     "date": "2026-01-15",
#     "location": "Nagpur",
#     "account_type": "savings",
#     "amount_involved": 3500,
#     "category": "UPI",
#     "severity": "High",
#     "sentiment": "Frustrated",
#     "draft_response": "Dear Pooja Mishra...",
#     "sla_due_date": "2026-01-19",
#     "sla_breach_prob": 0.35,
#     "risk_score": 19,
#     "priority_score": 68,
#     "status": "open",
#     ...
# }
```

---

## 11. Why Each Choice Over Alternatives

### LLM: Groq vs OpenAI vs Local Models

| Factor | Groq (Llama 3.3) | OpenAI (GPT-4) | Local (Llama 3 70B) |
|--------|------------------|----------------|---------------------|
| Speed | ⚡ Very fast | 🐢 Slow | 🐢 Very slow |
| Cost | 🆓 Free tier | 💰 Expensive | ⚠️ Hardware cost |
| Quality | ✅ Good | ✅ Best | ✅ Good |
| Setup | ✅ Easy | ✅ Easy | ❌ Complex |
| Latency | ~500ms | ~3000ms | ~30000ms |

**Decision: Groq**
- Best speed/cost for classification tasks
- Good enough quality for structured output
- Free tier sufficient for demo/project

### Vector Store: ChromaDB vs Pinecone vs Weaviate

| Factor | ChromaDB | Pinecone | Weaviate |
|--------|----------|----------|----------|
| Deployment | Local | Cloud | Local/Docker |
| Cost | 🆓 Free | 💰 Paid | 🆓 Free |
| Filtering | ✅ Easy | ✅ Easy | ✅ |
| Persistence | ✅ | ✅ | ✅ |
| Setup | 5 min | 10 min | 30 min |

**Decision: ChromaDB**
- Free, local, persistent
- Easy metadata filtering
- Simple Python API

### ML Models: Classical vs Deep Learning

| Factor | Classical (XGB, LogReg) | Deep Learning (BERT, etc.) |
|--------|-------------------------|---------------------------|
| Data needed | 1,000s | 100,000s |
| Training time | Minutes | Hours/Days |
| Inference speed | Fast (ms) | Slow (seconds) |
| GPU required | Optional | Required |
| Interpretability | ✅ Feature importance | ❌ Black box |
| Accuracy | ✅ Good enough | ✅ Slightly better |

**Decision: Classical ML**
- Data size (1000) is enough for classical ML
- Interpretable (understand why prediction made)
- Fast inference
- No GPU required
- Good enough accuracy (94% AUC)

### Dashboard: Streamlit vs Dash vs React

| Factor | Streamlit | Dash | React + Flask |
|--------|-----------|------|---------------|
| Development speed | ⚡ Fast | 😐 Medium | ❌ Slow |
| Python-only | ✅ | ⚠️ Partial | ❌ Need JS |
| Customization | Limited | Good | Unlimited |
| Learning curve | Low | Medium | High |

**Decision: Streamlit**
- Fastest development
- Native data science integration
- Good enough for internal tool

---

## 12. Hands-On: Tracing One Complaint

Let's trace a real complaint through the system step-by-step:

### Input Complaint

```json
{
    "id": "UBI-0006",
    "customer_name": "Pooja Mishra",
    "channel": "whatsapp",
    "complaint_text": "UPI payment failed again! This is the 2nd time this week. Amount 3500 debited but not credited. Very frustrated.",
    "language": "english",
    "date": "2026-01-20",
    "location": "Nagpur",
    "account_type": "savings",
    "amount_involved": 3500
}
```

### Step-by-Step Processing

```python
# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: INTAKE AGENT
# ═══════════════════════════════════════════════════════════════════════════

intake_result = {
    "customer_name": "Pooja Mishra",         # From input
    "issue_summary": "UPI payment failed, amount debited but not credited",
    "account_type": "savings",               # From input  
    "amount_involved": 3500,                 # Extracted from text
    "transaction_id": None,                  # Not mentioned
    "location_mentioned": "Nagpur",           # From input
    "urgency_keywords": ["failed", "frustrated", "2nd time"],
    "detected_language": "english"
}

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: CLASSIFIER AGENT
# ═══════════════════════════════════════════════════════════════════════════

# LLM classification
classifier_result = {
    "category": "UPI",           # Contains "UPI payment failed"
    "severity": "High",           # "2nd time" + amount stuck suggests High
    "sentiment": "Frustrated",    # "Very frustrated" = Frustrated
    "rationale": "Repeat UPI failure with amount stuck indicates persistence"
}

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: DUPLICATE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

# Check if same customer had similar complaint
duplicate_result = {
    "is_duplicate": True,                    # YES! 
    "duplicate_of": "UBI-0001",              # First complaint from Pooja
    "similarity": 0.82,                      # 82% similar
    "neighbours": [
        {"id": "UBI-0001", "similarity": 0.82, "meta": {...}}
    ]
}

# This is a duplicate! Skip response drafting.

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: SLA MONITOR
# ═══════════════════════════════════════════════════════════════════════════

sla_result = {
    "sla_due_date": "2026-01-24",     # 5 days × 0.7 (High) = 3.5 → 4 days
    "sla_days": 4,
    "breach_probability": 0.42,       # Higher because duplicate + frustrated
    "model_used": "xgboost_tuned"
}

# ═══════════════════════════════════════════════════════════════════════════
# ML SECOND OPINIONS
# ═══════════════════════════════════════════════════════════════════════════

# ML Category (TF-IDF + LogReg)
ml_category_result = {
    "category": "UPI",
    "probability": 0.91,
    "all_probabilities": {"UPI": 0.91, "ATM": 0.03, ...}
}
# Agreement: HIGH (LLM=UPI, ML=UPI ✓)

# ML Sentiment (RoBERTa)
ml_sentiment_result = {
    "label": "negative",
    "score": 0.89,
    "bucket": "negative"
}
# Agreement: NEEDS REVIEW (LLM=Frustrated → maps to negative, but confidence differs)

# ═══════════════════════════════════════════════════════════════════════════
# RISK SCORE
# ═══════════════════════════════════════════════════════════════════════════

# Get customer's history (UBI-0001)
history = [
    {
        "id": "UBI-0001",
        "date": "2026-01-15",
        "category": "UPI",
        "severity": "High",
        "sentiment": "Frustrated",
        "status": "open"
    }
]

risk_result = {
    "ombudsman": 52,    # breach=0.42×40=17 + severity=20 + count=2×3=6 + angry=2×2=4 + amount=6 = 53
    "churn": 47,        # complaints=1×6 + unresolved=1×5 + categories=1×6 + sentiment=12 + severity=8 = 37
    "social": 32,       # public=1×18 + sentiment=10 + severity=0 = 28
    "overall": 45       # 0.45×52 + 0.30×47 + 0.25×32 = 45
}

# ═══════════════════════════════════════════════════════════════════════════
# AUTO-RESOLUTION LOGIC
# ═══════════════════════════════════════════════════════════════════════════

# Check conditions:
# - is_duplicate = True → DUPLICATE!

# Status = "auto_resolved_dup"
# No new response drafted (use UBI-0001's response)

# ═══════════════════════════════════════════════════════════════════════════
# DATABASE UPDATE
# ═══════════════════════════════════════════════════════════════════════════

db.update_complaint(
    "UBI-0006",
    category="UPI",
    severity="High", 
    sentiment="Frustrated",
    intake_json="{...}",
    duplicate_of="UBI-0001",
    similarity=0.82,
    draft_response=None,  # Duplicate - use prior response
    sla_due_date="2026-01-24",
    sla_breach_prob=0.42,
    risk_score=45,
    risk_ombudsman=52,
    risk_churn=47,
    risk_social=32,
    ml_category="UPI",
    ml_category_prob=0.91,
    category_confidence="High Confidence",
    ml_sentiment="negative",
    sentiment_confidence="Needs Review",
    priority_score=72,
    status="auto_resolved_dup",  # ← Marked as duplicate
    resolved_at="2026-01-20T14:30:00Z",
    processed_at="2026-01-20T14:30:00Z"
)

# Index in ChromaDB for future duplicate detection
duplicate_detector.index_complaint(complaint)

# ═══════════════════════════════════════════════════════════════════════════
# DONE!
# ═══════════════════════════════════════════════════════════════════════════

# This complaint is now:
# - Categorized as UPI / High / Frustrated
# - Marked as duplicate of UBI-0001
# - Auto-resolved (uses prior response)
# - Has SLA due date and breach probability
# - Has risk score 45/100
```

---

## Summary

### Key Takeaways

1. **AI Agents handle complexity**: LLMs extract info, classify, generate responses
2. **ML models provide verification**: Second opinions catch LLM mistakes
3. **Vector databases enable semantic search**: Find duplicates, detect patterns
4. **Risk scoring enables proactivity**: Know which customers need attention
5. **Auto-resolution saves time**: 18.6% handled automatically
6. **Root cause detection finds systemic issues**: Actionable insights

### The Complete Picture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPLAINTIQ SUMMARY                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT:     1000 complaints across 7 channels, 3 languages                │
│                                                                             │
│  PROCESS:   6 AI Agents + 4 ML Models + Vector Database                     │
│                                                                             │
│  OUTPUT:    Fully enriched complaints with:                                 │
│             • Category, Severity, Sentiment                                 │
│             • SLA due date + breach probability                            │
│             • Customer risk score                                          │
│             • Drafted responses (914 generated, 86 duplicates)            │
│             • Root cause alerts (6 systemic issues)                        │
│                                                                             │
│  RESULTS:                                                                    │
│             • 100% processed                                                │
│             • 18.6% auto-resolved                                          │
│             • SLA breach prediction: 94% AUC                               │
│             • Category accuracy: 97%                                      │
│             • 6 systemic issues detected                                   │
│                                                                             │
│  SPEED:      15 min/complaint → 5 sec/complaint (180x faster)              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

This study guide covers everything you need to understand ComplaintIQ deeply. Each section builds on the previous one, explaining not just WHAT the code does, but WHY it works that way.

---

## 13. The Orchestrator: The Pipeline Conductor

### What is an Orchestrator?

Think of the orchestrator as the **conductor of an orchestra**. Each agent (agents) is a different musician - they all have their own instruments (capabilities), but without a conductor, they can't play together in harmony.

```
Without Orchestrator:
─────────────────────
Intake Agent     → "Here's some data"
Classifier      → "I need input in a specific format"
Duplicate       → "What am I searching for?"
Response Drafter → "When do I start?"
SLA Monitor      → "What category is this?"

EVERY AGENT DOES ITS OWN THING, NO COORDINATION = CHAOS

With Orchestrator:
──────────────────
Orchestrator → "First, Intake, extract fields from this complaint"
           → "Good. Now Classifier, categorize it using that extracted data"
           → "Great. Duplicate Detector, check for matches using the same text"
           → "Since it's not a duplicate, Response Drafter, write a reply"
           → "Finally, SLA Monitor, calculate deadline and risk"
           → "Now save everything to the database"

ALL AGENTS WORK TOGETHER IN ORDER = HARMONY
```

### The Orchestrator Code Flow

```python
# pipeline/orchestrator.py - Simplified process_one_streaming()

def process_one_streaming(complaint_id: str, on_step=None):
    """Process ONE complaint through ALL 6 agents sequentially"""
    
    # STEP 1: Get the raw complaint from database
    row = db.get_complaint(complaint_id)  # e.g., UBI-0001
    
    # STEP 2: AGENT 1 - INTAKE
    # Purpose: Extract structured information from raw text
    emit("Agent 1 (Intake)", "started")
    intake_out = intake.extract(row)  # Returns: {customer_name, amount, etc.}
    emit("Agent 1 (Intake)", "done", intake_out)
    
    # STEP 3: AGENT 2 - CLASSIFIER
    # Purpose: Assign category, severity, sentiment
    emit("Agent 2 (Classifier)", "started")
    cls = classifier.classify(row)     # Returns: {category, severity, sentiment}
    row.update(cls)  # Add classification to the row
    emit("Agent 2 (Classifier)", "done", cls)
    
    # STEP 4: AGENT 3 - DUPLICATE DETECTOR
    # Purpose: Find similar prior complaints from same customer
    emit("Agent 3 (Duplicate Detector)", "started")
    dup = dd.find_duplicate(row)  # Returns: {is_duplicate, duplicate_of, similarity}
    emit("Agent 3 (Duplicate Detector)", "done", dup)
    
    # STEP 5: AGENT 4 - RESPONSE DRAFTER (Conditional)
    # Purpose: Generate bank response
    if dup["is_duplicate"]:
        # Skip - don't generate new response for duplicates
        draft = None
        emit("Agent 4 (Response Drafter)", "skipped", None)
    else:
        emit("Agent 4 (Response Drafter)", "started")
        draft = response_drafter.draft(row, sla_days=4)
        emit("Agent 4 (Response Drafter)", "done", {"draft": draft})
    
    # STEP 6: AGENT 5 - SLA MONITOR
    # Purpose: Calculate deadline and breach probability
    emit("Agent 5 (SLA Monitor)", "started")
    sla = sla_monitor.predict_breach(row)
    row["sla_breach_prob"] = sla["breach_probability"]
    row["duplicate_of"] = dup["duplicate_of"]
    emit("Agent 5 (SLA Monitor)", "done", sla)
    
    # STEP 7: ML SECOND OPINIONS
    # Purpose: Verify LLM classifications with independent models
    
    # ML Category (TF-IDF + Logistic Regression)
    emit("ML Category (TF-IDF + LogReg)", "started")
    ml_cat = ml_category.predict(row.get("complaint_text") or "")
    cat_conf = ml_category.agreement(cls.get("category"), ml_cat["category"])
    emit("ML Category (TF-IDF + LogReg)", "done", {**ml_cat, "agreement": cat_conf})
    
    # ML Sentiment (RoBERTa)
    emit("ML Sentiment (HF Roberta)", "started")
    ml_sent = sentiment_ml.predict(row.get("complaint_text") or "")
    sent_conf = sentiment_ml.agreement(cls.get("sentiment"), ml_sent)
    emit("ML Sentiment (HF Roberta)", "done", {**ml_sent, "agreement": sent_conf})
    
    # STEP 8: COMPUTE RISK AND PRIORITY
    # Purpose: Calculate customer risk score and priority
    
    history = db.customer_history(row.get("customer_name") or "")
    risk = risk_module.compute(row, history, breach_prob=sla["breach_probability"])
    
    emit("ML Priority (Gradient Boosting)", "started")
    priority = priority_module.score({**row, "sla_breach_prob": sla["breach_probability"]})
    emit("ML Priority (Gradient Boosting)", "done", {"priority_score": priority})
    
    # STEP 9: AUTO-RESOLUTION LOGIC
    # Purpose: Decide if complaint can be auto-resolved
    
    if dup["is_duplicate"]:
        status_value = "auto_resolved_dup"
    elif (cls["severity"] in ("Low", "Medium")
          and cls["sentiment"] == "Polite"
          and cls["category"] in ("UPI", "ATM", "Card", "NetBanking", "Loan", "General")):
        status_value = "auto_resolved_std"
    else:
        status_value = "open"
    
    # STEP 10: SAVE TO DATABASE
    db.update_complaint(
        complaint_id,
        category=cls["category"],
        severity=cls["severity"],
        sentiment=cls["sentiment"],
        # ... many more fields
        status=status_value,
        processed_at=datetime.utcnow().isoformat() + "Z",
    )
    
    # STEP 11: INDEX FOR FUTURE DUPLICATE DETECTION
    # Add this complaint to ChromaDB so future complaints can find it
    dd.index_complaint(row)
    
    return {
        "id": complaint_id,
        "intake": intake_out,
        "classification": cls,
        "duplicate": dup,
        "draft_response": draft,
        "sla": sla,
        "risk": risk,
        # ...
    }
```

### Why Streaming Callback (`on_step`)?

The `on_step` parameter lets the **Dashboard show live progress**:

```python
# How the dashboard uses on_step

def on_step(label, status, payload):
    """Called after each agent completes"""
    if status == "started":
        agent_lines.append(f"-> {label}: working...")
    elif status == "skipped":
        agent_lines.append(f"-- {label}: skipped (duplicate)")
    elif status == "done":
        agent_lines[-1] = f"OK {label}: done"
    
    # Update the UI immediately
    status_box.markdown("\n".join(agent_lines))
    progress_bar.progress(min(step_count["i"] / 6, 1.0))

# Run pipeline with live updates
result = process_one_streaming("UBI-0001", on_step=on_step)
```

This creates this visual in the dashboard:

```
-> Agent 1 (Intake): working...
OK Agent 1 (Intake): done
-> Agent 2 (Classifier): working...
OK Agent 2 (Classifier): done
-> Agent 3 (Duplicate Detector): working...
OK Agent 3 (Duplicate Detector): done
-- Agent 4 (Response Drafter): skipped (duplicate)
-> Agent 5 (SLA Monitor): working...
OK Agent 5 (SLA Monitor): done
ALL AGENTS FINISHED -- result below
```

### Batch Processing: process_all()

```python
def process_all(limit=None, draft_response=True, progress=True):
    """Process EVERY unprocessed complaint in the database"""
    
    # Get all unprocessed complaints
    pending = db.list_unprocessed(limit=limit)  # WHERE processed_at IS NULL
    
    summaries = []
    for i, row in enumerate(pending, 1):
        try:
            summaries.append(process_one(row["id"], draft_response=draft_response))
        except Exception as e:
            summaries.append({"id": row["id], "error": str(e)})
        
        # Print progress every 10 complaints
        if progress and (i % 10 == 0 or i == len(pending)):
            print(f"  processed {i}/{len(pending)}")
    
    # After ALL complaints processed, run ROOT CAUSE detection
    alerts = root_cause.detect()           # KMeans clustering
    root_cause.store_alerts(alerts)        # Save to database
    
    return {
        "processed": len(summaries),
        "errors": sum(1 for s in summaries if "error" in s),
        "duplicates_found": sum(1 for s in summaries if s.get("is_duplicate")),
        "alerts_detected": len(alerts),
        "summaries": summaries,
        "alerts": alerts,
    }
```

This is what runs when you execute:
```bash
python -m pipeline.orchestrator
```

### Ingest New Complaints

```python
def ingest_new_complaint(raw: dict) -> str:
    """Insert a NEW complaint, return its ID"""
    
    db.init_db()  # Ensure tables exist
    
    if not raw.get("id"):
        # Auto-generate ID: UBI-0001, UBI-0002, etc.
        existing = db.list_complaints()
        next_n = len(existing) + 1
        raw = {**raw, "id": f"UBI-{next_n:04d}"}
    
    # Default date to today if not provided
    raw.setdefault("date", datetime.utcnow().date().isoformat())
    
    # Insert into database
    db.upsert_raw_complaint(raw)
    return raw["id"]
```

---

## 14. The API: FastAPI Backend

### What is FastAPI?

FastAPI is a **Python web framework** for building APIs (Application Programming Interfaces). It's like a waiter in a restaurant - it takes orders (requests) from customers (other programs) and brings back food (responses).

```
Client (Dashboard) → FastAPI Server → Database
        ↑                        ↓
        └──────── Response ←─────┘
```

### The API Endpoints

```python
# api/main.py - All endpoints

# ENDPOINT 1: Submit a new complaint
@app.post("/complaint")
def submit_complaint(payload: ComplaintIn):
    """Submit a new complaint and run it through all 6 agents"""
    
    # Step 1: Insert into database, get new ID
    new_id = ingest_new_complaint(payload.model_dump())
    
    # Step 2: Process through pipeline
    result = process_one_streaming(new_id)
    
    # Step 3: Optionally refresh root cause clusters
    if payload.refresh_root_cause:
        refresh_root_cause()
    
    return {"id": new_id, **result}


# ENDPOINT 2: List complaints with filters
@app.get("/complaints")
def list_complaints(
    category: str = None,
    severity: str = None,
    channel: str = None,
    customer: str = None,
    limit: int = 50,
):
    """List complaints with optional filters"""
    
    clauses, params = [], []
    if category:   clauses.append("category = ?"); params.append(category)
    if severity:   clauses.append("severity = ?"); params.append(severity)
    if channel:    clauses.append("channel = ?");  params.append(channel)
    if customer:   clauses.append("customer_name = ?"); params.append(customer)
    
    where = " AND ".join(clauses) if clauses else None
    rows = db.list_complaints(limit=limit, where=where, params=params)
    
    return {"count": len(rows), "complaints": rows}


# ENDPOINT 3: Dashboard statistics
@app.get("/stats")
def stats():
    """Dashboard KPI stats"""
    
    rows = db.list_complaints()
    df = pd.DataFrame(rows)
    
    processed = int(df["processed_at"].notna().sum())
    at_risk = int((df["sla_breach_prob"].fillna(0) >= 0.5).sum())
    duplicates = int(df["duplicate_of"].notna().sum())
    avg_breach = float(df["sla_breach_prob"].dropna().mean())
    
    return {
        "total": len(df),
        "processed": processed,
        "pending": len(df) - processed,
        "at_risk": at_risk,
        "duplicates": duplicates,
        "avg_breach_probability": round(avg_breach, 3),
        "by_category": df["category"].value_counts().to_dict(),
        "by_severity": df["severity"].value_counts().to_dict(),
        "by_channel": df["channel"].value_counts().to_dict(),
        "root_cause_alerts": len(db.list_root_cause_alerts()),
    }


# ENDPOINT 4: RBI Compliance Report
@app.get("/report")
def compliance_report(format: str = "csv", today: date = None):
    """Download RBI compliance CSV or JSON"""
    
    rows = db.list_complaints()
    
    if format == "csv":
        csv_bytes = rbi_report.to_csv(rows, today=today)
        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=rbi_compliance.csv"}
        )
    
    # Or JSON format
    df = rbi_report.build_report(rows, today=today)
    return {
        "summary": rbi_report.summary_stats(rows, today=today),
        "report": df.to_dict(orient="records"),
    }
```

### How to Run the API

```bash
# Start the API server
uvicorn api.main:app --reload --port 8000

# Now you can make requests:
# POST /complaint - Submit new complaint
# GET  /complaints - List complaints  
# GET  /stats - Get dashboard stats
# GET  /report - Download CSV

# Access the interactive API documentation:
# http://localhost:8000/docs
```

### Request/Response Models

```python
# Pydantic models define what data looks like

class ComplaintIn(BaseModel):
    """What a new complaint must contain"""
    complaint_text: str = Field(min_length=4, max_length=4000)
    customer_name: str = "Walk-in"
    channel: str = "email"
    language: str = "english"
    account_type: str = "savings"
    location: Optional[str] = None
    amount_involved: Optional[float] = None


class ComplaintOut(BaseModel):
    """What the API returns after processing"""
    id: str
    intake: dict
    classification: dict
    duplicate: dict
    draft_response: Optional[str]
    sla: dict
    risk: dict
```

---

## 15. The LLM Client: Groq Integration

### What is Groq?

Groq is an **AI inference platform** that provides fast access to Large Language Models (LLMs). It's like a super-fast AI brain that can understand and generate text.

```
ComplaintIQ           Groq (Cloud)              LLM (Llama 3.3)
    │                      │                         │
    │─── "Classify this" ─→│                         │
    │                      │─── "Here are the results"→│
    │←──────────────────────│                         │
```

### Why Groq Instead of OpenAI?

| Factor | Groq (Llama 3.3) | OpenAI (GPT-4) |
|--------|-----------------|----------------|
| Speed | ~500ms | ~3000ms |
| Cost | Free tier | Expensive |
| Quality | Good | Best |
| Setup | Easy | Easy |

**Groq wins** for this project because:
1. **Fast** - 6x faster than GPT-4
2. **Free** - Free tier is sufficient
3. **Good enough** - Quality meets requirements

### The LLM Client Code

```python
# agents/llm_client.py

DEFAULT_MODEL = "llama-3.3-70b-versatile"

def get_client():
    """Get or create the Groq client"""
    global _client
    if _client is not None:
        return _client
    
    api_key = _get_api_key()  # Try multiple sources
    from groq import Groq
    _client = Groq(api_key=api_key)
    return _client


def _get_api_key() -> str:
    """Try to get API key from multiple sources"""
    # 1. Environment variable
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    
    # 2. Streamlit secrets (for dashboard)
    import streamlit as st
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    
    # 3. .env file (via python-dotenv)
    raise RuntimeError("GROQ_API_KEY not set")


def chat(prompt: str, system: str = None, temperature: float = 0.2):
    """Make a chat request to Groq"""
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    client = get_client()
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=800,
    )
    
    return response.choices[0].message.content


def chat_json(prompt: str, system: str = None, temperature: float = 0.1):
    """Make a chat request expecting JSON response"""
    
    raw = chat(prompt, system=system, temperature=temperature)
    
    # Clean up any ```json fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    
    # Parse JSON
    return json.loads(raw)
```

### How Agents Use the LLM Client

```python
# Example: How the Classifier agent uses it

def classify(complaint: dict) -> dict:
    """Classify a complaint using LLM"""
    
    prompt = f"""Classify the complaint below.
    
Category: UPI, ATM, Card, Loan, NetBanking, General
Severity: Critical, High, Medium, Low
Sentiment: Angry, Frustrated, Neutral, Polite

Return JSON: {{"category": "...", "severity": "...", "sentiment": "...", "rationale": "..."}}

Complaint: {complaint['complaint_text']}
Amount: {complaint.get('amount_involved')}
"""
    
    # Call Groq through the client
    result = chat_json(prompt, system="You are a complaint classifier.")
    
    return result
```

---

## 16. The Database: SQLite Persistence

### What is SQLite?

SQLite is a **lightweight file-based database**. Think of it as a spreadsheet stored in a file - simple, fast, and no server required.

```
Traditional Database (MySQL/PostgreSQL):
┌─────────────────────────────────────┐
│  Database Server                    │
│  ┌─────────────────────────────┐   │
│  │ complaintiq database        │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
        ↑ Needs separate server

SQLite:
┌─────────────────────────────────────┐
│  complaintiq.sqlite (file)          │
│  ┌─────────────────────────────┐   │
│  │ complaints table            │   │
│  │ root_cause_alerts table     │   │
│  │ feedback table             │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
        ↑ Just a file!
```

### The Database Schema

```python
# database/db.py

SCHEMA = """
CREATE TABLE complaints (
    id              TEXT PRIMARY KEY,
    customer_name   TEXT NOT NULL,
    channel         TEXT NOT NULL,
    complaint_text  TEXT NOT NULL,
    language        TEXT,
    date            TEXT NOT NULL,
    location        TEXT,
    account_type    TEXT,
    amount_involved REAL,
    
    -- Enriched by agents
    category        TEXT,
    severity        TEXT,
    sentiment       TEXT,
    intake_json     TEXT,
    duplicate_of    TEXT,
    similarity      REAL,
    draft_response  TEXT,
    sla_due_date    TEXT,
    sla_breach_prob REAL,
    risk_score      INTEGER,
    risk_ombudsman  INTEGER,
    risk_churn      INTEGER,
    risk_social     INTEGER,
    cluster_id      INTEGER,
    status          TEXT DEFAULT 'open',
    processed_at    TEXT,
    
    -- ML Second Opinions
    ml_category          TEXT,
    ml_category_prob     REAL,
    category_confidence TEXT,
    ml_sentiment         TEXT,
    ml_sentiment_prob    REAL,
    sentiment_confidence TEXT,
    priority_score       INTEGER,
    resolved_at          TEXT
);

CREATE TABLE root_cause_alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id  INTEGER,
    category    TEXT,
    location    TEXT,
    count       INTEGER,
    summary     TEXT,
    created_at  TEXT
);

CREATE TABLE feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_id    TEXT NOT NULL,
    field           TEXT NOT NULL,
    original_value  TEXT,
    corrected_value TEXT,
    is_correct      INTEGER NOT NULL,
    created_at      TEXT NOT NULL
);
"""
```

### Database Operations

```python
# --- Reading Data ---

def get_complaint(complaint_id: str):
    """Get ONE complaint by ID"""
    with connect() as c:
        row = c.execute("SELECT * FROM complaints WHERE id = ?", (complaint_id,)).fetchone()
    return dict(row) if row else None


def list_complaints(limit=None, where=None, params=()):
    """List ALL complaints, optionally filtered"""
    q = "SELECT * FROM complaints"
    if where:
        q += " WHERE " + where
    q += " ORDER BY date DESC"
    if limit:
        q += f" LIMIT {limit}"
    
    with connect() as c:
        rows = c.execute(q, tuple(params)).fetchall()
    return [dict(r) for r in rows]


def list_unprocessed(limit=None):
    """Get complaints that haven't been through the pipeline"""
    return list_complaints(limit=limit, where="processed_at IS NULL")


def customer_history(customer_name: str):
    """Get all complaints from ONE customer"""
    return list_complaints(where="customer_name = ?", params=(customer_name,))


# --- Writing Data ---

def update_complaint(complaint_id: str, **fields):
    """Update a complaint with new values"""
    
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [complaint_id]
    
    with connect() as c:
        c.execute(f"UPDATE complaints SET {set_clause} WHERE id = ?", vals)


def record_feedback(complaint_id, field, original, corrected, is_correct):
    """Record human feedback on classification"""
    
    with connect() as c:
        c.execute(
            "INSERT INTO feedback (complaint_id, field, original_value, "
            "corrected_value, is_correct, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (complaint_id, field, original, corrected, 1 if is_correct else 0, 
             datetime.utcnow().isoformat())
        )
    
    # If corrected, also update the complaint
    if not is_correct and corrected:
        update_complaint(complaint_id, **{field: corrected})
```

### Context Manager for Connections

```python
# database/db.py

@contextmanager
def connect():
    """Connect to SQLite, auto-commit on success"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Access columns by name
    try:
        yield conn
        conn.commit()  # Save changes
    finally:
        conn.close()   # Always close

# Usage:
with connect() as c:
    c.execute("SELECT * FROM complaints WHERE id = ?", ("UBI-0001",))
    # Changes automatically committed when exiting the with block
```

---

## 17. The Dashboard: Streamlit Frontend

### What is Streamlit?

Streamlit is a **Python web app framework** for data science. It lets you build dashboards entirely in Python - no HTML/CSS/JavaScript required.

```
Traditional Web App:
Python + HTML + CSS + JavaScript + Database
         ↓
   Very Complex

Streamlit App:
Python (all logic) → Streamlit → Web Dashboard
         ↓
   Simple & Fast
```

### Dashboard Architecture

```python
# dashboard/app.py

def main():
    """Main dashboard function"""
    
    # === PAGE SETUP ===
    st.set_page_config(
        page_title="ComplaintIQ -- Union Bank of India",
        page_icon="UBI",
        layout="wide",
    )
    
    # === LOAD DATA ===
    df = load_complaints()   # Get all complaints from database
    alerts = load_alerts()  # Get root cause alerts
    
    # === SIDEBAR ===
    render_sidebar(df)  # RBI report download button
    
    # === KPI ROW ===
    render_kpis(df)  # Total, processed, at-risk, auto-resolved
    
    # === ALERT BANNERS ===
    render_alert_banners(df, alerts)  # Critical alerts at top
    
    # === LIVE SUBMIT ===
    render_live_submit()  # Form to submit new complaints
    
    # === TABS ===
    (tab_feed, tab_customer, tab_map, tab_sla, tab_alerts,
     tab_draft, tab_models, tab_analytics, tab_fb) = st.tabs([
        "Live feed", "Customer", "India map", "SLA tracker",
        "Root cause", "Drafted replies", "Model performance", 
        "Analytics", "Feedback",
    ])
    
    with tab_feed:
        render_live_feed(df)
    with tab_customer:
        render_customer_view(df)
    with tab_map:
        render_india_map(df)
    with tab_sla:
        render_sla_tracker(df)
    with tab_alerts:
        render_alerts(alerts)
    # ... and so on
```

### Dashboard Pages Explained

#### Page 1: Live Feed (Main Complaints Table)

```python
def render_live_feed(df):
    """Display all complaints in a filterable table"""
    
    # Filter controls
    sev = st.multiselect("Severity", sorted(df["severity"].unique()))
    cat = st.multiselect("Category", sorted(df["category"].unique()))
    chan = st.multiselect("Channel", sorted(df["channel"].unique()))
    only_open = st.checkbox("Only at-risk")
    
    # Apply filters
    view = df.copy()
    if sev: view = view[view["severity"].isin(sev)]
    if cat: view = view[view["category"].isin(cat)]
    if chan: view = view[view["channel"].isin(chan)]
    if only_open: view = view[view["sla_breach_prob"] >= 0.5]
    
    # Sort by priority (highest first)
    view = view.sort_values("priority_score", ascending=False)
    
    # Show (*) marker for ML/LLM disagreement
    view["Category"] = [
        f"{cat} (*)" if conf == "Needs Review" else cat
        for cat, conf in zip(view["category"], view["category_confidence"])
    ]
    
    # Display table
    st.dataframe(view, use_container_width=True)
```

#### Page 2: Customer Drill-Down

```python
def render_customer_view(df):
    """Show all complaints from ONE customer"""
    
    # Customer selector (default: customer with most complaints)
    names = sorted(df["customer_name"].unique())
    default = df.groupby("customer_name").size().sort_values(ascending=False).index[0]
    pick = st.selectbox("Customer", names, index=names.index(default))
    
    # Get customer's complaints
    hist = df[df["customer_name"] == pick].sort_values("date")
    
    # Left: Emotion timeline chart
    with left:
        fig = px.bar(
            hist, x="date", y="sentiment_score",
            color="severity",
            title=f"Emotion over time -- {pick}"
        )
        st.plotly_chart(fig)
    
    # Right: Risk score display
    with right:
        latest = hist.dropna(subset=["risk_score"]).tail(1)
        risk = latest["risk_score"].iloc[0]
        
        # Color-coded risk display
        if risk >= 70: color = "red"
        elif risk >= 40: color = "orange"
        else: color = "green"
        
        st.markdown(f"""
        <div style='background:{color}; padding:20px; border-radius:10px'>
            <div style='font-size:48px; color:white'>{risk}</div>
            <div style='color:white'>out of 100</div>
        </div>
        """, unsafe_allow_html=True)
```

#### Page 3: India Map (Geographic Distribution)

```python
def render_india_map(df):
    """Show complaint density on India map"""
    
    # Group by city
    by_city = df.groupby("location").size().reset_index(name="count")
    
    # Add lat/lon coordinates
    by_city["lat"] = by_city["location"].map(lambda c: CITY_COORDS[c][0])
    by_city["lon"] = by_city["location"].map(lambda c: CITY_COORDS[c][1])
    
    # Size markers by count
    sizes = by_city["count"] * 0.75 + 7
    
    # Plot on map
    fig = px.scatter_geo(
        by_city, lat="lat", lon="lon", size="count",
        hover_name="location",
        title="Complaint hotspots across India"
    )
    st.plotly_chart(fig)
```

#### Page 4: SLA Tracker

```python
def render_sla_tracker(df):
    """Show complaints approaching deadline"""
    
    # Calculate days until due
    today = pd.Timestamp(date.today())
    sla = df.dropna(subset=["sla_due_date"]).copy()
    sla["days_until_due"] = (sla["sla_due_date"] - today).dt.days
    
    # Bucket into time ranges
    def _bucket(d):
        if d < 0:    return "Overdue"
        if d <= 1:   return "<1 day"
        if d <= 3:   return "<3 days"
        if d <= 7:   return "<7 days"
        return ">7 days"
    
    sla["bucket"] = sla["days_until_due"].apply(_bucket)
    
    # Chart
    fig = px.bar(sla.groupby("bucket").size(), x="bucket", y="count",
                 title="Open complaints by SLA bucket")
    st.plotly_chart(fig)
```

#### Page 5: Root Cause Alerts

```python
def render_alerts(alerts):
    """Show systemic issue clusters"""
    
    for _, a in alerts.iterrows():
        cnt = int(a["count"])
        
        # Color by volume
        if cnt >= 40: bg = "red"
        elif cnt >= 20: bg = "orange"
        else: bg = "yellow"
        
        st.markdown(f"""
        <div style='background:{bg}; padding:15px; border-radius:10px; margin:10px 0'>
            <b>Cluster #{a['cluster_id']}</b> - {cnt} complaints
            
            <div style='margin-top:8px'>
                <span style='background:blue; color:white; padding:3px 8px; border-radius:4px'>
                    {a['category']}
                </span>
            </div>
            
            <div style='margin-top:8px; font-size:14px'>
                {a['summary']}
            </div>
        </div>
        """, unsafe_allow_html=True)
```

### Live Complaint Submission

```python
def render_live_submit():
    """Form to submit and process new complaint"""
    
    with st.form("live_submit"):
        text = st.text_area("Complaint text", placeholder="UPI payment failed...")
        name = st.text_input("Customer name")
        channel = st.selectbox("Channel", ["email", "whatsapp", "twitter", ...])
        # ... more fields
        submitted = st.form_submit_button("Run pipeline", type="primary")
    
    if submitted:
        # Run the pipeline
        result = process_one_streaming(new_id, on_step=on_step)
        
        # Show results
        st.metric("Category", result["classification"]["category"])
        st.metric("Risk score", result["risk"]["overall"])
        
        if result["draft_response"]:
            st.info(result["draft_response"])
```

### Running the Dashboard

```bash
# Start the dashboard
streamlit run dashboard/app.py

# Opens at http://localhost:8501
```

---

## 18. The RBI Report Generator

### What is RBI Compliance?

The Reserve Bank of India (RBI) requires banks to submit **compliance reports** showing:
- How many complaints received
- How many resolved within SLA
- How many breached
- Details of each complaint

This is a **legal requirement** - banks can be penalized for incorrect reports.

### Report Generation

```python
# dashboard/rbi_report.py

REPORT_COLUMNS = [
    "Complaint ID", "Customer Name", "Type", "Severity", "Channel",
    "Date Filed", "SLA Deadline", "Days Remaining",
    "Breach Status", "Resolution Status",
]


def build_report(rows, today=None):
    """Convert database rows to RBI-compliant format"""
    
    today_ts = pd.Timestamp(today or date.today())
    df = pd.DataFrame(rows)
    
    # Calculate breach status
    def _breach_status(due_date, breach_prob):
        if due_date < today_ts:
            return "Breached"
        if breach_prob and breach_prob >= 0.5:
            return "At Risk"
        return "On Track"
    
    # Calculate resolution status
    def _resolution_status(row):
        status = row.get("status", "open")
        if status == "resolved": return "Resolved"
        if status == "auto_resolved_dup": return "Auto-Resolved (Duplicate)"
        if status == "auto_resolved_std": return "Auto-Resolved (Standard)"
        
        due = row.get("sla_due_date")
        if due and due < today_ts:
            return "Breached"
        return "Pending"
    
    return pd.DataFrame({
        "Complaint ID": df["id"],
        "Customer Name": df["customer_name"],
        "Type": df["category"].fillna("General"),
        "Severity": df["severity"].fillna("Medium"),
        "Channel": df["channel"],
        "Date Filed": df["date"],
        "SLA Deadline": df["sla_due_date"],
        "Days Remaining": (df["sla_due_date"] - today_ts).dt.days,
        "Breach Status": [_breach_status(d, p) for d, p in zip(df["sla_due_date"], df["sla_breach_prob"])],
        "Resolution Status": [_resolution_status(r) for _, r in df.iterrows()],
    })[REPORT_COLUMNS]


def to_csv(rows, today=None):
    """Generate downloadable CSV"""
    df = build_report(rows, today)
    buf = StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8-sig")  # BOM for Excel Hindi support
```

### Download from Dashboard

```python
# Sidebar download button
csv_bytes = rbi_report.to_csv(df)
st.sidebar.download_button(
    label="Download RBI Compliance Report (CSV)",
    data=csv_bytes,
    file_name=f"rbi_compliance_{date.today()}.csv",
    mime="text/csv",
)
```

---

## 19. Pipeline Utilities: Data Remix & ML Backfill

### Data Remix: Making Data Realistic

```python
# pipeline/data_remix.py

def remix():
    """One-shot data enhancement"""
    
    # 1. Spread dates over Jan-May 2026
    # 2. Flip 50 rows to mobile_app channel
    # 3. Re-run SLA prediction (dates changed)
    # 4. Stamp resolution statuses
    
    # Result: More realistic test data
    return {
        "rows": 1000,
        "mobile_app": 50,
        "auto_resolved_dup": 86,
        "auto_resolved_std": 100,
        "resolved": 200,
        "open": 614,
    }
```

### ML Backfill: Adding Second Opinions

```python
# pipeline/ml_backfill.py

def backfill():
    """Add ML predictions to existing processed rows"""
    
    rows = db.list_complaints(where="processed_at IS NOT NULL")
    
    for r in rows:
        # ML Category
        ml_cat = ml_category.predict(r["complaint_text"])
        cat_conf = ml_category.agreement(r["category"], ml_cat["category"])
        
        # ML Sentiment  
        ml_sent = sentiment_ml.predict(r["complaint_text"])
        sent_conf = sentiment_ml.agreement(r["sentiment"], ml_sent)
        
        db.update_complaint(
            r["id"],
            ml_category=ml_cat["category"],
            ml_category_prob=ml_cat["probability"],
            category_confidence=cat_conf,
            ml_sentiment=ml_sent["bucket"],
            ml_sentiment_prob=ml_sent["score"],
            sentiment_confidence=sent_conf,
        )
```

---

## 20. ML Model Training Scripts

### Training the SLA Model

```python
# models/train_sla_model.py

def main():
    """Train SLA breach predictor"""
    
    # 1. Load data
    df = load_training_data()  # From database or JSON
    
    # 2. Feature engineering
    df = engineer_features(df)
    # - Text length, word count
    # - Hours since filed
    # - Repeat customer flag
    # - Severity/sentiment scores
    # - SLA days remaining
    # - % of SLA elapsed
    
    # 3. Create realistic labels
    y = realistic_labels(df)
    # breach = 1 if:
    #   - status=open AND days_since_filed > sla_days
    #   - status=open AND severity=Critical AND 80% of SLA used
    
    # 4. Train multiple models
    models = {
        "RandomForest": RandomForestClassifier(class_weight="balanced"),
        "XGBoost": XGBClassifier(objective="binary:logistic"),
        "LightGBM": LGBMClassifier(),
    }
    
    # 5. Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    
    for name, model in models.items():
        aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        print(f"{name}: AUC = {aucs.mean():.4f} +/- {aucs.std():.4f}")
    
    # 6. Pick winner and save
    winner = best_model  # XGBoost(tuned)
    joblib.dump(winner, "models/sla_best_model.joblib")
```

### Training the Category Model

```python
# models/train_category_classifier.py

def main():
    """Train TF-IDF + Logistic Regression category classifier"""
    
    # 1. TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["complaint_text"])
    
    # 2. Train Logistic Regression
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, df["category"])
    
    # 3. Evaluate
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    print(f"Accuracy: {accuracy:.3f}")  # ~97%
    
    # 4. Save
    joblib.dump({
        "vectorizer": vectorizer,
        "classifier": clf,
        "accuracy": accuracy,
    }, "models/category_clf.joblib")
```

---

## 21. Complete System Flow: End-to-End

### How Everything Connects

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPLAINTIQ SYSTEM ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐                                                       │
│  │  Raw Complaint   │                                                       │
│  │  (JSON/Form)     │                                                       │
│  └────────┬─────────┘                                                       │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  API Endpoint: POST /complaint                                       │   │
│  │  or Dashboard: Submit new complaint                                 │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Ingest into Database (SQLite)                                      │   │
│  │  complaintiq.sqlite                                                 │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ORCHESTRATOR (pipeline/orchestrator.py)                            │   │
│  │  ┌─────────────────────────────────────────────────────────────┐  │   │
│  │  │ Agent 1: INTAKE - Extract structured fields                  │  │   │
│  │  │ Agent 2: CLASSIFIER - Category/Severity/Sentiment           │  │   │
│  │  │ Agent 3: DUPLICATE - Semantic search (ChromaDB)              │  │   │
│  │  │ Agent 4: RESPONSE DRAFTER - Generate reply (if not duplicate)│ │   │
│  │  │ Agent 5: SLA MONITOR - Calculate deadline & breach prob     │  │   │
│  │  │ Agent 6: ROOT CAUSE - KMeans clustering (batch)             │  │   │
│  │  └─────────────────────────────────────────────────────────────┘  │   │
│  │  ┌─────────────────────────────────────────────────────────────┐  │   │
│  │  │ ML Second Opinions                                           │  │   │
│  │  │ • Category: TF-IDF + LogReg                                   │  │   │
│  │  │ • Sentiment: RoBERTa                                          │  │   │
│  │  │ • Priority: Gradient Boosting                                │  │   │
│  │  └─────────────────────────────────────────────────────────────┘  │   │
│  │  ┌─────────────────────────────────────────────────────────────┐  │   │
│  │  │ Risk Score Calculation                                      │  │   │
│  │  │ • Ombudsman Risk (45%)                                      │  │   │
│  │  │ • Churn Risk (30%)                                          │  │   │
│  │  │ • Social Media Risk (25%)                                   │  │   │
│  │  └─────────────────────────────────────────────────────────────┘  │   │
│  │  ┌─────────────────────────────────────────────────────────────┐  │   │
│  │  │ Auto-Resolution Logic                                       │  │   │
│  │  │ • Duplicate → auto_resolved_dup                              │  │   │
│  │  │ • Low+Polite → auto_resolved_std                            │  │   │
│  │  │ • Else → open                                                │  │   │
│  │  └─────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Update Database with enriched data                                 │   │
│  │  • category, severity, sentiment                                    │   │
│  │  • draft_response, sla_due_date, sla_breach_prob                   │   │
│  │  • risk_score (overall, ombudsman, churn, social)                 │   │
│  │  • ml_category, ml_sentiment, priority_score                       │   │
│  │  • status, processed_at                                            │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Update Vector Database (ChromaDB)                                  │   │
│  │  Index complaint embedding for future duplicate detection          │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    │
│  │  Dashboard       │    │  API Endpoint    │    │  RBI Report      │    │
│  │  (Streamlit)     │    │  (FastAPI)       │    │  (CSV Download)  │    │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Starting the Full System

```bash
# Terminal 1: Start the API
uvicorn api.main:app --reload --port 8000

# Terminal 2: Start the Dashboard
streamlit run dashboard/app.py

# Browser: Open http://localhost:8501
```

---

## 22. Why Each Technology Choice

### FastAPI over Flask/Django

| Factor | FastAPI | Flask | Django |
|--------|---------|-------|--------|
| Auto docs | ✅ `/docs` | ❌ | ✅ |
| Async support | ✅ | ⚠️ | ⚠️ |
| Type validation | ✅ Pydantic | ❌ | ✅ ORM |
| Learning curve | Low | Low | High |

**Decision: FastAPI** - Built-in docs, automatic type validation, perfect for ML data structures.

### Streamlit over Dash/React

| Factor | Streamlit | Dash | React+Flask |
|--------|-----------|------|------------|
| Development speed | ⚡ Fast | 😐 Medium | ❌ Slow |
| Python-only | ✅ | ⚠️ | ❌ |
| ML integration | ✅ Native | ✅ | ❌ |

**Decision: Streamlit** - Fastest development, native pandas/plotly support.

### SQLite over PostgreSQL/MySQL

| Factor | SQLite | PostgreSQL | MySQL |
|--------|--------|------------|-------|
| Setup | File only | Server | Server |
| Concurrent writes | Limited | ✅ | ✅ |
| For data science | ✅ | ✅ | ✅ |

**Decision: SQLite** - Zero setup, file-based backup, sufficient for 1000-10000 rows.

### ChromaDB over Pinecone/Weaviate

| Factor | ChromaDB | Pinecone | Weaviate |
|--------|----------|---------|----------|
| Local/Persistent | ✅ | ❌ (cloud) | ✅ |
| Cost | Free | Paid | Free |
| Metadata filtering | ✅ Easy | ✅ | ✅ |

**Decision: ChromaDB** - Free, local, Python-native, perfect for this scale.

---

## 23. Summary: The Complete Picture

### What We've Covered

1. **The Problem**: Manual complaint processing is slow, error-prone, and unscalable
2. **The Solution**: AI-powered pipeline with 6 specialized agents
3. **Each Agent**: How Intake, Classifier, Duplicate Detector, Response Drafter, SLA Monitor, and Root Cause Detector work
4. **ML Models**: TF-IDF+LogReg, RoBERTa, XGBoost, Gradient Boosting
5. **Vector Database**: ChromaDB for semantic search
6. **Risk Scoring**: Three sub-scores (ombudsman, churn, social) combined into overall risk
7. **Database**: SQLite schema and operations
8. **Orchestrator**: Pipeline coordinator connecting all agents
9. **API**: FastAPI backend endpoints
10. **Dashboard**: Streamlit frontend with 9 tabs
11. **RBI Reporting**: Compliance report generation

### Key Metrics

```
Processing Speed:    15 min/complaint → 5 sec/complaint (180x faster)
Auto-resolution:     18.6% handled automatically
Category Accuracy: 97%
SLA Prediction AUC: 94%
Root Cause Alerts:  6 systemic issues detected
```

### Files in the Project

```
ComplaintIQ/
├── agents/
│   ├── classifier.py          # Category/Severity/Sentiment
│   ├── duplicate_detector.py  # Semantic duplicate search
│   ├── intake.py              # Information extraction
│   ├── llm_client.py          # Groq integration
│   ├── ml_category.py        # TF-IDF category classifier
│   ├── priority.py           # Gradient Boosting priority
│   ├── response_drafter.py    # Generate bank replies
│   ├── risk_score.py         # Risk calculation
│   ├── root_cause.py         # KMeans clustering
│   ├── sentiment_ml.py       # RoBERTa sentiment
│   └── sla_monitor.py        # SLA breach prediction
├── api/
│   └── main.py               # FastAPI endpoints
├── dashboard/
│   ├── app.py                # Streamlit dashboard
│   └── rbi_report.py         # RBI compliance report
├── database/
│   └── db.py                 # SQLite persistence
├── models/
│   ├── train_category_classifier.py
│   ├── train_priority_model.py
│   └── train_sla_model.py
├── pipeline/
│   ├── orchestrator.py       # Main pipeline
│   ├── data_remix.py         # Data enhancement
│   └── ml_backfill.py        # ML predictions
├── data/
│   ├── complaints.json       # Seed data (1000 complaints)
│   ├── sla_rules.json        # SLA rules
│   ├── chroma_db/            # Vector database
│   └── complaintiq.sqlite    # SQLite database
└── study_guide.md            # This guide!
```

### What's NOT Covered (Future Work)

- **Authentication**: Add user login for production
- **Real-time streaming**: WebSocket for live updates
- **Multi-tenant**: Support multiple banks
- **Email/WhatsApp integration**: Actually receive complaints from channels
- **Deployment**: Docker, Kubernetes, cloud deployment
- **Monitoring**: Logging, alerting, metrics
- **A/B Testing**: Compare agent versions

---

Would you like me to dive even deeper into any specific component?