# Output Evaluator - Quality Check System

## Overview

The `OutputEvaluator` class implements 4 distinct quality checks that analyze LLM-generated responses to detect potential issues. Each evaluator serves a specific purpose in ensuring response quality and reliability.

---

## The 4 Evaluators

### 1. No-Context Evaluator (`no_context`)

**Purpose:** Detects when the LLM generates an answer without any supporting documentation.

**How it works:**
- Triggers when `chunks_retrieved == 0` AND the response is NOT a refusal
- This catches potential hallucinations where the LLM invents information

**Example scenario:**
```
Question: "What is ClearPath's API rate limit?"
Chunks Retrieved: 0
Response: "ClearPath allows 1000 API calls per hour."
Flag: ✓ no_context (LLM made up an answer without documentation)
```

**Why it matters:** If no relevant documentation was found, the LLM should refuse to answer rather than fabricate information.

---

### 2. Refusal Evaluator (`refusal`)

**Purpose:** Detects when the LLM explicitly refuses to answer the question entirely.

**How it works:**
- Looks for refusal phrases like "I don't have", "not mentioned", "cannot find", etc.
- Distinguishes between complete refusals and partial answers
- Uses contrast indicators ("but", "however") to identify partial answers
- Only flags complete refusals where no actual information is provided

**Example scenarios:**

**Complete Refusal (Flagged):**
```
Response: "I don't have information about that feature in the documentation."
Flag: ✓ refusal
```

**Partial Answer (NOT Flagged):**
```
Response: "The documentation doesn't mention X, but it does mention that ClearPath supports Y and Z."
Flag: ✗ (provides useful information despite initial disclaimer)
```

**Why it matters:** Helps identify when the system couldn't answer the user's question, which may indicate gaps in documentation or retrieval issues.

---

### 3. Unverified Features Evaluator (`unverified_feature`)

**Purpose:** Detects when the LLM mentions specific features, integrations, or proper nouns that don't appear in the retrieved documentation chunks.

**How it works:**
- Extracts proper nouns from both the response and retrieved chunks
- Compares them to find mentions in the response that aren't in the source material
- Uses case-insensitive matching with word boundaries
- Filters out common stop words and short terms

**Proper noun extraction includes:**
- Capitalized words mid-sentence (e.g., "Slack", "GitHub")
- Integration/tool names (e.g., "OAuth", "SAML", "API")
- Product names and features

**Example scenario:**
```
Question: "What integrations does ClearPath support?"
Retrieved Chunks: Mention "Slack" and "Google Calendar"
Response: "ClearPath integrates with Slack, Google Calendar, and Jira."
Flag: ✓ unverified_feature (Jira wasn't in the documentation)
```

**Why it matters:** Catches hallucinated features where the LLM uses general SaaS knowledge to invent capabilities not documented in the actual product docs.

---

### 4. Pricing Uncertainty Evaluator (`pricing_uncertainty`)

**Purpose:** Detects pricing-related responses that express uncertainty, mention conflicts, or refuse to answer.

**How it works:**
- First checks if the response is pricing-related (mentions "price", "cost", "plan", etc.)
- Then checks for THREE conditions:
  1. **Hedging language:** "may", "might", "approximately", "around", "varies"
  2. **Conflict indicators:** "contradict", "inconsistent", "different prices", "unclear"
  3. **Refusal:** Complete refusal to answer a pricing question

**Example scenarios:**

**Hedging (Flagged):**
```
Response: "The Enterprise plan costs approximately $500 per month."
Flag: ✓ pricing_uncertainty (uses "approximately")
```

**Conflict (Flagged):**
```
Response: "The documentation shows different prices in different sections."
Flag: ✓ pricing_uncertainty (mentions inconsistency)
```

**Refusal (Flagged):**
```
Response: "I don't have information about pricing in the documentation."
Flag: ✓ pricing_uncertainty (refusal on pricing question)
```

**Confident Answer (NOT Flagged):**
```
Response: "The Enterprise plan costs $500 per month according to the pricing sheet."
Flag: ✗ (definitive answer with source)
```

**Why it matters:** Pricing information is critical and must be accurate. This evaluator ensures the system doesn't provide uncertain or conflicting pricing data.

---

## Evaluation Flow

```python
def evaluate(response, chunks_retrieved, sources) -> List[str]:
    flags = []
    
    # Check 1: No-context detection
    if _is_no_context(response, chunks_retrieved):
        flags.append("no_context")
    
    # Check 2: Refusal detection
    if _is_refusal(response):
        flags.append("refusal")
    
    # Check 3: Groundedness check
    if _has_unverified_features(response, sources):
        flags.append("unverified_feature")
    
    # Check 4: Pricing uncertainty
    if _has_pricing_uncertainty(response, sources):
        flags.append("pricing_uncertainty")
    
    return flags
```

---

## Key Differences Summary

| Evaluator | What It Detects | Primary Concern | Trigger Condition |
|-----------|----------------|-----------------|-------------------|
| **no_context** | Answering without documentation | Hallucination | No chunks + non-refusal response |
| **refusal** | Complete inability to answer | Coverage gaps | Refusal phrases without actual info |
| **unverified_feature** | Mentioning undocumented features | Feature hallucination | Proper nouns not in sources |
| **pricing_uncertainty** | Uncertain/conflicting pricing info | Pricing accuracy | Hedging, conflicts, or refusal on pricing |

---

## Usage Example

```python
evaluator = OutputEvaluator()

flags = evaluator.evaluate(
    response="ClearPath integrates with Slack and costs around $500/month.",
    chunks_retrieved=3,
    sources=[chunk1, chunk2, chunk3]
)

# Possible flags: ["unverified_feature", "pricing_uncertainty"]
# - unverified_feature: if "Slack" not in chunks
# - pricing_uncertainty: due to "around" (hedging language)
```

---

## Design Philosophy

Each evaluator targets a specific failure mode:
1. **No-context:** Prevents fabrication when no data exists
2. **Refusal:** Identifies when the system can't help the user
3. **Unverified features:** Catches specific hallucinated details
4. **Pricing uncertainty:** Protects critical business information accuracy

Together, they provide comprehensive quality assurance for RAG system outputs.
