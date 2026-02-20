# Spec Refinements - High-ROI Improvements

This document summarizes the key refinements made to the ClearPath RAG Chatbot spec based on expert feedback.

## 1. Contextual Heading Injection (Task 3.1)

**Problem**: Simple font-size detection loses context across pages. If "Pricing" is H1 on page 2, chunks on page 10 won't have that context.

**Solution**: Hierarchical header stack state machine
- Maintains `current_header_stack` that persists across pages
- Tracks H1/H2/H3 hierarchy using font size thresholds
- Updates stack only when new header of equal/higher hierarchy found
- Format: `[Context: Pricing > Enterprise Plan] {chunk_text}`

**Impact**: Dramatically reduces retrieval failures by providing persistent topical context across entire document sections.

---

## 2. Query Complexity Scoring (Task 10.1)

**Problem**: Binary classification logs don't provide data-backed evidence for Q1 written answer.

**Solution**: Add `complexity_score` object to routing logs
```json
"complexity_score": {
  "word_count": 6,
  "complex_keyword_count": 0,
  "question_mark_count": 1,
  "comparison_word_count": 0
}
```

**Impact**: Makes Q1 analysis much easier with concrete metrics showing why queries were classified as simple/complex.

---

## 3. Groundedness Check - Proper Noun Verification (Task 9.1)

**Problem**: Llama models are "too helpful" and hallucinate features based on general SaaS knowledge.

**Solution**: Extract and compare proper nouns
- Extract proper nouns (capitalized terms, integration names) from LLM response
- Extract proper nouns from retrieved chunks
- Flag `unverified_feature` if LLM mentions proper noun not in chunks
- Example: LLM says "Slack integration" but no chunks contain "Slack" → flag it

**Impact**: Catches the most dangerous hallucination type - when LLM invents features ClearPath doesn't have.

---

## 4. Accurate Token Counting (Tasks 3.1, 13.1, 13.2)

**Problem**: Using all-mpnet-base-v2 tokenizer to count Llama 3 tokens will underestimate costs (different vocabularies).

**Solution**: Separate tokenizers for different purposes
- Use `AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")` for embedding token counts (chunking)
- Use `tiktoken` with `o200k_base` encoding OR `transformers.AutoTokenizer` for Llama 3 prompt token counting
- Document this distinction clearly in tasks

**Impact**: Accurate cost estimation for Q3 written answer.

---

## 5. Aggressive HF API Retry Strategy (Task 4.1)

**Problem**: HF Inference API free tier models "sleep" and take 15-20s to load on first query (503 errors).

**Solution**: Aggressive exponential backoff
- 5 retries with delays: 5s, 10s, 20s, 40s, 60s
- Log model loading delays for monitoring
- Consider warming up model at startup with dummy query

**Impact**: Prevents first-query failures and provides better user experience despite free tier limitations.

---

## 6. API Contract Compliance

**Verified**: All response fields match API_CONTRACT.md specification
- `tokens` object has `input` and `output` (not system_prompt_tokens/context_tokens)
- `sources` array can be empty `[]` when no chunks retrieved
- `evaluator_flags` array can be empty `[]` when no issues detected
- `conversation_id` always present (generated if not provided)

---

## 7. Written Answers Guidance (Task 18.2)

**Enhanced Q1**: Use `rule_triggered` and `complexity_score` logs to find real misclassification with data

**Enhanced Q2**: Discuss groundedness check and contextual heading injection as improvements

**Enhanced Q3**: Use tiktoken for accurate Llama 3 token estimation (not all-mpnet-base-v2)

**Enhanced Q4**: Add HF API cold start delays (15-20s) as a genuine limitation to discuss

---

## Summary

These refinements transform the spec from "good" to "production-ready" by:
1. Preventing context loss across pages (hierarchical headers)
2. Providing data-backed evidence for written answers (complexity scoring)
3. Catching dangerous hallucinations (groundedness check)
4. Ensuring accurate cost estimation (proper tokenizers)
5. Handling free tier limitations gracefully (aggressive retries)

All changes are backward-compatible and enhance the original design without breaking existing requirements.
