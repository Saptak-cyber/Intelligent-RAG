# Evaluation Test Harness

This evaluation harness tests the ClearPath RAG Chatbot system end-to-end, measuring routing accuracy, retrieval quality, latency, token usage, and evaluator flag frequency.

## Overview

The evaluation script (`evaluate_system.py`) executes a comprehensive test suite based on the enhanced test questions from `test_questions_ENHANCED_routing_and_evaluator.md`. It includes:

- **115+ test queries** covering diverse scenarios
- **Model Router tests**: Simple vs complex classification, OOD detection, edge cases
- **Output Evaluator tests**: No-context detection, refusal detection, unverified features, pricing uncertainty
- **Performance metrics**: Latency distribution (p50, p95, p99), token usage, retrieval quality
- **Category breakdown**: Performance analysis by query type

## Prerequisites

1. **Backend API running**: The evaluation script requires the FastAPI backend to be running
2. **Database populated**: Ensure documents have been ingested and vector store is populated
3. **Environment variables**: All required API keys must be configured

## Usage

### Basic Usage

Run the evaluation with default settings (localhost:8000):

```bash
cd backend
python evaluate_system.py
```

### Custom API URL

Test against a different API endpoint:

```bash
python evaluate_system.py --api-url http://localhost:8001
```

### Custom Output Path

Save the report to a different location:

```bash
python evaluate_system.py --output results/my_evaluation.txt
```

### Adjust Request Delay

Change the delay between requests (useful for rate limiting):

```bash
python evaluate_system.py --delay 200  # 200ms delay
```

### Full Example

```bash
python evaluate_system.py \
  --api-url http://localhost:8000 \
  --output logs/evaluation_$(date +%Y%m%d_%H%M%S).txt \
  --delay 100
```

## Test Query Categories

The evaluation includes the following test categories:

### Model Router Tests

1. **Simple Questions (15 queries)**: Basic factual lookups that should use the 8B model
2. **Complex - Keywords (15 queries)**: Questions with complex keywords (explain, compare, analyze)
3. **Complex - Length (5 queries)**: Long questions (>15 words) requiring the 70B model
4. **Complex - Multiple Questions (10 queries)**: Queries with multiple question marks
5. **Complex - Comparison (10 queries)**: Questions with comparison words (vs, better, worse)
6. **OOD - Should Skip Retrieval (15 queries)**: Greetings and meta-questions
7. **Edge - Not OOD (8 queries)**: Questions containing OOD words but are legitimate queries
8. **Edge - CSV/VS (7 queries)**: Tests for the CSV/VS bug (csv contains "vs")

### Output Evaluator Tests

9. **Evaluator - No Context (10 queries)**: Questions completely outside document scope
10. **Evaluator - Refusal (10 queries)**: Questions with no answer in docs (should refuse)
11. **Specific Details (10 queries)**: Questions testing specific numbers, dates, names from docs

## Metrics Reported

### Routing Performance
- **Classification Accuracy**: Percentage of queries correctly classified as simple/complex
- **Skip Retrieval Accuracy**: Percentage of OOD queries that correctly skipped retrieval

### Model Usage
- Distribution of queries across models (llama-3.1-8b-instant vs llama-3.1-70b-versatile)

### Latency Distribution
- **Mean**: Average latency across all queries
- **P50**: Median latency (50th percentile)
- **P95**: 95th percentile latency
- **P99**: 99th percentile latency

### Token Usage
- **Total Tokens**: Sum of input + output tokens
- **Input Tokens**: Tokens in prompts (system + context + query)
- **Output Tokens**: Tokens in LLM responses
- **By Query Type**: Token usage for simple vs complex queries
- **Average per Query**: Mean tokens per query

### Retrieval Quality
- **Avg Chunks Retrieved**: Mean number of chunks returned per query
- **Queries with Chunks**: Number of queries that retrieved at least one chunk
- **Queries without Chunks**: Number of queries with zero chunks (OOD or no matches)

### Evaluator Flags
- Frequency of each flag type:
  - `no_context`: LLM answered without relevant context
  - `refusal`: LLM refused to answer (correct behavior)
  - `unverified_feature`: LLM mentioned features not in docs
  - `pricing_uncertainty`: Pricing answer with hedging language

### Category Breakdown
- Per-category statistics:
  - Query count
  - Average latency
  - Average token usage

## Output Format

The evaluation generates a detailed text report with the following sections:

```
================================================================================
ClearPath RAG Chatbot - Evaluation Report
================================================================================

Generated: 2024-01-15 14:30:00
API URL: http://localhost:8000

--------------------------------------------------------------------------------
SUMMARY
--------------------------------------------------------------------------------
Total Queries:      115
Successful:         115
Failed:             0

--------------------------------------------------------------------------------
ROUTING PERFORMANCE
--------------------------------------------------------------------------------
Classification Accuracy:  96.52% (84/87)
Skip Retrieval Accuracy:  100.00% (15/15)

[... additional sections ...]
```

## Expected Results

### Success Criteria

Based on the task requirements:

- **Router Accuracy**: >95% correct routing
- **OOD Detection**: 100% of greetings skip retrieval
- **Edge Case Handling**: CSV/VS bug fixed, "help" context-aware
- **Evaluator Precision**: Flags only when appropriate
- **Evaluator Recall**: Catches all quality issues
- **Specific Detail Accuracy**: >90% correct on specific numbers/names/dates

### Typical Performance

On a properly configured system:

- **Latency**: P50 ~1-2s, P95 ~3-5s (depends on Groq API and HF Inference API)
- **Token Usage**: ~300-500 tokens per simple query, ~800-1500 per complex query
- **Retrieval**: ~2-4 chunks per query on average
- **Evaluator Flags**: 
  - `no_context`: ~10 occurrences (out-of-scope questions)
  - `refusal`: ~10 occurrences (unanswerable questions)
  - `unverified_feature`: <5 occurrences (hallucinations)
  - `pricing_uncertainty`: ~5 occurrences (Enterprise pricing questions)

## Troubleshooting

### API Connection Errors

If you see connection errors:

```
ERROR: HTTPConnectionPool(host='localhost', port=8000): Max retries exceeded
```

**Solution**: Ensure the backend API is running:

```bash
cd backend
python main.py
```

### Rate Limiting

If you encounter rate limit errors from Groq or Hugging Face:

**Solution**: Increase the delay between requests:

```bash
python evaluate_system.py --delay 500  # 500ms delay
```

### Timeout Errors

If queries timeout:

**Solution**: The default timeout is 30 seconds. For slower APIs, you may need to modify the timeout in `evaluate_system.py`:

```python
response = requests.post(
    f"{self.api_url}/query",
    json={"question": query.question},
    timeout=60  # Increase to 60 seconds
)
```

### Missing Test Queries

If the script reports fewer queries than expected:

**Solution**: Verify the test query loading logic in `load_test_queries()` method. The script should load 115+ queries across all categories.

## Integration with CI/CD

You can integrate the evaluation harness into your CI/CD pipeline:

```bash
#!/bin/bash
# Start backend
cd backend
python main.py &
BACKEND_PID=$!

# Wait for backend to be ready
sleep 5

# Run evaluation
python evaluate_system.py --output logs/ci_evaluation.txt

# Capture exit code
EXIT_CODE=$?

# Stop backend
kill $BACKEND_PID

# Exit with evaluation result
exit $EXIT_CODE
```

The script exits with code 1 if any queries fail, making it suitable for automated testing.

## Extending the Test Suite

To add new test queries:

1. Open `evaluate_system.py`
2. Find the `load_test_queries()` method
3. Add your queries to the appropriate category or create a new category:

```python
# New category
my_custom_queries = [
    "My custom test question 1?",
    "My custom test question 2?",
]

for i, q in enumerate(my_custom_queries, start=200):
    queries.append(TestQuery(
        id=i,
        question=q,
        expected_routing="simple",  # or "complex"
        expected_skip_retrieval=False,
        expected_flags=[],  # e.g., ["no_context"]
        category="My Custom Category"
    ))
```

## Related Files

- `test_questions_ENHANCED_routing_and_evaluator.md`: Source document with all test questions and expected behaviors
- `logs/routing_decisions.jsonl`: Detailed routing logs for each query
- `logs/evaluation_report.txt`: Generated evaluation report

## Support

For issues or questions about the evaluation harness, refer to:
- Main project README: `../README.md`
- API documentation: `../API_CONTRACT.md`
- Task specification: `../.kiro/specs/clearpath-rag-chatbot/tasks.md`
