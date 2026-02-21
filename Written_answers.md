# Written Answers: ClearPath RAG Chatbot Analysis

## Q1: Routing Logic

The ModelRouter uses a deterministic decision tree with six rules applied in order:

**Rule 0 (OOD Filter):** Detects greetings ("hi", "hello", "hey", "thanks") and meta-queries ("who are you", "what can you do") → Routes to Simple model with `skip_retrieval=True`. Uses regex word boundaries to prevent false positives. For example, "help" only triggers if it's a standalone request (≤3 words), not "I need help with my server". This saves embedding and retrieval costs for non-informational queries.

**Rule 1:** Complex keywords ("explain", "compare", "analyze", "difference", "relationship") → Complex model (70b)

**Rule 2:** Query length > 15 words → Complex model

**Rule 3:** Multiple question marks (>1) → Complex model

**Rule 4:** Comparison words ("versus", "vs", "better", "worse", "compared to") → Complex model. Uses word boundaries to prevent false positives like "csv" matching "vs".

**Rule 5 (Default):** All other queries → Simple model (8b)

**Boundary Reasoning:** The 15-word threshold balances precision and recall. Short queries are typically factual lookups ("What is the pricing?"), while longer queries often require reasoning. Complex keywords directly indicate analytical intent. Regex word boundaries prevent substring matching bugs.

**Real Misclassification Example:** From routing logs, the query "What integrations does ClearPath support" (6 words, word_count=6, complex_keyword_count=0) was classified as Simple, but required synthesizing information across multiple chunks. The router missed this because "integrations" isn't a complex keyword and the query is short.

**Improvements:** Add domain-specific keywords like "integrations", "features", "capabilities" to Rule 1. Implement conversation-aware routing that escalates to Complex model when Simple model previously flagged unverified_feature or refusal. The current implementation already includes robust regex patterns to prevent false positives in OOD detection and comparison word matching.

## Q2: Retrieval Failures

Two key improvements address retrieval failures:

**Contextual Heading Injection:** The ChunkingEngine maintains a hierarchical header stack across pages using font size detection. When "Pricing" appears as H1 on page 2, all subsequent chunks until the next H1 receive "[Context: Pricing]" prefix. This prevents context loss when chunks are retrieved out of order. For example, a chunk saying "Enterprise plan includes unlimited users" becomes "[Context: Pricing > Enterprise Plan] Enterprise plan includes unlimited users", making it self-contained.

**Dynamic K-Cutoff:** The RetrievalEngine retrieves up to k=5 chunks but only includes those within 20% of the top score. If the top chunk scores 0.85, only chunks ≥0.68 are included. This prevents the "Lost in the Middle" problem where low-relevance chunks dilute context and confuse the LLM.

**Groundedness Check:** The OutputEvaluator extracts proper nouns from the LLM response using regex patterns that handle Markdown formatting and normalize casing. It looks for capitalized words mid-sentence and common integration names (Slack, GitHub, Jira, etc.). All proper nouns are stored in lowercase for case-insensitive comparison. If the LLM mentions "Slack integration" but no chunks contain "slack", it flags `unverified_feature`. The implementation strips possessives ("ClearPath's" → "ClearPath") and handles Markdown list markers to prevent false positives. This catches hallucinations where Llama models invent features based on general SaaS knowledge.

**Failure Case:** Query: "Does ClearPath integrate with Salesforce?" If no chunks mention Salesforce, but the LLM responds "Yes, ClearPath integrates with Salesforce via API", the groundedness check flags this. The contextual headers ensure that if integration information exists, it's properly labeled with "[Context: Integrations Catalog]" for accurate retrieval. The improved proper noun extraction prevents false positives from sentence-initial capitalization and common stop words.

## Q3: Cost and Scale

**Token Usage Estimation (5,000 queries/day):**

Using tiktoken with o200k_base encoding for accurate Llama 3 token counting:

- System prompt: ~150 tokens (fixed)
- Average context (3 chunks × 300 tokens): ~900 tokens
- Average query: ~20 tokens
- Average output: ~150 tokens
- **Input tokens per query:** 150 + 900 + 20 = 1,070 tokens
- **Output tokens per query:** 150 tokens

**Daily totals:**
- Simple model (70% of queries): 3,500 × 1,220 tokens = 4.27M tokens
- Complex model (30% of queries): 1,500 × 1,220 tokens = 1.83M tokens
- **Total daily:** ~6.1M tokens

**Monthly cost estimate (assuming Groq pricing):**
- Llama 3 8b: ~$0.05-0.10 per 1M tokens
- Llama 3 70b: ~$0.50-0.80 per 1M tokens
- Simple model: 128M tokens/month × $0.10 = $12.80
- Complex model: 55M tokens/month × $0.80 = $44.00
- **Total monthly LLM cost:** ~$57

**OOD Filter ROI:** Greetings/meta-queries (~10% of traffic) skip embedding ($0.20/1M tokens) and retrieval, saving ~$3/month plus reduced latency.

**Cost drivers:** Context size (900 tokens/query) is the largest driver. Optimizations: reduce chunk size, implement caching for repeated queries, use smaller embedding models, compress context with summarization.

## Q4: What Is Broken

**Stateless Router in Multi-Turn Conversations:** The ModelRouter classifies each query independently without conversation history. If a user asks "What is the pricing?" (Simple) then "Why is it structured that way?" (Complex), the second query lacks context. The router sees "why" and routes to Complex, but doesn't know it's a follow-up. A conversation-aware router could escalate to Complex when previous turns had evaluator flags or maintain complexity across related turns.

**Groundedness Check Limitations:** While the unverified_feature flag detects hallucinated proper nouns using sophisticated regex patterns and case-insensitive matching, it doesn't prevent the LLM from being "helpful" by inferring answers. If asked "Can I export data to CSV?" and no chunks mention CSV, the LLM might say "Most SaaS tools support CSV export" rather than "I don't have information about CSV export." The groundedness check flags this, but the user still receives a misleading answer. The improved refusal detection now distinguishes between pure refusals and partial answers: responses with contrast words ("but", "however") and >12 words are treated as successful partial answers, not refusals. This prevents false positives when the LLM provides useful information despite some limitations.

**HF Inference API Cold Starts:** The free-tier all-mpnet-base-v2 model "sleeps" and takes 15-20 seconds to load on first query. Despite aggressive retry-backoff (5 retries with exponential delays), the first user experiences significant latency. This creates a poor initial impression.

**Why Shipped Anyway:** These are acceptable MVP trade-offs. The deterministic routing requirement (no LLM-based classification) prevents conversation-aware routing without significant architecture changes. The groundedness check provides transparency via flags, allowing users to verify answers. Cold start delays affect only the first query and resolve automatically.

**Fix Approach:** (1) Implement conversation-aware routing that tracks evaluator flags and escalates complexity. (2) Add stricter system prompts: "Only answer if information is in the provided context. If uncertain, say 'I don't have information about that.'" (3) Warm up the embedding model at startup with a dummy query to eliminate cold starts.

## AI Usage

This section documents all prompts given to LLMs during the development of this project:

1. "Help me understand how to implement a model router that uses deterministic rules to classify queries as simple or complex"

2. "How can I extract proper nouns from text using regex in Python while handling markdown formatting"

3. "What's the best way to implement retry logic with exponential backoff for API calls in Python"

4. "Explain how to use tiktoken to count tokens for Llama 3 models"

5. "How do I implement contextual heading injection in a document chunking system"

6. "What are common patterns for implementing groundedness checks in RAG systems"

7. "Help me debug why my regex pattern for detecting greetings is matching words like 'help' inside longer sentences"

8. "How can I implement a dynamic k-cutoff for vector similarity search results"

9. "What's the best way to handle HuggingFace Inference API cold starts"

10. "Explain the 'Lost in the Middle' problem in RAG systems and how to mitigate it"

11. "How do I implement conversation-aware routing without using an LLM classifier"

12. "Help me write a system prompt that prevents LLMs from hallucinating features not in the context"

13. "What are best practices for cost optimization in production RAG systems"

14. "How can I distinguish between pure refusals and partial answers in LLM responses"

15. "Help me implement proper noun extraction that handles possessives and markdown list markers"
