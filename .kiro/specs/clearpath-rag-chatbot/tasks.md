# Implementation Plan: ClearPath RAG Chatbot

## Overview

This implementation plan breaks down the ClearPath RAG Chatbot into incremental coding tasks. The system will be built in Python using FastAPI for the backend, with a three-layer architecture: RAG Pipeline, Model Router, and Output Evaluator. Each task builds on previous work, with property-based tests integrated throughout to validate correctness early.

## Tasks

- [x] 1. Project setup and core infrastructure
  - Create project directory structure (backend/, frontend/, tests/, logs/, clearpath_docs/)
  - Set up Python virtual environment and dependencies (FastAPI, PyMuPDF, Groq SDK, Hypothesis, pytest, psycopg2, supabase-py)
  - Create .env.example with required environment variables (GROQ_API_KEY, HUGGINGFACE_API_KEY, SUPABASE_URL, SUPABASE_KEY, PORT, LOG_LEVEL)
  - Set up Supabase project and enable pgvector extension
  - Set up logging configuration with JSON structured logging
  - Create main.py entry point with FastAPI app initialization
  - Initialize Next.js frontend project
  - _Requirements: 18.2, 18.3_

- [ ] 2. Document loading and PDF processing
  - [x] 2.1 Implement DocumentLoader class
    - Create Document and Page dataclasses
    - Implement load_documents() to read all PDFs from clearpath_docs/
    - Extract text page-by-page with page number tracking using PyMuPDF
    - Handle corrupted PDFs gracefully (log error, skip file)
    - _Requirements: 1.1, 1.2_
  
  - [ ]* 2.2 Write property test for document loading
    - **Property 1: Chunk Metadata Preservation**
    - **Validates: Requirements 1.4**
  
  - [ ]* 2.3 Write unit tests for PDF processing
    - Test loading specific PDF files
    - Test handling of corrupted PDFs
    - Test page number tracking accuracy
    - _Requirements: 1.1, 1.2_

- [x] 3. Document chunking engine
  - [x] 3.1 Implement ChunkingEngine class with contextual heading injection
    - Create Chunk dataclass with text, document_name, page_number, chunk_id, context_header
    - Implement token-aware recursive splitting (300 tokens, 50 overlap)
    - Use AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2") to measure token count for embeddings
    - For Llama 3 prompt token counting, use tiktoken with o200k_base encoding
    - Implement recursive splitting with separators: ["\n\n", "\n", ". ", " ", ""]
    - Implement hierarchical header stack state machine that persists across pages
      - Use page.get_text("dict") in PyMuPDF to detect font sizes
      - Treat text with font size > 12pt as headers, use size thresholds for H1/H2/H3 hierarchy
      - Maintain current_header_stack that updates when new headers of equal/higher hierarchy found
      - Example: If "Pricing" is H1 on page 2, all chunks until next H1 get "[Context: Pricing]"
      - Format: "[Context: {H1} > {H2}] {chunk_text}" for nested headers
    - Try splitting at paragraphs first, then sentences, then words, characters as last resort
    - Preserve document and page metadata in each chunk
    - Generate unique chunk_id format: "{filename}_{page}_{chunk_index}"
    - _Requirements: 1.3, 1.4_
  
  - [ ]* 3.2 Write property test for chunking
    - **Property 1: Chunk Metadata Preservation**
    - Test that all chunks contain source filename and page number
    - **Validates: Requirements 1.4**
  
  - [ ]* 3.3 Write unit tests for chunking edge cases
    - Test chunking of very short documents
    - Test chunking with various overlap sizes
    - Test chunk boundary handling
    - _Requirements: 1.3_

- [x] 4. Vector store and embedding system
  - [x] 4.1 Implement embedding model integration with Hugging Face Inference API
    - Create EmbeddingModel wrapper class
    - Initialize Hugging Face API client with API key
    - Implement embed_text() and embed_batch() methods using all-mpnet-base-v2 model
    - Implement aggressive retry-backoff strategy for 503 Service Unavailable errors
      - HF free tier models "sleep" and take 15-20s to load on first query
      - Use exponential backoff with max retries (e.g., 5 retries with 5s, 10s, 20s, 40s, 60s delays)
      - Log model loading delays for monitoring
      - Consider warming up model at startup with dummy query
    - Add error handling for API failures and rate limits
    - _Requirements: 1.5_
  
  - [x] 4.2 Implement VectorStore class using Supabase pgvector
    - Create ScoredChunk dataclass
    - Set up Supabase client connection
    - Create database table for chunks with vector column
    - Implement add_chunks() to store embeddings in Supabase using batch embedding
    - Use embed_batch() to send multiple chunks in one API call (stay under rate limits)
    - Implement search() with cosine similarity using pgvector
    - Add relevance score normalization to [0, 1] range
    - _Requirements: 1.5, 2.3_
  
  - [ ]* 4.3 Write property test for relevance scores
    - **Property 2: Relevance Score Bounds**
    - Test that all relevance scores are between 0.0 and 1.0
    - **Validates: Requirements 2.3**
  
  - [ ]* 4.4 Write unit tests for vector store
    - Test adding chunks and searching
    - Test empty search results
    - Test Supabase connection and pgvector queries
    - _Requirements: 1.5_

- [~] 5. Retrieval engine
  - [-] 5.1 Implement RetrievalEngine class with dynamic K-cutoff
    - Initialize with VectorStore and EmbeddingModel
    - Implement retrieve() method with query embedding
    - Apply relevance threshold (score > 0.3) to filter low-quality matches
    - Implement dynamic K-cutoff: retrieve up to k=5, but only include chunks within 20% of top score
      - Example: If top chunk is 0.85, only include chunks with score >= 0.68 (0.85 * 0.8)
      - Rationale: Prevents "Lost in the Middle" problem where low-relevance chunks hurt LLM performance
    - Return filtered chunks sorted by relevance
    - Handle empty query strings gracefully
    - _Requirements: 2.1, 2.2, 2.4_
  
  - [ ]* 5.2 Write property test for retrieval ordering
    - **Property 3: Retrieval Result Ordering**
    - Test that chunks are sorted by relevance score descending
    - **Validates: Requirements 2.4**
  
  - [ ]* 5.3 Write property test for selective retrieval
    - **Property 4: Selective Retrieval**
    - Test that retrieved chunks < total chunks in corpus
    - **Validates: Requirements 2.2**
  
  - [ ]* 5.4 Write unit tests for retrieval edge cases
    - Test empty query handling
    - Test no relevant results scenario
    - Test retrieval with various top_k values
    - _Requirements: 2.1, 2.5_

- [~] 6. Checkpoint - Ensure RAG pipeline works end-to-end
  - Ensure all tests pass, ask the user if questions arise.

- [~] 7. Model router with deterministic classification and OOD filter
  - [~] 7.1 Implement ModelRouter class with decision tree logic
    - Create Classification dataclass
    - Implement classify_query() using tiered rule list (NOT weighted scoring)
    - Rule 0 (OOD Filter): If query is greeting ("hi", "hello", "hey", "thanks") or meta-comment ("who are you", "what can you do") → Simple + skip_retrieval flag
    - Rule 1: If query contains complex keywords ("why", "how", "explain", "compare", "analyze", "difference") → Complex
    - Rule 2: Else if query length > 15 words → Complex
    - Rule 3: Else if multiple question marks (>1) → Complex
    - Rule 4: Else if contains comparison words ("versus", "vs", "better", "worse") → Complex
    - Rule 5: Else → Simple
    - Log classification reasoning for each decision
    - _Requirements: 4.1, 4.3, 4.4, 4.5, 4.6_
  
  - [ ]* 7.2 Write property test for classification determinism
    - **Property 6: Classification Determinism**
    - Test that same query always produces same classification
    - **Validates: Requirements 4.5**
  
  - [ ]* 7.3 Write property test for classification validity
    - **Property 7: Classification Validity**
    - Test that classification is always "simple" or "complex"
    - **Validates: Requirements 4.1**
  
  - [ ]* 7.4 Write property test for model selection consistency
    - **Property 8: Model Selection Consistency**
    - Test simple → 8b model, complex → 70b model
    - **Validates: Requirements 4.3, 4.4**
  
  - [ ]* 7.5 Write unit tests for routing examples
    - Test specific queries with expected classifications
    - Test boundary cases (exactly 15 words, edge of complex keyword detection)
    - Test decision tree logic at each rule level
    - _Requirements: 4.1, 4.3, 4.4_

- [~] 8. Groq API client integration
  - [~] 8.1 Implement LLMClient class
    - Create LLMResponse dataclass
    - Initialize Groq client with API key from environment
    - Implement generate() method with model selection
    - Build prompt template with context and query
    - Track token usage (input/output) from API response
    - Measure latency for each API call
    - _Requirements: 13.1, 13.2, 13.3_
  
  - [~] 8.2 Implement error handling for Groq API
    - Handle rate limit errors (429) with retry suggestion
    - Handle authentication errors (401)
    - Handle network timeouts
    - Return structured error responses
    - Log all API errors with full context
    - _Requirements: 13.5_
  
  - [ ]* 8.3 Write property test for API error handling
    - **Property 21: API Error Graceful Handling**
    - Test that API errors return error response, not crash
    - **Validates: Requirements 13.5**
  
  - [ ]* 8.4 Write unit tests for LLM client
    - Test prompt template formatting
    - Test token counting
    - Test latency measurement
    - _Requirements: 13.2, 13.3_

- [~] 9. Output evaluator with quality checks
  - [~] 9.1 Implement OutputEvaluator class with four quality checks
    - Implement evaluate() method returning list of flags
    - Implement no-context detection (chunks_retrieved == 0, not refusal)
    - Implement refusal detection (check for refusal phrases)
    - Implement groundedness check (unverified_feature detection)
      - Extract proper nouns from LLM response using regex/NER (capitalized terms, integration names)
      - Extract proper nouns from retrieved chunks
      - Compare: if LLM mentions proper noun not in chunks, flag as unverified_feature
      - Example: LLM says "Slack integration" but no chunks contain "Slack" → flag it
      - Rationale: Llama models hallucinate features based on general SaaS knowledge; this catches invented integrations
    - Implement pricing_uncertainty detection (pricing query + hedging language or conflicting sources)
    - _Requirements: 6.1, 6.2, 7.1, 7.2, 8.1, 8.2_
  
  - [ ]* 9.2 Write property test for no-context detection
    - **Property 10: No-Context Flag Detection**
    - Test that no_context flag raised when chunks=0 and not refusal
    - **Validates: Requirements 6.1, 6.2, 6.3**
  
  - [~] 9.3 Write property test for groundedness detection
    - **Property 11: Groundedness Flag Detection**
    - Test that unverified_feature flag raised for proper nouns not in chunks
    - **Validates: Requirements 7.1, 7.3**
  
  - [ ]* 9.4 Write property test for refusal detection
    - **Property 12: Refusal Flag Detection**
    - Test that refusal flag raised for refusal phrases
    - **Validates: Requirements 7.1, 7.3**
  
  - [ ]* 9.5 Write property test for pricing uncertainty detection
    - **Property 13: Domain-Specific Flag Detection**
    - Test pricing_uncertainty flag for hedging language
    - **Validates: Requirements 8.2, 8.3**
  
  - [ ]* 9.6 Write property test for non-blocking flags
    - **Property 14: Non-Blocking Flags**
    - Test that answer is still returned when flags raised
    - **Validates: Requirements 6.4, 7.4, 8.4**
  
  - [ ]* 9.7 Write unit tests for evaluator
    - Test specific refusal phrases
    - Test specific pricing uncertainty scenarios
    - Test proper noun extraction and matching
    - Test flag combinations
    - _Requirements: 6.1, 7.1, 8.1_

- [~] 10. Routing decision logging system
  - [~] 10.1 Implement RoutingLogger class with complexity scoring
    - Create log_routing_decision() method
    - Format logs as JSON Lines
    - Write to logs/routing_decisions.jsonl
    - Include all required fields: query, classification, model_used, tokens, latency
    - Add rule_triggered field to track which decision tree rule was applied
      - Values: "ood_filter", "complex_keyword", "query_length", "multiple_questions", "comparison_words", "default"
      - Rationale: Helps identify misclassified queries for Q1 analysis
    - Add complexity_score object with data-backed metrics:
      - word_count: Number of words in query
      - complex_keyword_count: Number of complex keywords found
      - question_mark_count: Number of question marks
      - comparison_word_count: Number of comparison words found
      - Rationale: Makes Q1 written answer easier with concrete evidence
    - Implement log rotation (daily)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_
  
  - [ ]* 10.2 Write property test for logging completeness
    - **Property 9: Routing Decision Log Completeness**
    - Test that all required fields present in log entries
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**
  
  - [ ]* 10.3 Write unit tests for logging
    - Test log file creation
    - Test JSON formatting
    - Test log rotation
    - _Requirements: 5.1_

- [~] 11. Checkpoint - Ensure core processing pipeline works
  - Ensure all tests pass, ask the user if questions arise.

- [~] 12. Conversation manager for multi-turn support
  - [~] 12.1 Implement ConversationManager class
    - Create Conversation and Turn dataclasses
    - Implement get_or_create_conversation() with ID generation
    - Implement add_turn() to store query-response pairs
    - Implement get_context() to format last N turns for prompt
    - Use Supabase PostgreSQL for conversation storage
    - Create database tables for conversations and turns
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_
  
  - [ ]* 12.2 Write property test for conversation ID uniqueness
    - **Property 19: Conversation ID Uniqueness**
    - Test that new conversations have unique IDs
    - **Validates: Requirements 15.4**
  
  - [ ]* 12.3 Write property test for conversation persistence
    - **Property 18: Conversation History Persistence**
    - Test that added turns are retrievable
    - **Validates: Requirements 15.5**
  
  - [ ]* 12.4 Write property test for context retrieval
    - **Property 17: Conversation Context Retrieval**
    - Test that valid conversation_id retrieves history
    - **Validates: Requirements 15.1**
  
  - [ ]* 12.5 Write unit tests for conversation manager
    - Test conversation creation
    - Test turn addition
    - Test context formatting
    - _Requirements: 15.1, 15.4_

- [~] 13. API endpoint and request handling
  - [~] 13.1 Implement API models with Pydantic
    - Create QueryRequest model (question, optional conversation_id)
    - Create QueryResponse model (answer, metadata, sources, conversation_id)
    - Create ResponseMetadata model with token breakdown
      - Fields: model_used, classification, tokens (with input, output), latency_ms, chunks_retrieved, evaluator_flags
      - Note: Use tiktoken with o200k_base or transformers.AutoTokenizer for accurate Llama 3 token counting
      - Rationale: all-mpnet-base-v2 tokenizer has different vocabulary than Llama 3, will underestimate costs
    - Create TokenUsage model (input, output)
    - Create Source model (document, page, relevance_score)
    - _Requirements: 9.1, 9.2, 9.3, 9.5, 10.1_
  
  - [~] 13.2 Implement POST /query endpoint with accurate token tracking
    - Validate request body (required question field)
    - Get or create conversation
    - Classify query using ModelRouter (check for skip_retrieval flag)
    - If skip_retrieval is False: Retrieve relevant chunks using RetrievalEngine
    - If skip_retrieval is True: Skip retrieval, set chunks_retrieved to 0
    - Build prompt with context and conversation history
    - Use tiktoken (o200k_base) to count Llama 3 tokens accurately
    - Track input tokens (system prompt + context + query) and output tokens separately
    - Generate response using LLMClient
    - Evaluate response using OutputEvaluator
    - Log routing decision using RoutingLogger (include rule_triggered and complexity_score)
    - Format and return QueryResponse with token counts
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  
  - [ ]* 13.3 Write property test for conversation ID generation
    - **Property 14: Conversation ID Generation**
    - Test that requests without conversation_id get new ID
    - **Validates: Requirements 9.4**
  
  - [ ]* 13.4 Write property test for response schema completeness
    - **Property 15: Response Schema Completeness**
    - Test that all required top-level fields present
    - **Validates: Requirements 9.5**
  
  - [ ]* 13.5 Write property test for metadata completeness
    - **Property 16: Metadata Schema Completeness**
    - Test that all metadata fields present
    - **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7**
  
  - [ ]* 13.6 Write property test for source attribution
    - **Property 5: Source Attribution Completeness**
    - Test that source entries have required fields
    - **Validates: Requirements 3.2, 3.3, 3.4**
  
  - [ ]* 13.7 Write integration tests for API endpoint
    - Test complete request-response flow
    - Test multi-turn conversations
    - Test error responses
    - _Requirements: 9.1, 9.5_

- [~] 14. Web interface with chat UI and debug panel
  - [~] 14.1 Create Next.js chat interface
    - Set up Next.js project with TypeScript
    - Create chat page with message input and submit button
    - Implement message display area with conversation history
    - Style with Tailwind CSS or similar for clean, minimal design
    - _Requirements: 11.1, 11.2, 11.4_
  
  - [~] 14.2 Implement debug panel in Next.js UI
    - Create debug panel component
    - Display model_used for each response
    - Display token usage (input/output)
    - Display evaluator_flags with visual indicators
    - Update debug panel with each new response
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_
  
  - [~] 14.3 Implement API client in Next.js
    - Create API route or client-side fetch() call to POST /query
    - Handle response and update UI state
    - Handle errors and display error messages
    - Maintain conversation_id across requests using React state
    - _Requirements: 11.2, 11.3_
  
  - [~] 14.4 Configure CORS in FastAPI for Next.js frontend
    - Add CORS middleware to FastAPI app
    - Allow requests from Next.js development server
    - _Requirements: 11.1, 11.4_
  
  - [ ]* 14.5 Write UI integration tests
    - Test that Next.js interface loads successfully
    - Test that debug panel displays metadata
    - _Requirements: 11.1, 12.1_

- [~] 15. Checkpoint - Ensure complete system works end-to-end
  - Ensure all tests pass, ask the user if questions arise.

- [~] 16. Streaming response support (bonus feature)
  - [~] 16.1 Implement streaming in LLMClient
    - Add generate_stream() method using Groq streaming API
    - Yield tokens as they arrive
    - Track metadata during streaming
    - _Requirements: 16.1, 16.2_
  
  - [~] 16.2 Add streaming endpoint to API
    - Create POST /query/stream endpoint
    - Use FastAPI StreamingResponse
    - Send tokens as Server-Sent Events (SSE)
    - Send final metadata after stream completes
    - _Requirements: 16.1, 16.2_
  
  - [ ]* 16.3 Write property test for streaming metadata
    - **Property 20: Streaming Metadata Completeness**
    - Test that final metadata is complete after streaming
    - **Validates: Requirements 16.4, 16.5**
  
  - [~] 14.4 Update UI to support streaming
    - Implement EventSource or fetch with streaming to receive SSE
    - Display tokens progressively as they arrive in Next.js UI
    - Update debug panel after stream completes
    - _Requirements: 16.3_
  
  - [ ]* 16.5 Write integration tests for streaming
    - Test streaming response delivery
    - Test metadata completeness after streaming
    - _Requirements: 16.1, 16.5_

- [~] 17. Evaluation test harness (bonus feature)
  - [~] 17.1 Create test query dataset
    - Create tests/test_queries.json with diverse queries
    - Include simple factual questions
    - Include complex analytical questions
    - Include multi-turn conversation examples
    - Include edge cases (no relevant docs, ambiguous)
    - _Requirements: 17.1_
  
  - [~] 17.2 Implement evaluation harness script
    - Load test queries from JSON
    - Execute all queries through API
    - Collect responses and metadata
    - Calculate routing accuracy metrics
    - Calculate retrieval quality metrics
    - Measure latency distribution (p50, p95, p99)
    - Track token usage by query type
    - Track evaluator flag frequency
    - _Requirements: 17.2, 17.3, 17.4_
  
  - [~] 17.3 Generate evaluation report
    - Format results as summary report
    - Include all metrics from test run
    - Save report to logs/evaluation_report.txt
    - _Requirements: 17.5_
  
  - [ ]* 17.4 Write unit tests for evaluation harness
    - Test query loading
    - Test metric calculation
    - Test report generation
    - _Requirements: 17.1, 17.5_

- [~] 18. Documentation and deployment preparation
  - [~] 18.1 Create comprehensive README.md
    - Document project overview and architecture
    - List all required environment variables
    - Provide setup instructions (venv, dependencies, .env)
    - Provide instructions to run locally on localhost:8000
    - Document API endpoints and request/response formats
    - Include example queries and expected responses
    - _Requirements: 18.1, 18.4_
  
  - [~] 18.2 Create written_answers.md with detailed analysis
    - Answer Q1: Routing Logic
      - Document exact decision tree rules including OOD filter
      - Explain boundary reasoning (why 15 words, why these keywords)
      - Use rule_triggered and complexity_score logs to find real misclassification example
      - Show data: word_count, keyword_count for misclassified query
      - Discuss improvements (e.g., conversation-aware routing)
    - Answer Q2: Retrieval Failures
      - Discuss groundedness check (unverified_feature) and contextual heading injection improvements
      - Provide real or constructed failure case with analysis
      - Explain how hierarchical header stack prevents context loss across pages
      - Explain how dynamic K-cutoff prevents "Lost in the Middle"
    - Answer Q3: Cost and Scale
      - Use tiktoken (o200k_base) to estimate Llama 3 token usage for 5k queries/day
      - Show working: system_prompt_tokens (fixed) + context_tokens (variable) + output tokens
      - Discuss OOD filter ROI (saves embedding + LLM costs for greetings)
      - Note: Don't use all-mpnet-base-v2 tokenizer for Llama 3 cost estimation (different vocabulary)
      - Identify cost drivers and optimization strategies
    - Answer Q4: What Is Broken
      - Discuss stateless router limitation in multi-turn conversations
      - Explain groundedness check addresses hallucination but doesn't stop "helpful" LLM
      - Discuss HF Inference API cold start delays (15-20s on free tier)
      - Justify why shipped anyway (MVP trade-offs, deterministic requirement)
      - Propose fix approach (conversation-aware routing, stricter prompting, model warming)
    - Each answer 150-250 words
    - _Requirements: Project deliverables_
  
  - [~] 18.3 Add deployment configuration files
    - Create requirements.txt with all Python dependencies
    - Create package.json for Next.js frontend
    - Create Dockerfile for backend (optional)
    - Create deployment instructions for Vercel (frontend) and Railway/Render (backend)
    - Document Supabase setup requirements
    - _Requirements: 18.5_
  
  - [ ]* 18.4 Write deployment verification tests
    - Test that environment variables are loaded correctly
    - Test that server starts on configured port
    - _Requirements: 18.2, 18.3_

- [~] 19. Final checkpoint - Complete system validation
  - Run full test suite (unit + property + integration)
  - Run evaluation harness and review metrics
  - Test complete user flow through web interface
  - Verify all logging is working correctly
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at major milestones
- Property tests validate universal correctness properties across random inputs
- Unit tests validate specific examples, edge cases, and error conditions
- Integration tests validate end-to-end system behavior
- The implementation uses Python with FastAPI, Supabase pgvector, Hugging Face Inference API, and Groq SDK
- Frontend uses Next.js with TypeScript
- Streaming support (Task 16) and evaluation harness (Task 17) are bonus features
