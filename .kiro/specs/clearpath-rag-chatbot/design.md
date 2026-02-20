# Design Document: ClearPath RAG Chatbot

## Overview

The ClearPath RAG Chatbot is a three-layer customer support system that combines document retrieval, intelligent model routing, and response quality evaluation. The architecture prioritizes transparency, cost-efficiency, and reliability while maintaining simplicity and avoiding external RAG frameworks.

The system processes user queries through three sequential layers:
1. **RAG Pipeline**: Retrieves relevant document chunks from 30 PDF files using vector similarity search
2. **Model Router**: Classifies queries and routes them to appropriate LLM models using deterministic rules
3. **Output Evaluator**: Analyzes responses and flags potential quality issues

The design emphasizes observability through comprehensive logging and metadata exposure, enabling debugging and continuous improvement.

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Interface                          │
│                  (Chat UI + Debug Panel)                    │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP POST /query
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway                             │
│              (Request/Response Handler)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Document   │  │   Chunking   │  │   Vector     │     │
│  │   Loader     │─▶│   Engine     │─▶│   Store      │     │
│  └──────────────┘  └──────────────┘  └──────┬───────┘     │
│                                              │              │
│  ┌──────────────────────────────────────────▼───────┐     │
│  │          Retrieval Engine                        │     │
│  │  (Query → Relevant Chunks + Scores)              │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │ Retrieved Chunks
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Model Router                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Query Classifier (Rule-Based)                   │      │
│  │  • Length Analysis                               │      │
│  │  • Keyword Detection                             │      │
│  │  • Complexity Heuristics                         │      │
│  └────────────┬─────────────────────────────────────┘      │
│               │                                             │
│      ┌────────┴────────┐                                   │
│      ▼                 ▼                                    │
│  llama-3.1-8b    llama-3.3-70b                             │
│  (simple)        (complex)                                  │
└────────────────────────┬────────────────────────────────────┘
                         │ Generated Response
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 Output Evaluator                            │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Quality Checks:                                 │      │
│  │  • No-Context Detection                          │      │
│  │  • Refusal Detection                             │      │
│  │  • Pricing Uncertainty Detection                 │      │
│  └──────────────────────────────────────────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │ Response + Flags
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Response Formatter                             │
│  (Assembles final JSON with metadata)                      │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. User submits query via web interface
2. API Gateway receives POST /query request
3. RAG Pipeline retrieves relevant document chunks
4. Model Router classifies query and selects appropriate LLM
5. LLM generates response using retrieved context
6. Output Evaluator analyzes response and raises flags
7. Response Formatter assembles final JSON with metadata
8. API Gateway returns response to web interface

### Technology Stack

- **Backend Framework**: FastAPI (Python)
- **PDF Processing**: PyMuPDF
- **Vector Search**: Supabase pg vector
- **Embeddings**: use hugging face inference api (e.g., all-mpnet-base-v2)
- **LLM API**: Groq API (llama-3.1-8b-instant, llama-3.3-70b-versatile)
- **Frontend**: Next Js
- **Storage**:  use supabase for postgresql
- **Logging**: Python logging module

## Components and Interfaces

### 1. Document Loader

**Responsibility**: Load and extract text from PDF files with page tracking.

**Interface**:
```python
class DocumentLoader:
    def load_documents(self, directory_path: str) -> List[Document]:
        """
        Load all PDF files from directory.
        
        Args:
            directory_path: Path to clearpath_docs/ folder
            
        Returns:
            List of Document objects with text, filename, and page numbers
        """
        pass

class Document:
    filename: str
    pages: List[Page]
    
class Page:
    page_number: int
    text: str
```

**Implementation Notes**:
- Use PyPDF2 or pdfplumber for PDF text extraction
- Handle malformed PDFs gracefully (log errors, skip corrupted pages)
- Preserve page boundaries for accurate source attribution
- Extract text page-by-page to maintain page number tracking

### 2. Chunking Engine

**Responsibility**: Segment documents into retrievable chunks while preserving context.

**Interface**:
```python
class ChunkingEngine:
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Chunk documents using strategic segmentation.
        
        Args:
            documents: List of loaded documents
            
        Returns:
            List of Chunk objects with text, metadata, and embeddings
        """
        pass

class Chunk:
    text: str
    document_name: str
    page_number: int
    chunk_id: str
    embedding: Optional[List[float]] = None
```

**Chunking Strategy**:
- **Token-Aware Recursive Splitting with Contextual Heading Injection**: 300 tokens per chunk, 50 token overlap
- Use tokenizer to measure text "weight" rather than character count
- Target chunk size: 300 tokens (leaves 50-token buffer for metadata to stay under 384 total)
- Chunk overlap: 50 tokens (prevents sentences from being cut where meaning is lost)
- **Contextual Heading Injection**: Maintain a hierarchical header stack that persists across pages
  - Use PyMuPDF's page.get_text("dict") to detect font sizes
  - Treat text with font size > 12pt as headers
  - Track header hierarchy (H1 > H2 > H3) using font size thresholds
  - Maintain current_header_stack that updates when new headers of equal/higher hierarchy are found
  - Example: If "Pricing" is H1 on page 2, all chunks until next H1 get "[Context: Pricing]"
  - Format: "[Context: {H1} > {H2}] {chunk_text}" for nested headers
  - Rationale: Provides persistent topical context across pages, dramatically reduces retrieval failures
- Separators (in priority order): ["\n\n", "\n", ". ", " ", ""]
  - Try splitting at paragraphs first (\n\n)
  - Then sentences (\n, ". ")
  - Then words (" ")
  - Only split characters as last resort ("")
- Rationale: Balances context preservation with retrieval granularity while respecting semantic boundaries
- Preserve document and page metadata for each chunk

**Implementation Notes**:
- Use AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2") to measure token count for embeddings
- For Llama 3 prompt token counting, use tiktoken with o200k_base encoding or transformers.AutoTokenizer for llama-3.1
- Implement recursive splitting with separators: ["\n\n", "\n", ". ", " ", ""]
- Try splitting at semantic boundaries (paragraphs, sentences) before words or characters
- Maintain header_stack state machine across pages to ensure contextual continuity
- Store chunk metadata for source attribution

### 3. Vector Store

**Responsibility**: Store chunk embeddings and enable similarity search.

**Interface**:
```python
class VectorStore:
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks with embeddings to the vector store."""
        pass
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[ScoredChunk]:
        """
        Find most similar chunks to query.
        
        Args:
            query_embedding: Embedding vector for user query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of chunks with relevance scores (0-1)
        """
        pass

class ScoredChunk:
    chunk: Chunk
    relevance_score: float
```

**Implementation Options**:
1. **FAISS** (recommended): Fast, efficient, supports large-scale similarity search
2. **ChromaDB**: Simple API, built-in embedding support
3. **Custom numpy implementation**: Cosine similarity with brute-force search (acceptable for 30 docs)

**Embedding Model**:
- Use sentence-transformers: `all-MiniLM-L6-v2` (384 dimensions, fast, good quality)
- Alternative: `all-mpnet-base-v2` (768 dimensions, higher quality, slower)

### 4. Retrieval Engine

**Responsibility**: Orchestrate query embedding and chunk retrieval.

**Interface**:
```python
class RetrievalEngine:
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
    
    def retrieve(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        """
        Retrieve relevant chunks for query.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            
        Returns:
            List of scored chunks, empty if no relevant results
        """
        pass
```

**Retrieval Logic**:
1. Embed user query using same model as chunks
2. Perform similarity search in vector store
3. Apply relevance threshold (e.g., score > 0.3) to filter low-quality matches
4. Apply dynamic K-cutoff: retrieve up to k=5, but only include chunks within 20% of top score
   - Example: If top chunk scores 0.85, only include chunks with score >= 0.68
   - Rationale: Prevents "Lost in the Middle" problem where low-relevance chunks hurt LLM performance
5. Return filtered chunks sorted by relevance score

### 5. Model Router

**Responsibility**: Classify queries and route to appropriate LLM using deterministic rules.

**Interface**:
```python
class ModelRouter:
    def classify_query(self, query: str) -> Classification:
        """
        Classify query as simple or complex.
        
        Args:
            query: User question
            
        Returns:
            Classification with category and selected model
        """
        pass

class Classification:
    category: str  # "simple" or "complex"
    model_name: str  # "llama-3.1-8b-instant" or "llama-3.3-70b-versatile"
    reasoning: str  # Explanation of classification decision
```

**Classification Rules** (Deterministic):

The router uses a tiered decision tree with an OOD (Out-of-Distribution) filter:

**Decision Tree Logic**:
0. **Rule 0 - OOD Filter**: If query is greeting or meta-comment → Route to llama-3.1-8b-instant + skip retrieval
   - Greetings: "hi", "hello", "hey", "thanks", "thank you"
   - Meta-comments: "who are you", "what can you do", "help"
   - Rationale: Saves embedding costs and LLM tokens by skipping retrieval for non-content queries

1. **Rule 1 - Complex Keywords**: If query contains complex keywords → Route to llama-3.3-70b-versatile
   - Complex keywords: "why", "how", "explain", "compare", "analyze", "difference", "relationship"
   
2. **Rule 2 - Query Length**: Else if query length > 15 words → Route to llama-3.3-70b-versatile

3. **Rule 3 - Multiple Questions**: Else if multiple question marks (>1) → Route to llama-3.3-70b-versatile

4. **Rule 4 - Comparison Words**: Else if contains comparison words → Route to llama-3.3-70b-versatile
   - Comparison words: "versus", "vs", "better", "worse", "compared to"

5. **Rule 5 - Default**: Else → Route to llama-3.1-8b-instant

**Examples**:
- "Hello!" → Simple + skip retrieval (OOD filter)
- "What is the Pro plan price?" → Simple (no complex keywords, < 15 words, single question)
- "How do I configure custom workflows?" → Complex (contains "how")
- "Compare Enterprise vs Pro features" → Complex (contains "compare" and "vs")
- "List keyboard shortcuts" → Simple (no triggers, straightforward)

**Rationale**:
- Deterministic: Same query always gets same classification (clear decision tree)
- Cost-effective: Routes simple queries to faster, cheaper model; skips retrieval for greetings
- Quality-focused: Routes complex queries to more capable model
- Transparent: Each rule is explicit and easy to defend in written answers

**Known Limitation (Stateless Router)**:
- The router only looks at the current query, not conversation history
- Example failure: "Tell me about complex API integration" (→ 70B) followed by "How do I do it?" (→ 8B, loses context)
- This is a genuine limitation to discuss in Q4 (What Is Broken)

### 6. LLM Client

**Responsibility**: Interface with Groq API for text generation.

**Interface**:
```python
class LLMClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 500
    ) -> LLMResponse:
        """
        Generate response using Groq API.
        
        Args:
            model: Model name (llama-3.1-8b-instant or llama-3.3-70b-versatile)
            prompt: Complete prompt with context and query
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse with text, token counts, and latency
        """
        pass

class LLMResponse:
    text: str
    tokens_input: int
    tokens_output: int
    latency_ms: int
    model_used: str
```

**Prompt Template**:
```
You are a helpful customer support assistant for ClearPath, a project management tool.

Context from documentation:
{retrieved_chunks}

User question: {query}

Instructions:
- Answer based on the provided context
- If the context doesn't contain relevant information, say so clearly
- Be concise and helpful
- Cite specific features or details from the documentation when applicable

Answer:
```

**Implementation Notes**:
- Use Groq Python SDK or direct HTTP requests
- Handle rate limits and API errors gracefully
- Track token usage for cost monitoring
- Measure latency for performance monitoring

### 7. Output Evaluator

**Responsibility**: Analyze generated responses and flag quality issues.

**Interface**:
```python
class OutputEvaluator:
    def evaluate(
        self,
        response: str,
        chunks_retrieved: int,
        sources: List[ScoredChunk]
    ) -> List[str]:
        """
        Evaluate response quality and return flags.
        
        Args:
            response: Generated LLM response
            chunks_retrieved: Number of chunks retrieved
            sources: Retrieved chunks with metadata
            
        Returns:
            List of flag strings (empty if no issues)
        """
        pass
```

**Quality Checks**:

1. **No-Context Detection** (`"no_context"`):
   - Condition: `chunks_retrieved == 0` AND response is not a refusal
   - Indicates: LLM hallucinated an answer without documentation support

2. **Refusal Detection** (`"refusal"`):
   - Condition: Response contains refusal phrases
   - Phrases: "I don't have", "not mentioned", "cannot find", "don't know", "no information"
   - Indicates: LLM appropriately declined to answer

3. **Groundedness Check** (`"unverified_feature"`) - Proper Noun Verification:
   - Condition: LLM mentions specific features, integrations, or proper nouns not present in retrieved chunks
   - Logic: Extract proper nouns (capitalized terms, integration names) from response using regex/NER
   - Compare extracted proper nouns against proper nouns in retrieved chunks
   - Example: If LLM mentions "Slack integration" but no chunks contain "Slack", flag as unverified_feature
   - Rationale: Llama models are "too helpful" and hallucinate features based on general SaaS knowledge
   - This catches when LLM invents integrations or features ClearPath doesn't actually have
   - This is a high-value domain-specific check for Q2

4. **Pricing Uncertainty** (`"pricing_uncertainty"`) - Custom Domain Check:
   - Condition: Query mentions pricing AND (multiple conflicting sources OR vague language detected)
   - Vague language: "may", "might", "approximately", "around", "varies"
   - Rationale: Pricing information is critical and must be accurate; conflicting sources or hedging language indicates uncertainty
   - Example: If query asks about "Pro plan price" and retrieved chunks come from both "14_Pricing_Sheet_2024.pdf" and "15_Enterprise_Plan_Details.pdf" with different values

**Implementation Notes**:
- Use simple string matching for refusal detection
- Consider response length (very short responses may indicate refusal)
- Log all evaluator decisions for analysis

### 8. Conversation Manager

**Responsibility**: Maintain conversation context across multiple turns.

**Interface**:
```python
class ConversationManager:
    def get_or_create_conversation(self, conversation_id: Optional[str]) -> Conversation:
        """Get existing conversation or create new one."""
        pass
    
    def add_turn(self, conversation_id: str, query: str, response: str) -> None:
        """Add query-response pair to conversation history."""
        pass
    
    def get_context(self, conversation_id: str, max_turns: int = 3) -> str:
        """Get formatted conversation history for prompt."""
        pass

class Conversation:
    conversation_id: str
    turns: List[Turn]
    created_at: datetime
    
class Turn:
    query: str
    response: str
    timestamp: datetime
```

**Context Window Strategy**:
- Include last 3 turns in prompt (configurable)
- Format: "Previous Q: ... A: ..." for each turn
- Truncate old turns to manage token budget
- Store full history for potential future use

### 9. API Gateway

**Responsibility**: Handle HTTP requests and orchestrate all components.

**Interface**:
```python
@app.post("/query")
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Main query endpoint.
    
    Request body:
        {
            "question": "What is the Pro plan price?",
            "conversation_id": "optional-conv-id"
        }
    
    Response:
        {
            "answer": "...",
            "metadata": {...},
            "sources": [...],
            "conversation_id": "conv_abc123"
        }
    """
    pass

class QueryRequest:
    question: str
    conversation_id: Optional[str] = None

class QueryResponse:
    answer: str
    metadata: ResponseMetadata
    sources: List[Source]
    conversation_id: str

class ResponseMetadata:
    model_used: str
    classification: str
    tokens: TokenUsage
    latency_ms: int
    chunks_retrieved: int
    evaluator_flags: List[str]

class TokenUsage:
    input: int
    output: int

class Source:
    document: str
    page: int
    relevance_score: float
```

**Request Processing Flow**:
1. Validate request (check required fields)
2. Get or create conversation
3. Retrieve relevant chunks
4. Classify query and select model
5. Build prompt with context and conversation history
6. Generate response via LLM
7. Evaluate response quality
8. Log routing decision
9. Format and return response

### 10. Logging System

**Responsibility**: Record routing decisions and system events.

**Interface**:
```python
class RoutingLogger:
    def log_routing_decision(
        self,
        query: str,
        classification: str,
        model_used: str,
        tokens_input: int,
        tokens_output: int,
        latency_ms: int
    ) -> None:
        """Log routing decision with full metadata."""
        pass
```

**Log Format** (JSON Lines):
```json
{
  "timestamp": "2024-01-15T10:30:45Z",
  "query": "What is the Pro plan price?",
  "classification": "simple",
  "model_used": "llama-3.1-8b-instant",
  "rule_triggered": "default",
  "complexity_score": {
    "word_count": 6,
    "complex_keyword_count": 0,
    "question_mark_count": 1,
    "comparison_word_count": 0
  },
  "tokens_input": 234,
  "tokens_output": 45,
  "system_prompt_tokens": 150,
  "context_tokens": 84,
  "latency_ms": 342,
  "chunks_retrieved": 2,
  "evaluator_flags": []
}
```

**Storage**:
- Write to file: `logs/routing_decisions.jsonl`
- Rotate logs daily
- Consider structured logging library (e.g., structlog)

## Data Models

### Document Representation

```python
@dataclass
class Document:
    """Represents a loaded PDF document."""
    filename: str
    pages: List[Page]
    total_pages: int
    
@dataclass
class Page:
    """Represents a single page from a document."""
    page_number: int
    text: str
    word_count: int
```

### Chunk Representation

```python
@dataclass
class Chunk:
    """Represents a document chunk for retrieval."""
    chunk_id: str  # Format: "{filename}_{page}_{chunk_index}"
    text: str
    document_name: str
    page_number: int
    embedding: Optional[np.ndarray] = None
    token_count: int = 0
    
@dataclass
class ScoredChunk:
    """Chunk with relevance score from retrieval."""
    chunk: Chunk
    relevance_score: float  # 0.0 to 1.0
```

### Query Processing

```python
@dataclass
class QueryContext:
    """Complete context for processing a query."""
    query: str
    conversation_id: str
    conversation_history: List[Turn]
    retrieved_chunks: List[ScoredChunk]
    classification: Classification
    
@dataclass
class Classification:
    """Query classification result."""
    category: str  # "simple" or "complex"
    model_name: str
    reasoning: str
    confidence_score: float
```

### Response Models

```python
@dataclass
class GeneratedResponse:
    """Raw LLM response with metadata."""
    text: str
    model_used: str
    tokens_input: int
    tokens_output: int
    latency_ms: int
    
@dataclass
class EvaluatedResponse:
    """Response after quality evaluation."""
    generated_response: GeneratedResponse
    evaluator_flags: List[str]
    confidence_level: str  # "high", "medium", "low"
```

### API Models

```python
class QueryRequest(BaseModel):
    """POST /query request body."""
    question: str = Field(..., min_length=1, max_length=1000)
    conversation_id: Optional[str] = None

class QueryResponse(BaseModel):
    """POST /query response body."""
    answer: str
    metadata: ResponseMetadata
    sources: List[Source]
    conversation_id: str

class ResponseMetadata(BaseModel):
    """Metadata about query processing."""
    model_used: str
    classification: str
    tokens: TokenUsage
    latency_ms: int
    chunks_retrieved: int
    evaluator_flags: List[str]

class TokenUsage(BaseModel):
    """Token usage breakdown."""
    input: int
    output: int
    system_prompt_tokens: int  # Fixed cost per query
    context_tokens: int  # Variable cost from retrieved chunks

class Source(BaseModel):
    """Source document reference."""
    document: str
    page: Optional[int] = None
    relevance_score: Optional[float] = None
```

## Correctness Properties


*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Reflection

After analyzing all acceptance criteria, several redundancies were identified:

- **Requirements 6.1 and 6.2** are identical (no-context detection) - consolidated into one property
- **Requirements 10.2-10.7** all test metadata field presence - combined into one comprehensive schema validation property
- **Requirements 3.2, 3.3, 3.4** all test source field presence - combined into one comprehensive property
- **Requirements 5.1-5.6** all test log field presence - combined into one comprehensive logging property
- **Requirements 4.3 and 4.4** test model routing for simple/complex - can be combined into one bidirectional property
- **Requirements 7.3, 6.3, 8.3** all test flag inclusion - combined into one general flag inclusion property
- **Requirements 7.4, 6.4, 8.4** all test answer still returned - combined into one property about non-blocking flags

The following properties represent the unique, non-redundant correctness guarantees. Note that we have 21 properties total (not 20) because the groundedness check (Property 11) is an additional evaluator property beyond the original three minimum flags.

### Core Retrieval Properties

**Property 1: Chunk Metadata Preservation**
*For any* document chunk created from a source document, the chunk must contain the original document filename and page number from which it was extracted.
**Validates: Requirements 1.4**

**Property 2: Relevance Score Bounds**
*For any* retrieved document chunk, the relevance score must be a value between 0.0 and 1.0 inclusive.
**Validates: Requirements 2.3**

**Property 3: Retrieval Result Ordering**
*For any* retrieval operation that returns multiple chunks, the chunks must be ordered by relevance score in descending order (highest score first).
**Validates: Requirements 2.4**

**Property 4: Selective Retrieval**
*For any* query, the number of chunks passed to the LLM must be strictly less than the total number of chunks in the document corpus.
**Validates: Requirements 2.2**

**Property 5: Source Attribution Completeness**
*For any* source entry in the response, it must contain a document filename, and if the chunk has page information, it must include the page number and relevance score.
**Validates: Requirements 3.2, 3.3, 3.4**

### Routing Properties

**Property 6: Classification Determinism**
*For any* query string, classifying it multiple times must always produce the same classification result (simple or complex).
**Validates: Requirements 4.5**

**Property 7: Classification Validity**
*For any* query, the classification result must be exactly one of the two valid values: "simple" or "complex".
**Validates: Requirements 4.1**

**Property 8: Model Selection Consistency**
*For any* query, if classified as "simple" then the selected model must be "llama-3.1-8b-instant", and if classified as "complex" then the selected model must be "llama-3.3-70b-versatile".
**Validates: Requirements 4.3, 4.4**

### Logging Properties

**Property 9: Routing Decision Log Completeness**
*For any* processed query, the routing log entry must contain all required fields: query text, classification, model_used, tokens_input, tokens_output, and latency_ms.
**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**

### Evaluator Properties

**Property 10: No-Context Flag Detection**
*For any* response where chunks_retrieved equals zero and the response text does not contain refusal phrases, the evaluator_flags array must contain "no_context".
**Validates: Requirements 6.1, 6.2, 6.3**

**Property 11: Groundedness Flag Detection**
*For any* response text that mentions specific proper nouns (features, integrations, product names) not present in the retrieved chunks, the evaluator_flags array must contain "unverified_feature".
**Validates: Requirements 7.1, 7.3**

**Property 12: Refusal Flag Detection**
*For any* response text containing refusal phrases ("I don't have", "not mentioned", "cannot find", "don't know", "no information"), the evaluator_flags array must contain "refusal".
**Validates: Requirements 7.1, 7.3**

**Property 13: Domain-Specific Flag Detection**
*For any* response about pricing that contains hedging language ("may", "might", "approximately", "around", "varies") or has multiple conflicting sources, the evaluator_flags array must contain "pricing_uncertainty".
**Validates: Requirements 8.2, 8.3**

**Property 14: Non-Blocking Flags**
*For any* response with evaluator flags raised, the answer field must still be populated with the generated text (flags are informational, not blocking).
**Validates: Requirements 6.4, 7.4, 8.4**

### API Contract Properties

**Property 14: Conversation ID Generation**
*For any* request without a conversation_id field, the response must contain a newly generated unique conversation_id.
**Validates: Requirements 9.4**

**Property 15: Response Schema Completeness**
*For any* successful API response, it must contain all required top-level fields: answer (string), metadata (object), sources (array), and conversation_id (string).
**Validates: Requirements 9.5**

**Property 16: Metadata Schema Completeness**
*For any* response metadata object, it must contain all required fields: model_used, classification, tokens (with input and output), latency_ms, chunks_retrieved, and evaluator_flags (array).
**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7**

### Conversation Memory Properties

**Property 17: Conversation Context Retrieval**
*For any* request with a valid existing conversation_id, the system must retrieve the previous conversation history for that conversation.
**Validates: Requirements 15.1**

**Property 18: Conversation History Persistence**
*For any* conversation turn added to storage, retrieving that conversation must include the added turn in the history.
**Validates: Requirements 15.5**

**Property 19: Conversation ID Uniqueness**
*For any* two new conversations created, their generated conversation_ids must be different.
**Validates: Requirements 15.4**

### Streaming Properties

**Property 20: Streaming Metadata Completeness**
*For any* streaming response that completes successfully, the final metadata must contain all required fields with accurate values.
**Validates: Requirements 16.4, 16.5**

### Error Handling Properties

**Property 21: API Error Graceful Handling**
*For any* Groq API error condition (rate limit, network failure, invalid request), the system must return an error response with appropriate message rather than crashing.
**Validates: Requirements 13.5**

## Error Handling

### Error Categories

1. **PDF Processing Errors**
   - Corrupted or unreadable PDF files
   - Missing files in clearpath_docs/ directory
   - Encoding issues in text extraction
   - **Handling**: Log error, skip problematic file, continue with remaining documents

2. **Retrieval Errors**
   - Empty query string
   - Vector store initialization failure
   - Embedding model unavailable
   - **Handling**: Return empty chunks list, log error, allow system to continue

3. **Groq API Errors**
   - Rate limit exceeded (429)
   - Authentication failure (401)
   - Network timeout
   - Invalid model name
   - **Handling**: Return error response to user with retry suggestion, log error with full context

4. **Validation Errors**
   - Missing required fields in request
   - Invalid conversation_id format
   - Query exceeds maximum length
   - **Handling**: Return 400 Bad Request with specific error message

5. **Conversation Storage Errors**
   - Conversation not found
   - Storage write failure
   - **Handling**: Log error, continue without conversation context (graceful degradation)

### Error Response Format

```json
{
  "error": {
    "code": "GROQ_API_ERROR",
    "message": "Rate limit exceeded. Please try again in 60 seconds.",
    "details": {
      "retry_after": 60,
      "request_id": "req_abc123"
    }
  }
}
```

### Error Logging

All errors must be logged with:
- Timestamp
- Error type/code
- Full error message
- Stack trace (for unexpected errors)
- Request context (query, conversation_id)
- User-facing error message

## Testing Strategy

### Dual Testing Approach

The system requires both unit tests and property-based tests for comprehensive coverage:

- **Unit tests**: Verify specific examples, edge cases, and error conditions
- **Property tests**: Verify universal properties across all inputs
- Both approaches are complementary and necessary

### Unit Testing Focus

Unit tests should focus on:
- Specific examples that demonstrate correct behavior (e.g., loading a specific PDF)
- Integration points between components (e.g., API endpoint to retrieval pipeline)
- Edge cases (e.g., empty retrieval results, missing conversation_id)
- Error conditions (e.g., API failures, invalid inputs)

Avoid writing too many unit tests for scenarios that property tests can cover through randomization.

### Property-Based Testing Configuration

**Library Selection**: Use Hypothesis (Python) or fast-check (JavaScript/TypeScript)

**Test Configuration**:
- Minimum 100 iterations per property test
- Each test must reference its design document property
- Tag format: `# Feature: clearpath-rag-chatbot, Property {number}: {property_text}`

**Property Test Implementation**:
- Each correctness property must be implemented by a SINGLE property-based test
- Tests should generate random inputs within valid domains
- Tests should verify the property holds for all generated inputs

### Test Coverage by Component

**RAG Pipeline**:
- Unit: Test loading specific PDFs, chunking specific documents
- Property: Chunk metadata preservation (Property 1), relevance score bounds (Property 2)

**Model Router**:
- Unit: Test specific query classifications (e.g., "What is pricing?" → simple)
- Property: Classification determinism (Property 6), model selection consistency (Property 8)

**Output Evaluator**:
- Unit: Test specific refusal phrases trigger flags
- Property: No-context detection (Property 10), refusal detection (Property 11)

**API Endpoint**:
- Unit: Test specific request/response examples
- Property: Response schema completeness (Property 15), metadata completeness (Property 16)

**Conversation Manager**:
- Unit: Test specific conversation flows
- Property: Conversation ID uniqueness (Property 19), history persistence (Property 18)

### Integration Testing

**End-to-End Tests**:
- Submit query → verify complete response with all metadata
- Multi-turn conversation → verify context maintained
- Streaming response → verify progressive delivery and final metadata

**Performance Tests**:
- Measure latency for simple vs complex queries
- Verify retrieval time scales reasonably with corpus size
- Test concurrent request handling

### Evaluation Harness

**Test Query Categories**:
1. Simple factual questions (e.g., "What is ClearPath?")
2. Complex analytical questions (e.g., "Compare Pro and Enterprise plans")
3. Multi-turn conversations (e.g., "What is pricing?" → "What about Enterprise?")
4. Edge cases (e.g., queries with no relevant docs, ambiguous questions)

**Metrics to Track**:
- Routing accuracy (% correctly classified as simple/complex)
- Retrieval precision (% of retrieved chunks actually relevant)
- Response quality (manual evaluation of answer correctness)
- Latency distribution (p50, p95, p99)
- Token usage (average per query type)

**Test Harness Output**:
```
=== ClearPath RAG Chatbot Evaluation ===
Total Queries: 50
Routing Accuracy: 92% (46/50)
Average Latency: 487ms
Token Usage:
  - Simple queries: avg 234 input, 45 output
  - Complex queries: avg 512 input, 128 output
Evaluator Flags:
  - no_context: 3 occurrences
  - refusal: 5 occurrences
  - pricing_uncertainty: 2 occurrences
```

### Testing Written Answers

The four written questions (Q1-Q4) should be answered based on actual system behavior:
- Q1: Document actual routing rules and provide real misclassification examples from testing
- Q2: Identify real retrieval failures from test runs or construct realistic scenarios
- Q3: Calculate token usage from actual test data or realistic estimates
- Q4: Identify genuine system flaws discovered during development

## Implementation Notes

### Initialization Sequence

1. Load environment variables (GROQ_API_KEY, PORT)
2. Initialize embedding model
3. Load and chunk all PDFs from clearpath_docs/
4. Build vector store index
5. Initialize conversation storage
6. Start API server

### Performance Considerations

**Chunking Strategy**:
- Pre-process all documents at startup (one-time cost)
- Cache embeddings to avoid recomputation
- Consider async processing for large document sets

**Retrieval Optimization**:
- Use approximate nearest neighbor search (FAISS) for speed
- Limit top_k to 5-10 chunks to control context size
- Apply relevance threshold to filter low-quality matches

**Token Budget Management**:
- Monitor prompt size to stay within model limits
- Truncate conversation history if needed
- Prioritize recent turns over older ones

**Caching Opportunities**:
- Cache document embeddings (persist to disk)
- Cache frequent query embeddings (in-memory LRU cache)
- Consider caching LLM responses for identical queries (optional)

### Deployment Considerations

**Environment Variables**:
```bash
GROQ_API_KEY=gsk_...
PORT=8000
LOG_LEVEL=INFO
EMBEDDING_MODEL=all-mpnet-base-v2
MAX_CHUNKS=5
CHUNK_SIZE=300
CHUNK_OVERLAP=50
```

**Resource Requirements**:
- Memory: ~2GB for embeddings and vector store
- Disk: ~100MB for cached embeddings
- CPU: Minimal (embedding inference is fast)
- Network: Depends on Groq API latency

**Scaling Considerations**:
- Current design handles single-server deployment
- For high traffic: Add Redis for conversation storage
- For large document sets: Use persistent vector database (Pinecone, Weaviate)
- For multiple servers: Share vector store and conversation storage

### Security Considerations

**API Key Protection**:
- Never commit API keys to version control
- Use environment variables or secret management
- Rotate keys periodically

**Input Validation**:
- Sanitize user queries (max length, character restrictions)
- Validate conversation_id format
- Rate limit requests per IP/user

**Output Safety**:
- Monitor for prompt injection attempts
- Log suspicious queries
- Consider content filtering for generated responses

### Monitoring and Observability

**Metrics to Track**:
- Request rate (queries per minute)
- Latency percentiles (p50, p95, p99)
- Error rate by type
- Token usage and cost
- Model routing distribution (% simple vs complex)
- Evaluator flag frequency

**Logging Strategy**:
- Structured JSON logs for machine parsing
- Include request IDs for tracing
- Log all routing decisions
- Log all evaluator flags
- Separate log levels (DEBUG, INFO, WARNING, ERROR)

**Alerting**:
- High error rate (>5%)
- High latency (p95 > 2s)
- Groq API failures
- Unusual evaluator flag frequency
