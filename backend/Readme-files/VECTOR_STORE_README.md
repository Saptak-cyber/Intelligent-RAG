# VectorStore Implementation

This document describes the VectorStore implementation using Supabase pgvector for the ClearPath RAG Chatbot.

## Overview

The VectorStore class provides a high-level interface for storing document chunk embeddings and performing similarity search using Supabase's pgvector extension. It handles:

- Batch embedding generation to stay under API rate limits
- Storing chunks with metadata in PostgreSQL
- Vector similarity search using cosine distance
- Relevance score normalization to [0, 1] range

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    VectorStore                          │
│                                                         │
│  ┌──────────────────┐      ┌──────────────────┐       │
│  │ EmbeddingModel   │      │ Supabase Client  │       │
│  │ (HuggingFace)    │      │ (PostgreSQL)     │       │
│  └────────┬─────────┘      └────────┬─────────┘       │
│           │                         │                  │
│           │ embed_batch()           │ upsert()         │
│           ▼                         ▼                  │
│  ┌──────────────────────────────────────────────┐     │
│  │         add_chunks()                         │     │
│  │  1. Extract texts from chunks                │     │
│  │  2. Generate embeddings in batch             │     │
│  │  3. Store in Supabase with metadata          │     │
│  └──────────────────────────────────────────────┘     │
│                                                         │
│  ┌──────────────────────────────────────────────┐     │
│  │         search()                             │     │
│  │  1. Call match_chunks RPC function           │     │
│  │  2. Get results with cosine similarity       │     │
│  │  3. Normalize scores to [0, 1]               │     │
│  │  4. Return ScoredChunk objects               │     │
│  └──────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## Database Schema

### Table: document_chunks

```sql
CREATE TABLE document_chunks (
    id BIGSERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    text TEXT NOT NULL,
    document_name TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    token_count INTEGER DEFAULT 0,
    context_header TEXT,
    embedding vector(768),  -- all-mpnet-base-v2 embeddings
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Indexes

- `idx_chunk_id`: Fast lookups by chunk_id
- `idx_document_name`: Fast filtering by document
- `idx_embedding_hnsw`: HNSW index for fast vector similarity search

### RPC Function: match_chunks

```sql
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(768),
    match_threshold float DEFAULT 0.0,
    match_count int DEFAULT 5
)
RETURNS TABLE (
    chunk_id text,
    text text,
    document_name text,
    page_number int,
    token_count int,
    context_header text,
    similarity float
)
```

The function:
1. Computes cosine similarity: `1 - (embedding <=> query_embedding)`
2. Filters by threshold
3. Orders by similarity (descending)
4. Limits to match_count results

## Usage

### Initialization

```python
from services.embedding_model import EmbeddingModel
from services.vector_store import VectorStore

# Initialize embedding model
embedding_model = EmbeddingModel(api_key="your_hf_api_key")

# Initialize vector store
vector_store = VectorStore(
    embedding_model=embedding_model,
    supabase_url="https://your-project.supabase.co",
    supabase_key="your_supabase_key"
)
```

### Adding Chunks

```python
from models.chunk import Chunk

chunks = [
    Chunk(
        chunk_id="doc1_1_0",
        text="ClearPath is a project management tool...",
        document_name="doc1.pdf",
        page_number=1,
        token_count=30,
        context_header="[Context: Introduction]"
    ),
    # ... more chunks
]

# Add chunks (embeddings generated automatically in batch)
vector_store.add_chunks(chunks)
```

### Searching

```python
# Generate query embedding
query = "What is the pricing?"
query_embedding = embedding_model.embed_text(query)

# Search for similar chunks
results = vector_store.search(query_embedding, top_k=5)

for scored_chunk in results:
    print(f"Score: {scored_chunk.relevance_score:.4f}")
    print(f"Text: {scored_chunk.chunk.text}")
    print(f"Document: {scored_chunk.chunk.document_name}")
    print(f"Page: {scored_chunk.chunk.page_number}")
```

### Utility Methods

```python
# Get total number of chunks
count = vector_store.count()

# Clear all chunks (useful for testing)
vector_store.clear()
```

## Key Features

### 1. Batch Embedding

The `add_chunks()` method uses `embed_batch()` to generate embeddings for multiple chunks in a single API call. This:
- Reduces API calls and stays under rate limits
- Improves performance for large document sets
- Handles HuggingFace API cold starts with retry logic

### 2. Relevance Score Normalization

The `search()` method normalizes similarity scores to [0, 1] range:
- Cosine similarity from pgvector is already in [0, 1] (using `1 - distance`)
- Additional clamping ensures scores never exceed bounds
- Consistent scoring across different embedding models

### 3. Error Handling

All methods include comprehensive error handling:
- Input validation (empty lists, invalid parameters)
- API failures (embedding generation, database operations)
- Graceful degradation with informative error messages

### 4. Metadata Preservation

All chunk metadata is preserved:
- `chunk_id`: Unique identifier
- `document_name`: Source document
- `page_number`: Page in source document
- `token_count`: Number of tokens
- `context_header`: Hierarchical header context

## Performance Considerations

### Embedding Generation

- **Batch size**: Process chunks in batches to balance API limits and performance
- **Cold start**: HuggingFace free tier models take 15-20s to load on first query
- **Retry strategy**: Exponential backoff with 5 retries for 503 errors

### Vector Search

- **HNSW index**: Provides fast approximate nearest neighbor search
- **Top-k**: Limit results to 5-10 chunks to control context size
- **Threshold**: Filter low-quality matches (score < 0.3)

### Database Operations

- **Upsert**: Use upsert to handle duplicate chunk_ids gracefully
- **Connection pooling**: Supabase client handles connection pooling automatically
- **Batch inserts**: Insert multiple chunks in a single transaction

## Testing

### Unit Tests

Run unit tests with mocked dependencies:

```bash
pytest tests/test_vector_store.py -v
```

Tests cover:
- Initialization and validation
- Adding chunks (success and failure cases)
- Searching (various scenarios)
- Score normalization
- Error handling
- Utility methods (clear, count)

### Integration Tests

Run integration tests with real Supabase instance:

```bash
python backend/demo_vector_store.py
```

This demonstrates:
- Adding sample chunks
- Searching with different queries
- Relevance score ranking

## Setup Instructions

### 1. Enable pgvector Extension

In Supabase SQL Editor:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Run Migrations

Execute the migration file in Supabase SQL Editor:

```bash
backend/migrations/001_create_chunks_table.sql
```

### 3. Configure Environment Variables

Add to `.env`:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
HUGGINGFACE_API_KEY=your_hf_api_key
```

### 4. Verify Setup

```python
from services.vector_store import VectorStore
from services.embedding_model import EmbeddingModel

embedding_model = EmbeddingModel()
vector_store = VectorStore(embedding_model=embedding_model)

# Should return 0 for empty store
count = vector_store.count()
print(f"Chunks in store: {count}")
```

## Troubleshooting

### Issue: "Function match_chunks does not exist"

**Solution**: Run the migration file to create the RPC function.

### Issue: "Extension vector does not exist"

**Solution**: Enable pgvector extension in Supabase:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Issue: "Embedding API timeout"

**Solution**: The HuggingFace free tier has cold starts. The retry logic handles this automatically. Wait 15-20s for the model to load.

### Issue: "Rate limit exceeded"

**Solution**: Use batch embedding (`embed_batch()`) instead of individual calls. The VectorStore already does this.

## Future Enhancements

1. **Caching**: Cache embeddings to disk to avoid recomputation
2. **Incremental updates**: Support updating individual chunks without reindexing
3. **Filtering**: Add support for filtering by document or metadata
4. **Hybrid search**: Combine vector search with keyword search
5. **Compression**: Use quantization to reduce embedding storage size

## References

- [Supabase pgvector Documentation](https://supabase.com/docs/guides/ai/vector-columns)
- [HuggingFace Inference API](https://huggingface.co/docs/api-inference/index)
- [all-mpnet-base-v2 Model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320)
