# Database Migrations

This directory contains SQL migration files for setting up the Supabase database.

## Setup Instructions

1. Log in to your Supabase project dashboard
2. Navigate to the SQL Editor
3. Run the migration files in order:
   - `001_create_chunks_table.sql` - Creates the document_chunks table and match_chunks function
   - `002_create_conversations_tables.sql` - Creates the conversations and turns tables

## What Gets Created

### Tables

- **document_chunks**: Stores document chunks with their embeddings
  - `chunk_id`: Unique identifier for each chunk
  - `text`: The chunk text content
  - `document_name`: Source document filename
  - `page_number`: Page number in source document
  - `token_count`: Number of tokens in the chunk
  - `context_header`: Hierarchical header context
  - `embedding`: 768-dimensional vector embedding (all-mpnet-base-v2)

- **conversations**: Stores conversation metadata
  - `conversation_id`: Unique identifier for each conversation
  - `created_at`: Timestamp when conversation was created

- **turns**: Stores individual turns in conversations
  - `conversation_id`: Foreign key to conversations table
  - `query`: User query text
  - `response`: System response text
  - `timestamp`: Timestamp when turn was created

### Indexes

- `idx_chunk_id`: Fast lookups by chunk_id
- `idx_document_name`: Fast filtering by document
- `idx_embedding_hnsw`: HNSW index for fast vector similarity search
- `idx_conversation_id`: Fast lookups by conversation_id
- `idx_turns_conversation_id`: Fast turn retrieval by conversation
- `idx_turns_timestamp`: Fast ordering of turns by timestamp

### Functions

- **match_chunks**: RPC function for similarity search
  - Parameters:
    - `query_embedding`: The query vector (768 dimensions)
    - `match_threshold`: Minimum similarity score (default: 0.0)
    - `match_count`: Maximum number of results (default: 5)
  - Returns: Table of matching chunks with similarity scores

## Verification

After running the migrations, verify the setup:

```sql
-- Check if pgvector extension is enabled
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Check if tables exist
SELECT * FROM information_schema.tables WHERE table_name IN ('document_chunks', 'conversations', 'turns');

-- Check if function exists
SELECT * FROM pg_proc WHERE proname = 'match_chunks';
```
