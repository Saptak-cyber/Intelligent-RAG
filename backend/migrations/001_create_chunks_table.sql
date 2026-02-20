-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create document_chunks table
CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    text TEXT NOT NULL,
    document_name TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    token_count INTEGER DEFAULT 0,
    context_header TEXT,
    embedding vector(768),  -- all-mpnet-base-v2 produces 768-dimensional embeddings
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on chunk_id for fast lookups
CREATE INDEX IF NOT EXISTS idx_chunk_id ON document_chunks(chunk_id);

-- Create index on document_name for filtering
CREATE INDEX IF NOT EXISTS idx_document_name ON document_chunks(document_name);

-- Create vector similarity search index using HNSW (Hierarchical Navigable Small World)
-- This significantly speeds up similarity searches
CREATE INDEX IF NOT EXISTS idx_embedding_hnsw ON document_chunks 
USING hnsw (embedding vector_cosine_ops);

-- Create RPC function for similarity search
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
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        document_chunks.chunk_id,
        document_chunks.text,
        document_chunks.document_name,
        document_chunks.page_number,
        document_chunks.token_count,
        document_chunks.context_header,
        1 - (document_chunks.embedding <=> query_embedding) AS similarity
    FROM document_chunks
    WHERE 1 - (document_chunks.embedding <=> query_embedding) > match_threshold
    ORDER BY document_chunks.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Grant permissions (adjust as needed for your Supabase setup)
-- These are typically handled by Supabase's RLS policies
-- ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
