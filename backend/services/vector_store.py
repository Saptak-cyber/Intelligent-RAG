"""Vector store implementation using Supabase pgvector."""
import logging
from typing import List, Optional
from supabase import create_client, Client
from models.chunk import Chunk, ScoredChunk
from services.embedding_model import EmbeddingModel
from config import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger(__name__)


class VectorStore:
    """Store chunk embeddings and enable similarity search using Supabase pgvector."""
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        supabase_url: str = SUPABASE_URL,
        supabase_key: str = SUPABASE_KEY,
        table_name: str = "document_chunks"
    ):
        """
        Initialize the vector store with Supabase client.
        
        Args:
            embedding_model: EmbeddingModel instance for generating embeddings
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
            table_name: Name of the table to store chunks
            
        Raises:
            ValueError: If Supabase credentials are missing
        """
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")
        
        self.embedding_model = embedding_model
        self.table_name = table_name
        
        # Initialize Supabase client
        self.client: Client = create_client(supabase_url, supabase_key)
        
        logger.info(f"Initialized VectorStore with table: {table_name}")
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks with embeddings to the vector store.
        
        Uses batch embedding to send multiple chunks in one API call
        to stay under rate limits.
        
        Args:
            chunks: List of Chunk objects to store
            
        Raises:
            ValueError: If chunks list is empty
            RuntimeError: If database operation fails
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty")
        
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        
        try:
            # Extract texts for batch embedding
            texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings in batch
            logger.debug(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embedding_model.embed_batch(texts)
            
            # Prepare records for insertion
            records = []
            for chunk, embedding in zip(chunks, embeddings):
                record = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "token_count": chunk.token_count,
                    "context_header": chunk.context_header,
                    "embedding": embedding
                }
                records.append(record)
            
            # Insert records into Supabase
            # Use upsert to handle duplicate chunk_ids
            response = self.client.table(self.table_name).upsert(records).execute()
            
            logger.info(f"Successfully added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            error_msg = f"Failed to add chunks to vector store: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[ScoredChunk]:
        """
        Find most similar chunks to query using cosine similarity.
        
        Args:
            query_embedding: Embedding vector for user query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of ScoredChunk objects with relevance scores normalized to [0, 1]
            
        Raises:
            ValueError: If query_embedding is empty or top_k is invalid
            RuntimeError: If database operation fails
        """
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")
        
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        try:
            # Use Supabase RPC to call pgvector similarity search
            # The RPC function should be created in Supabase with:
            # CREATE OR REPLACE FUNCTION match_chunks(
            #   query_embedding vector(768),
            #   match_threshold float,
            #   match_count int
            # )
            # RETURNS TABLE (
            #   chunk_id text,
            #   text text,
            #   document_name text,
            #   page_number int,
            #   token_count int,
            #   context_header text,
            #   similarity float
            # )
            # LANGUAGE plpgsql
            # AS $$
            # BEGIN
            #   RETURN QUERY
            #   SELECT
            #     document_chunks.chunk_id,
            #     document_chunks.text,
            #     document_chunks.document_name,
            #     document_chunks.page_number,
            #     document_chunks.token_count,
            #     document_chunks.context_header,
            #     1 - (document_chunks.embedding <=> query_embedding) AS similarity
            #   FROM document_chunks
            #   WHERE 1 - (document_chunks.embedding <=> query_embedding) > match_threshold
            #   ORDER BY document_chunks.embedding <=> query_embedding
            #   LIMIT match_count;
            # END;
            # $$;
            
            response = self.client.rpc(
                "match_chunks",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.0,  # We'll filter later
                    "match_count": top_k
                }
            ).execute()
            
            # Parse results into ScoredChunk objects
            scored_chunks = []
            for row in response.data:
                chunk = Chunk(
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    document_name=row["document_name"],
                    page_number=row["page_number"],
                    token_count=row.get("token_count", 0),
                    context_header=row.get("context_header")
                )
                
                # Normalize similarity score to [0, 1] range
                # Cosine similarity is already in [-1, 1], but pgvector distance is [0, 2]
                # The RPC function returns 1 - distance, so it's already in [0, 1]
                similarity = row["similarity"]
                
                # Ensure score is in [0, 1] range
                relevance_score = max(0.0, min(1.0, similarity))
                
                scored_chunks.append(ScoredChunk(
                    chunk=chunk,
                    relevance_score=relevance_score
                ))
            
            logger.debug(f"Found {len(scored_chunks)} chunks for query")
            return scored_chunks
            
        except Exception as e:
            error_msg = f"Failed to search vector store: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def clear(self) -> None:
        """
        Clear all chunks from the vector store.
        
        Useful for testing or reindexing.
        
        Raises:
            RuntimeError: If database operation fails
        """
        try:
            response = self.client.table(self.table_name).delete().neq("chunk_id", "").execute()
            logger.info("Cleared all chunks from vector store")
        except Exception as e:
            error_msg = f"Failed to clear vector store: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def count(self) -> int:
        """
        Get the total number of chunks in the vector store.
        
        Returns:
            Number of chunks stored
            
        Raises:
            RuntimeError: If database operation fails
        """
        try:
            response = self.client.table(self.table_name).select("chunk_id", count="exact").execute()
            return response.count if response.count is not None else 0
        except Exception as e:
            error_msg = f"Failed to count chunks in vector store: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
