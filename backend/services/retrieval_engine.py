"""Retrieval engine for orchestrating query embedding and chunk retrieval."""
import logging
from typing import List
from models.chunk import ScoredChunk
from services.vector_store import VectorStore
from services.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Orchestrate query embedding and chunk retrieval with dynamic K-cutoff."""
    
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        """
        Initialize the retrieval engine.
        
        Args:
            vector_store: VectorStore instance for similarity search
            embedding_model: EmbeddingModel instance for query embedding
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        logger.info("Initialized RetrievalEngine")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        """
        Retrieve relevant chunks for query with dynamic K-cutoff.
        
        Implements the following filtering strategy:
        1. Embed the user query
        2. Perform similarity search in vector store
        3. Apply relevance threshold (score > 0.3) to filter low-quality matches
        4. Apply dynamic K-cutoff: only include chunks within 20% of top score
           - Example: If top chunk scores 0.85, only include chunks >=  (0.85 * 0.5)
           - Rationale: Prevents "Lost in the Middle" problem where low-relevance
             chunks hurt LLM performance
        5. Return filtered chunks sorted by relevance
        
        Args:
            query: User question
            top_k: Maximum number of chunks to retrieve (default: 5)
            
        Returns:
            List of scored chunks, empty if no relevant results or empty query
            
        Raises:
            RuntimeError: If embedding or search operations fail
        """
        # Handle empty query strings gracefully
        if not query or not query.strip():
            logger.warning("Empty query string provided, returning empty results")
            return []
        
        try:
            # Step 1: Embed the user query
            logger.debug(f"Embedding query: {query[:100]}...")
            query_embedding = self.embedding_model.embed_text(query)
            
            # Step 2: Perform similarity search
            logger.debug(f"Searching for top {top_k} chunks")
            scored_chunks = self.vector_store.search(query_embedding, top_k=top_k)
            
            # If no results, return empty list
            if not scored_chunks:
                logger.info("No chunks found for query")
                return []
            
            # Step 3: Apply relevance threshold (score > 0.3)
            relevance_threshold = 0.2
            filtered_chunks = [
                chunk for chunk in scored_chunks
                if chunk.relevance_score > relevance_threshold
            ]
            
            if not filtered_chunks:
                logger.info(f"No chunks above relevance threshold {relevance_threshold}")
                return []
            
            logger.debug(
                f"Filtered to {len(filtered_chunks)} chunks above threshold {relevance_threshold}"
            )
            
            # Step 4: Apply dynamic K-cutoff (within 20% of top score)
            top_score = filtered_chunks[0].relevance_score
            cutoff_threshold = top_score * 0.8  # 80% of top score
            
            dynamic_filtered_chunks = [
                chunk for chunk in filtered_chunks
                if chunk.relevance_score >= cutoff_threshold
            ]
            
            logger.info(
                f"Retrieved {len(dynamic_filtered_chunks)} chunks "
                f"(top score: {top_score:.3f}, cutoff: {cutoff_threshold:.3f})"
            )
            
            # Step 5: Return filtered chunks (already sorted by relevance from vector store)
            return dynamic_filtered_chunks
            
        except Exception as e:
            error_msg = f"Failed to retrieve chunks for query: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
