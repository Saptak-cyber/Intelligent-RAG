"""Unit tests for RetrievalEngine."""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest
from unittest.mock import Mock, MagicMock
from services.retrieval_engine import RetrievalEngine
from models.chunk import Chunk, ScoredChunk


class TestRetrievalEngine:
    """Test suite for RetrievalEngine class."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStore."""
        return Mock()
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock EmbeddingModel."""
        return Mock()
    
    @pytest.fixture
    def retrieval_engine(self, mock_vector_store, mock_embedding_model):
        """Create a RetrievalEngine instance with mocks."""
        return RetrievalEngine(mock_vector_store, mock_embedding_model)
    
    def test_initialization(self, retrieval_engine, mock_vector_store, mock_embedding_model):
        """Test that RetrievalEngine initializes correctly."""
        assert retrieval_engine.vector_store == mock_vector_store
        assert retrieval_engine.embedding_model == mock_embedding_model
    
    def test_retrieve_empty_query(self, retrieval_engine):
        """Test that empty query returns empty list."""
        result = retrieval_engine.retrieve("")
        assert result == []
        
        result = retrieval_engine.retrieve("   ")
        assert result == []
    
    def test_retrieve_no_results(self, retrieval_engine, mock_embedding_model, mock_vector_store):
        """Test retrieval when no chunks are found."""
        mock_embedding_model.embed_text.return_value = [0.1] * 768
        mock_vector_store.search.return_value = []
        
        result = retrieval_engine.retrieve("test query")
        
        assert result == []
        mock_embedding_model.embed_text.assert_called_once_with("test query")
        mock_vector_store.search.assert_called_once()
    
    def test_retrieve_below_relevance_threshold(
        self, retrieval_engine, mock_embedding_model, mock_vector_store
    ):
        """Test that chunks below relevance threshold (0.3) are filtered out."""
        mock_embedding_model.embed_text.return_value = [0.1] * 768
        
        # Create chunks with low relevance scores
        low_score_chunks = [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="doc1_1_0",
                    text="Low relevance chunk",
                    document_name="doc1.pdf",
                    page_number=1
                ),
                relevance_score=0.25
            ),
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="doc1_1_1",
                    text="Another low relevance chunk",
                    document_name="doc1.pdf",
                    page_number=1
                ),
                relevance_score=0.15
            )
        ]
        
        mock_vector_store.search.return_value = low_score_chunks
        
        result = retrieval_engine.retrieve("test query")
        
        # All chunks should be filtered out
        assert result == []
    
    def test_retrieve_with_dynamic_k_cutoff(
        self, retrieval_engine, mock_embedding_model, mock_vector_store
    ):
        """Test dynamic K-cutoff: only chunks within 20% of top score are included."""
        mock_embedding_model.embed_text.return_value = [0.1] * 768
        
        # Create chunks with varying relevance scores
        # Top score: 0.85, cutoff threshold: 0.85 * 0.8 = 0.68
        scored_chunks = [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="doc1_1_0",
                    text="Highly relevant chunk",
                    document_name="doc1.pdf",
                    page_number=1
                ),
                relevance_score=0.85  # Top score
            ),
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="doc1_1_1",
                    text="Also relevant chunk",
                    document_name="doc1.pdf",
                    page_number=1
                ),
                relevance_score=0.75  # Within 20% of top (>= 0.68)
            ),
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="doc1_2_0",
                    text="Moderately relevant chunk",
                    document_name="doc1.pdf",
                    page_number=2
                ),
                relevance_score=0.68  # Exactly at cutoff
            ),
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="doc1_2_1",
                    text="Less relevant chunk",
                    document_name="doc1.pdf",
                    page_number=2
                ),
                relevance_score=0.65  # Below cutoff (< 0.68)
            ),
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="doc1_3_0",
                    text="Low relevance chunk",
                    document_name="doc1.pdf",
                    page_number=3
                ),
                relevance_score=0.40  # Well below cutoff
            )
        ]
        
        mock_vector_store.search.return_value = scored_chunks
        
        result = retrieval_engine.retrieve("test query", top_k=5)
        
        # Should return only the top 3 chunks (scores >= 0.68)
        assert len(result) == 3
        assert result[0].relevance_score == 0.85
        assert result[1].relevance_score == 0.75
        assert result[2].relevance_score == 0.68
    
    def test_retrieve_all_chunks_within_cutoff(
        self, retrieval_engine, mock_embedding_model, mock_vector_store
    ):
        """Test when all chunks are within the dynamic cutoff."""
        mock_embedding_model.embed_text.return_value = [0.1] * 768
        
        # All chunks have similar high scores
        scored_chunks = [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id=f"doc1_1_{i}",
                    text=f"Chunk {i}",
                    document_name="doc1.pdf",
                    page_number=1
                ),
                relevance_score=0.90 - (i * 0.02)  # 0.90, 0.88, 0.86, 0.84, 0.82
            )
            for i in range(5)
        ]
        
        mock_vector_store.search.return_value = scored_chunks
        
        result = retrieval_engine.retrieve("test query", top_k=5)
        
        # All chunks should be included (all >= 0.90 * 0.8 = 0.72)
        assert len(result) == 5
    
    def test_retrieve_sorted_by_relevance(
        self, retrieval_engine, mock_embedding_model, mock_vector_store
    ):
        """Test that results are sorted by relevance score descending."""
        mock_embedding_model.embed_text.return_value = [0.1] * 768
        
        scored_chunks = [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="doc1_1_0",
                    text="Chunk 1",
                    document_name="doc1.pdf",
                    page_number=1
                ),
                relevance_score=0.80
            ),
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="doc1_1_1",
                    text="Chunk 2",
                    document_name="doc1.pdf",
                    page_number=1
                ),
                relevance_score=0.75
            ),
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="doc1_1_2",
                    text="Chunk 3",
                    document_name="doc1.pdf",
                    page_number=1
                ),
                relevance_score=0.70
            )
        ]
        
        mock_vector_store.search.return_value = scored_chunks
        
        result = retrieval_engine.retrieve("test query")
        
        # Verify descending order
        assert len(result) == 3
        assert result[0].relevance_score >= result[1].relevance_score
        assert result[1].relevance_score >= result[2].relevance_score
    
    def test_retrieve_handles_embedding_error(
        self, retrieval_engine, mock_embedding_model, mock_vector_store
    ):
        """Test that embedding errors are properly raised."""
        mock_embedding_model.embed_text.side_effect = RuntimeError("Embedding failed")
        
        with pytest.raises(RuntimeError, match="Failed to retrieve chunks"):
            retrieval_engine.retrieve("test query")
    
    def test_retrieve_handles_search_error(
        self, retrieval_engine, mock_embedding_model, mock_vector_store
    ):
        """Test that search errors are properly raised."""
        mock_embedding_model.embed_text.return_value = [0.1] * 768
        mock_vector_store.search.side_effect = RuntimeError("Search failed")
        
        with pytest.raises(RuntimeError, match="Failed to retrieve chunks"):
            retrieval_engine.retrieve("test query")
    
    def test_retrieve_with_custom_top_k(
        self, retrieval_engine, mock_embedding_model, mock_vector_store
    ):
        """Test retrieval with custom top_k parameter."""
        mock_embedding_model.embed_text.return_value = [0.1] * 768
        
        scored_chunks = [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id=f"doc1_1_{i}",
                    text=f"Chunk {i}",
                    document_name="doc1.pdf",
                    page_number=1
                ),
                relevance_score=0.85 - (i * 0.05)
            )
            for i in range(10)
        ]
        
        mock_vector_store.search.return_value = scored_chunks[:10]
        
        result = retrieval_engine.retrieve("test query", top_k=10)
        
        # Verify that top_k was passed to search
        mock_vector_store.search.assert_called_once()
        call_args = mock_vector_store.search.call_args
        assert call_args[1]['top_k'] == 10
