"""Unit tests for VectorStore class."""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest
from unittest.mock import Mock, patch, MagicMock
from models.chunk import Chunk, ScoredChunk
from services.vector_store import VectorStore
from services.embedding_model import EmbeddingModel


class TestVectorStore:
    """Test suite for VectorStore."""
    
    @patch('services.vector_store.create_client')
    def test_initialization_success(self, mock_create_client):
        """Test successful initialization with credentials."""
        mock_embedding_model = Mock(spec=EmbeddingModel)
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        
        store = VectorStore(
            embedding_model=mock_embedding_model,
            supabase_url="https://test.supabase.co",
            supabase_key="test_key"
        )
        
        assert store.embedding_model == mock_embedding_model
        assert store.table_name == "document_chunks"
        mock_create_client.assert_called_once_with("https://test.supabase.co", "test_key")
    
    def test_initialization_without_credentials(self):
        """Test initialization fails without Supabase credentials."""
        mock_embedding_model = Mock(spec=EmbeddingModel)
        
        with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_KEY"):
            VectorStore(
                embedding_model=mock_embedding_model,
                supabase_url=None,
                supabase_key="test_key"
            )
        
        with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_KEY"):
            VectorStore(
                embedding_model=mock_embedding_model,
                supabase_url="https://test.supabase.co",
                supabase_key=None
            )
    
    @patch('services.vector_store.create_client')
    def test_add_chunks_empty_list(self, mock_create_client):
        """Test add_chunks raises error for empty list."""
        mock_embedding_model = Mock(spec=EmbeddingModel)
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        
        store = VectorStore(
            embedding_model=mock_embedding_model,
            supabase_url="https://test.supabase.co",
            supabase_key="test_key"
        )
        
        with pytest.raises(ValueError, match="Chunks list cannot be empty"):
            store.add_chunks([])
    
    @patch('services.vector_store.create_client')
    def test_add_chunks_success(self, mock_create_client):
        """Test successful addition of chunks."""
        # Setup mocks
        mock_embedding_model = Mock(spec=EmbeddingModel)
        mock_embedding_model.embed_batch.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_upsert = MagicMock()
        mock_execute = MagicMock()
        
        mock_client.table.return_value = mock_table
        mock_table.upsert.return_value = mock_upsert
        mock_upsert.execute.return_value = mock_execute
        mock_create_client.return_value = mock_client
        
        # Create chunks
        chunks = [
            Chunk(
                chunk_id="doc1_1_0",
                text="Test chunk 1",
                document_name="doc1.pdf",
                page_number=1,
                token_count=10,
                context_header="[Context: Introduction]"
            ),
            Chunk(
                chunk_id="doc1_1_1",
                text="Test chunk 2",
                document_name="doc1.pdf",
                page_number=1,
                token_count=12,
                context_header="[Context: Introduction]"
            )
        ]
        
        # Execute
        store = VectorStore(
            embedding_model=mock_embedding_model,
            supabase_url="https://test.supabase.co",
            supabase_key="test_key"
        )
        store.add_chunks(chunks)
        
        # Verify
        mock_embedding_model.embed_batch.assert_called_once()
        texts_arg = mock_embedding_model.embed_batch.call_args[0][0]
        assert texts_arg == ["Test chunk 1", "Test chunk 2"]
        
        mock_client.table.assert_called_with("document_chunks")
        mock_table.upsert.assert_called_once()
        
        # Check the records passed to upsert
        records = mock_table.upsert.call_args[0][0]
        assert len(records) == 2
        assert records[0]["chunk_id"] == "doc1_1_0"
        assert records[0]["text"] == "Test chunk 1"
        assert records[0]["embedding"] == [0.1, 0.2, 0.3]
        assert records[1]["chunk_id"] == "doc1_1_1"
        assert records[1]["embedding"] == [0.4, 0.5, 0.6]
    
    @patch('services.vector_store.create_client')
    def test_add_chunks_embedding_failure(self, mock_create_client):
        """Test add_chunks handles embedding failure."""
        mock_embedding_model = Mock(spec=EmbeddingModel)
        mock_embedding_model.embed_batch.side_effect = RuntimeError("Embedding API error")
        
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        
        chunks = [
            Chunk(
                chunk_id="doc1_1_0",
                text="Test chunk",
                document_name="doc1.pdf",
                page_number=1
            )
        ]
        
        store = VectorStore(
            embedding_model=mock_embedding_model,
            supabase_url="https://test.supabase.co",
            supabase_key="test_key"
        )
        
        with pytest.raises(RuntimeError, match="Failed to add chunks"):
            store.add_chunks(chunks)
    
    @patch('services.vector_store.create_client')
    def test_add_chunks_database_failure(self, mock_create_client):
        """Test add_chunks handles database failure."""
        mock_embedding_model = Mock(spec=EmbeddingModel)
        mock_embedding_model.embed_batch.return_value = [[0.1, 0.2, 0.3]]
        
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_table.upsert.side_effect = Exception("Database error")
        mock_client.table.return_value = mock_table
        mock_create_client.return_value = mock_client
        
        chunks = [
            Chunk(
                chunk_id="doc1_1_0",
                text="Test chunk",
                document_name="doc1.pdf",
                page_number=1
            )
        ]
        
        store = VectorStore(
            embedding_model=mock_embedding_model,
            supabase_url="https://test.supabase.co",
            supabase_key="test_key"
        )
        
        with pytest.raises(RuntimeError, match="Failed to add chunks"):
            store.add_chunks(chunks)
    
    @patch('services.vector_store.create_client')
    def test_search_empty_embedding(self, mock_create_client):
        """Test search raises error for empty embedding."""
        mock_embedding_model = Mock(spec=EmbeddingModel)
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        
        store = VectorStore(
            embedding_model=mock_embedding_model,
            supabase_url="https://test.supabase.co",
            supabase_key="test_key"
        )
        
        with pytest.raises(ValueError, match="Query embedding cannot be empty"):
            store.search([])
    
    @patch('services.vector_store.create_client')
    def test_search_invalid_top_k(self, mock_create_client):
        """Test search raises error for invalid top_k."""
        mock_embedding_model = Mock(spec=EmbeddingModel)
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        
        store = VectorStore(
            embedding_model=mock_embedding_model,
            supabase_url="https://test.supabase.co",
            supabase_key="test_key"
        )
        
        with pytest.raises(ValueError, match="top_k must be positive"):
            store.search([0.1, 0.2, 0.3], top_k=0)
        
        with pytest.raises(ValueError, match="top_k must be positive"):
            store.search([0.1, 0.2, 0.3], top_k=-1)
    
    @patch('services.vector_store.create_client')
    def test_search_success(self, mock_create_client):
        """Test successful search."""
        mock_embedding_model = Mock(spec=EmbeddingModel)
        
        # Setup mock response
        mock_client = MagicMock()
        mock_rpc = MagicMock()
        mock_execute = MagicMock()
        
        mock_execute.data = [
            {
                "chunk_id": "doc1_1_0",
                "text": "Test chunk 1",
                "document_name": "doc1.pdf",
                "page_number": 1,
                "token_count": 10,
                "context_header": "[Context: Introduction]",
                "similarity": 0.85
            },
            {
                "chunk_id": "doc2_2_0",
                "text": "Test chunk 2",
                "document_name": "doc2.pdf",
                "page_number": 2,
                "token_count": 12,
                "context_header": None,
                "similarity": 0.72
            }
        ]
        
        mock_client.rpc.return_value = mock_rpc
        mock_rpc.execute.return_value = mock_execute
        mock_create_client.return_value = mock_client
        
        # Execute
        store = VectorStore(
            embedding_model=mock_embedding_model,
            supabase_url="https://test.supabase.co",
            supabase_key="test_key"
        )
        
        query_embedding = [0.1, 0.2, 0.3]
        results = store.search(query_embedding, top_k=5)
        
        # Verify
        mock_client.rpc.assert_called_once_with(
            "match_chunks",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.0,
                "match_count": 5
            }
        )
        
        assert len(results) == 2
        assert isinstance(results[0], ScoredChunk)
        assert results[0].chunk.chunk_id == "doc1_1_0"
        assert results[0].chunk.text == "Test chunk 1"
        assert results[0].relevance_score == 0.85
        assert results[1].chunk.chunk_id == "doc2_2_0"
        assert results[1].relevance_score == 0.72
    
    @patch('services.vector_store.create_client')
    def test_search_empty_results(self, mock_create_client):
        """Test search with no results."""
        mock_embedding_model = Mock(spec=EmbeddingModel)
        
        mock_client = MagicMock()
        mock_rpc = MagicMock()
        mock_execute = MagicMock()
        mock_execute.data = []
        
        mock_client.rpc.return_value = mock_rpc
        mock_rpc.execute.return_value = mock_execute
        mock_create_client.return_value = mock_client
        
        store = VectorStore(
            embedding_model=mock_embedding_model,
            supabase_url="https://test.supabase.co",
            supabase_key="test_key"
        )
        
        results = store.search([0.1, 0.2, 0.3], top_k=5)
        
        assert len(results) == 0
        assert results == []
    
    @patch('services.vector_store.create_client')
    def test_search_score_normalization(self, mock_create_client):
        """Test that similarity scores are normalized to [0, 1] range."""
        mock_embedding_model = Mock(spec=EmbeddingModel)
        
        mock_client = MagicMock()
        mock_rpc = MagicMock()
        mock_execute = MagicMock()
        
        # Test edge cases for score normalization
        mock_execute.data = [
            {
                "chunk_id": "doc1_1_0",
                "text": "Test chunk 1",
                "document_name": "doc1.pdf",
                "page_number": 1,
                "token_count": 10,
                "context_header": None,
                "similarity": 1.5  # Above 1.0
            },
            {
                "chunk_id": "doc1_1_1",
                "text": "Test chunk 2",
                "document_name": "doc1.pdf",
                "page_number": 1,
                "token_count": 10,
                "context_header": None,
                "similarity": -0.2  # Below 0.0
            },
            {
                "chunk_id": "doc1_1_2",
                "text": "Test chunk 3",
                "document_name": "doc1.pdf",
                "page_number": 1,
                "token_count": 10,
                "context_header": None,
                "similarity": 0.5  # Normal
            }
        ]
        
        mock_client.rpc.return_value = mock_rpc
        mock_rpc.execute.return_value = mock_execute
        mock_create_client.return_value = mock_client
        
        store = VectorStore(
            embedding_model=mock_embedding_model,
            supabase_url="https://test.supabase.co",
            supabase_key="test_key"
        )
        
        results = store.search([0.1, 0.2, 0.3], top_k=5)
        
        # Verify scores are clamped to [0, 1]
        assert results[0].relevance_score == 1.0  # Clamped from 1.5
        assert results[1].relevance_score == 0.0  # Clamped from -0.2
        assert results[2].relevance_score == 0.5  # Unchanged
    
    @patch('services.vector_store.create_client')
    def test_search_database_failure(self, mock_create_client):
        """Test search handles database failure."""
        mock_embedding_model = Mock(spec=EmbeddingModel)
        
        mock_client = MagicMock()
        mock_client.rpc.side_effect = Exception("Database error")
        mock_create_client.return_value = mock_client
        
        store = VectorStore(
            embedding_model=mock_embedding_model,
            supabase_url="https://test.supabase.co",
            supabase_key="test_key"
        )
        
        with pytest.raises(RuntimeError, match="Failed to search vector store"):
            store.search([0.1, 0.2, 0.3], top_k=5)
    
    @patch('services.vector_store.create_client')
    def test_clear_success(self, mock_create_client):
        """Test successful clearing of vector store."""
        mock_embedding_model = Mock(spec=EmbeddingModel)
        
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_delete = MagicMock()
        mock_neq = MagicMock()
        mock_execute = MagicMock()
        
        mock_client.table.return_value = mock_table
        mock_table.delete.return_value = mock_delete
        mock_delete.neq.return_value = mock_neq
        mock_neq.execute.return_value = mock_execute
        mock_create_client.return_value = mock_client
        
        store = VectorStore(
            embedding_model=mock_embedding_model,
            supabase_url="https://test.supabase.co",
            supabase_key="test_key"
        )
        
        store.clear()
        
        mock_client.table.assert_called_with("document_chunks")
        mock_table.delete.assert_called_once()
        mock_delete.neq.assert_called_with("chunk_id", "")
    
    @patch('services.vector_store.create_client')
    def test_clear_failure(self, mock_create_client):
        """Test clear handles database failure."""
        mock_embedding_model = Mock(spec=EmbeddingModel)
        
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_table.delete.side_effect = Exception("Database error")
        mock_client.table.return_value = mock_table
        mock_create_client.return_value = mock_client
        
        store = VectorStore(
            embedding_model=mock_embedding_model,
            supabase_url="https://test.supabase.co",
            supabase_key="test_key"
        )
        
        with pytest.raises(RuntimeError, match="Failed to clear vector store"):
            store.clear()
    
    @patch('services.vector_store.create_client')
    def test_count_success(self, mock_create_client):
        """Test successful count of chunks."""
        mock_embedding_model = Mock(spec=EmbeddingModel)
        
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_select = MagicMock()
        mock_execute = MagicMock()
        mock_execute.count = 42
        
        mock_client.table.return_value = mock_table
        mock_table.select.return_value = mock_select
        mock_select.execute.return_value = mock_execute
        mock_create_client.return_value = mock_client
        
        store = VectorStore(
            embedding_model=mock_embedding_model,
            supabase_url="https://test.supabase.co",
            supabase_key="test_key"
        )
        
        count = store.count()
        
        assert count == 42
        mock_table.select.assert_called_with("chunk_id", count="exact")
    
    @patch('services.vector_store.create_client')
    def test_count_empty_store(self, mock_create_client):
        """Test count returns 0 for empty store."""
        mock_embedding_model = Mock(spec=EmbeddingModel)
        
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_select = MagicMock()
        mock_execute = MagicMock()
        mock_execute.count = None
        
        mock_client.table.return_value = mock_table
        mock_table.select.return_value = mock_select
        mock_select.execute.return_value = mock_execute
        mock_create_client.return_value = mock_client
        
        store = VectorStore(
            embedding_model=mock_embedding_model,
            supabase_url="https://test.supabase.co",
            supabase_key="test_key"
        )
        
        count = store.count()
        
        assert count == 0
    
    @patch('services.vector_store.create_client')
    def test_count_failure(self, mock_create_client):
        """Test count handles database failure."""
        mock_embedding_model = Mock(spec=EmbeddingModel)
        
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_table.select.side_effect = Exception("Database error")
        mock_client.table.return_value = mock_table
        mock_create_client.return_value = mock_client
        
        store = VectorStore(
            embedding_model=mock_embedding_model,
            supabase_url="https://test.supabase.co",
            supabase_key="test_key"
        )
        
        with pytest.raises(RuntimeError, match="Failed to count chunks"):
            store.count()
