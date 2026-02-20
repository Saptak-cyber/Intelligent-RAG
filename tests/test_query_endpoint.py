"""Integration tests for the POST /query endpoint."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


@pytest.fixture
def client():
    """Create a test client with mocked services."""
    # Import after path is set
    from main import app
    
    # Mock the startup event to avoid initializing real services
    with patch('main.startup_event'):
        client = TestClient(app)
        
        # Manually set the global services to mocks
        import main
        main.model_router = Mock()
        main.retrieval_engine = Mock()
        main.llm_client = Mock()
        main.output_evaluator = Mock()
        main.conversation_manager = Mock()
        main.routing_logger = Mock()
        main.tiktoken_encoder = Mock()
        
        yield client


@pytest.fixture
def mock_services(client):
    """Mock all services to avoid external dependencies."""
    import main
    from models.chunk import Chunk, ScoredChunk
    from services.model_router import Classification
    from services.llm_client import LLMResponse
    
    # Mock model router
    main.model_router.classify_query.return_value = Classification(
        category="simple",
        model_name="llama-3.1-8b-instant",
        reasoning="Test classification",
        skip_retrieval=False,
        rule_triggered="default"
    )
    
    # Mock retrieval engine
    mock_chunk = Chunk(
        chunk_id="test_1",
        text="Test chunk text",
        document_name="test.pdf",
        page_number=1,
        token_count=10
    )
    main.retrieval_engine.retrieve.return_value = [
        ScoredChunk(chunk=mock_chunk, relevance_score=0.85)
    ]
    
    # Mock LLM client
    main.llm_client.generate.return_value = LLMResponse(
        text="This is a test answer.",
        tokens_input=100,
        tokens_output=20,
        latency_ms=500,
        model_used="llama-3.1-8b-instant"
    )
    
    # Mock LLMClient.build_prompt as a static method
    from services.llm_client import LLMClient
    with patch.object(LLMClient, 'build_prompt', return_value="Test prompt"):
        # Mock output evaluator
        main.output_evaluator.evaluate.return_value = []
        
        # Mock conversation manager
        mock_conversation = MagicMock()
        mock_conversation.conversation_id = "conv_test123"
        mock_conversation.turns = []
        main.conversation_manager.get_or_create_conversation.return_value = mock_conversation
        main.conversation_manager.get_context.return_value = ""
        
        # Mock tiktoken encoder
        main.tiktoken_encoder.encode.return_value = [1] * 100  # 100 tokens
        
        yield {
            'router': main.model_router,
            'retrieval': main.retrieval_engine,
            'llm': main.llm_client,
            'evaluator': main.output_evaluator,
            'conv_manager': main.conversation_manager,
            'logger': main.routing_logger,
            'tiktoken': main.tiktoken_encoder
        }


def test_query_endpoint_basic(client, mock_services):
    """Test basic query endpoint functionality."""
    response = client.post(
        "/query",
        json={"question": "What is ClearPath?"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "answer" in data
    assert "metadata" in data
    assert "sources" in data
    assert "conversation_id" in data
    
    # Verify metadata structure
    metadata = data["metadata"]
    assert "model_used" in metadata
    assert "classification" in metadata
    assert "tokens" in metadata
    assert "latency_ms" in metadata
    assert "chunks_retrieved" in metadata
    assert "evaluator_flags" in metadata
    
    # Verify token structure
    tokens = metadata["tokens"]
    assert "input" in tokens
    assert "output" in tokens


def test_query_endpoint_with_conversation_id(client, mock_services):
    """Test query endpoint with existing conversation ID."""
    response = client.post(
        "/query",
        json={
            "question": "What is pricing?",
            "conversation_id": "conv_existing123"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify conversation_id is returned
    assert data["conversation_id"] == "conv_test123"


def test_query_endpoint_empty_question(client, mock_services):
    """Test query endpoint with empty question."""
    response = client.post(
        "/query",
        json={"question": ""}
    )
    
    # Pydantic validation returns 422 for validation errors
    assert response.status_code == 422


def test_query_endpoint_ood_filter(client, mock_services):
    """Test query endpoint with OOD filter (skip retrieval)."""
    import main
    from services.model_router import Classification
    
    # Mock OOD classification
    main.model_router.classify_query.return_value = Classification(
        category="simple",
        model_name="llama-3.1-8b-instant",
        reasoning="Greeting detected",
        skip_retrieval=True,
        rule_triggered="ood_filter"
    )
    
    response = client.post(
        "/query",
        json={"question": "Hello!"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify chunks_retrieved is 0 when retrieval is skipped
    assert data["metadata"]["chunks_retrieved"] == 0


def test_query_endpoint_with_evaluator_flags(client, mock_services):
    """Test query endpoint with evaluator flags."""
    import main
    
    # Mock evaluator to return flags
    main.output_evaluator.evaluate.return_value = ["no_context", "refusal"]
    
    response = client.post(
        "/query",
        json={"question": "What is the price?"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify evaluator flags are included
    assert "no_context" in data["metadata"]["evaluator_flags"]
    assert "refusal" in data["metadata"]["evaluator_flags"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
