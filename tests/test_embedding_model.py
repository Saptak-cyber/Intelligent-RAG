"""Unit tests for EmbeddingModel class."""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest
from unittest.mock import Mock, patch, MagicMock
import httpx
from services.embedding_model import EmbeddingModel


class TestEmbeddingModel:
    """Test suite for EmbeddingModel."""
    
    def test_initialization_success(self):
        """Test successful initialization with API key."""
        model = EmbeddingModel(api_key="test_key")
        assert model.api_key == "test_key"
        assert model.model_name == "sentence-transformers/all-mpnet-base-v2"
        assert model.max_retries == 5
    
    def test_initialization_without_api_key(self):
        """Test initialization fails without API key."""
        with patch('services.embedding_model.HUGGINGFACE_API_KEY', None):
            with pytest.raises(ValueError, match="HUGGINGFACE_API_KEY"):
                EmbeddingModel(api_key=None)
    
    def test_embed_text_empty_string(self):
        """Test embed_text raises error for empty string."""
        model = EmbeddingModel(api_key="test_key")
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            model.embed_text("")
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            model.embed_text("   ")
    
    def test_embed_batch_empty_list(self):
        """Test embed_batch raises error for empty list."""
        model = EmbeddingModel(api_key="test_key")
        
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            model.embed_batch([])
    
    def test_embed_batch_all_empty_strings(self):
        """Test embed_batch raises error when all strings are empty."""
        model = EmbeddingModel(api_key="test_key")
        
        with pytest.raises(ValueError, match="All texts in batch are empty"):
            model.embed_batch(["", "   ", ""])
    
    @patch('httpx.Client')
    def test_embed_text_success(self, mock_client_class):
        """Test successful single text embedding."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [[0.1, 0.2, 0.3]]
        
        # Mock client
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        model = EmbeddingModel(api_key="test_key")
        result = model.embed_text("test text")
        
        assert result == [0.1, 0.2, 0.3]
        assert mock_client.__enter__.return_value.post.called
    
    @patch('httpx.Client')
    def test_embed_batch_success(self, mock_client_class):
        """Test successful batch embedding."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        
        # Mock client
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        model = EmbeddingModel(api_key="test_key")
        result = model.embed_batch(["text1", "text2"])
        
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
    
    @patch('httpx.Client')
    @patch('time.sleep')  # Mock sleep to speed up test
    def test_retry_on_503_success(self, mock_sleep, mock_client_class):
        """Test retry logic succeeds after 503 error."""
        # First call returns 503, second call succeeds
        mock_response_503 = Mock()
        mock_response_503.status_code = 503
        mock_response_503.json.return_value = {"estimated_time": 10}
        mock_response_503.text = '{"estimated_time": 10}'
        
        mock_response_200 = Mock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = [[0.1, 0.2, 0.3]]
        
        # Mock client to return 503 then 200
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.side_effect = [
            mock_response_503,
            mock_response_200
        ]
        mock_client_class.return_value = mock_client
        
        model = EmbeddingModel(api_key="test_key", initial_delay=1.0)
        result = model.embed_text("test text")
        
        assert result == [0.1, 0.2, 0.3]
        assert mock_sleep.called  # Verify backoff was used
        assert mock_client.__enter__.return_value.post.call_count == 2
    
    @patch('httpx.Client')
    @patch('time.sleep')
    def test_retry_exhausted_on_503(self, mock_sleep, mock_client_class):
        """Test retry logic fails after max retries on 503."""
        # Always return 503
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.json.return_value = {"estimated_time": 10}
        mock_response.text = '{"estimated_time": 10}'
        
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        model = EmbeddingModel(api_key="test_key", max_retries=3, initial_delay=0.1)
        
        with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
            model.embed_text("test text")
        
        assert mock_client.__enter__.return_value.post.call_count == 3
    
    @patch('httpx.Client')
    def test_rate_limit_error(self, mock_client_class):
        """Test handling of 429 rate limit error."""
        mock_response = Mock()
        mock_response.status_code = 429
        
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        model = EmbeddingModel(api_key="test_key")
        
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            model.embed_text("test text")
    
    @patch('httpx.Client')
    def test_authentication_error(self, mock_client_class):
        """Test handling of 401 authentication error."""
        mock_response = Mock()
        mock_response.status_code = 401
        
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        model = EmbeddingModel(api_key="invalid_key")
        
        with pytest.raises(RuntimeError, match="Invalid API key"):
            model.embed_text("test text")
    
    @patch('httpx.Client')
    @patch('time.sleep')
    def test_timeout_with_retry(self, mock_sleep, mock_client_class):
        """Test handling of timeout with retry."""
        # First call times out, second succeeds
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [[0.1, 0.2, 0.3]]
        
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.side_effect = [
            httpx.TimeoutException("Timeout"),
            mock_response
        ]
        mock_client_class.return_value = mock_client
        
        model = EmbeddingModel(api_key="test_key", initial_delay=0.1)
        result = model.embed_text("test text")
        
        assert result == [0.1, 0.2, 0.3]
        assert mock_sleep.called
    
    @patch('httpx.Client')
    @patch('time.sleep')
    def test_network_error_with_retry(self, mock_sleep, mock_client_class):
        """Test handling of network error with retry."""
        # First call has network error, second succeeds
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [[0.1, 0.2, 0.3]]
        
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.side_effect = [
            httpx.RequestError("Network error"),
            mock_response
        ]
        mock_client_class.return_value = mock_client
        
        model = EmbeddingModel(api_key="test_key", initial_delay=0.1)
        result = model.embed_text("test text")
        
        assert result == [0.1, 0.2, 0.3]
        assert mock_sleep.called
    
    @patch('httpx.Client')
    def test_warmup_success(self, mock_client_class):
        """Test successful model warmup."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [[0.1, 0.2, 0.3]]
        
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        model = EmbeddingModel(api_key="test_key")
        result = model.warmup()
        
        assert result is True
        assert mock_client.__enter__.return_value.post.called
    
    @patch('httpx.Client')
    def test_warmup_failure(self, mock_client_class):
        """Test model warmup handles failure gracefully."""
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.side_effect = Exception("API error")
        mock_client_class.return_value = mock_client
        
        model = EmbeddingModel(api_key="test_key")
        result = model.warmup()
        
        assert result is False
    
    @patch('httpx.Client')
    def test_exponential_backoff_delays(self, mock_client_class):
        """Test that exponential backoff increases delays correctly."""
        mock_response_503 = Mock()
        mock_response_503.status_code = 503
        mock_response_503.json.return_value = {"estimated_time": 5}
        mock_response_503.text = '{"estimated_time": 5}'
        
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response_503
        mock_client_class.return_value = mock_client
        
        model = EmbeddingModel(api_key="test_key", max_retries=3, initial_delay=2.0)
        
        with patch('time.sleep') as mock_sleep:
            with pytest.raises(RuntimeError):
                model.embed_text("test text")
            
            # Check that delays increased: 2s, 4s (exponential backoff)
            calls = mock_sleep.call_args_list
            assert len(calls) == 2  # 3 attempts = 2 sleeps
            assert calls[0][0][0] == 2.0  # First delay
            assert calls[1][0][0] == 4.0  # Second delay (doubled)
