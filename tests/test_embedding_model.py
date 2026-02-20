"""Unit tests for EmbeddingModel class."""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest
from unittest.mock import Mock, patch, MagicMock
from services.embedding_model import EmbeddingModel


class TestEmbeddingModel:
    """Test suite for EmbeddingModel."""
    
    def test_initialization_success(self):
        """Test successful initialization with API key."""
        with patch('services.embedding_model.InferenceClient'):
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
        with patch('services.embedding_model.InferenceClient'):
            model = EmbeddingModel(api_key="test_key")
            
            with pytest.raises(ValueError, match="Text cannot be empty"):
                model.embed_text("")
            
            with pytest.raises(ValueError, match="Text cannot be empty"):
                model.embed_text("   ")
    
    def test_embed_batch_empty_list(self):
        """Test embed_batch raises error for empty list."""
        with patch('services.embedding_model.InferenceClient'):
            model = EmbeddingModel(api_key="test_key")
            
            with pytest.raises(ValueError, match="Texts list cannot be empty"):
                model.embed_batch([])
    
    def test_embed_batch_all_empty_strings(self):
        """Test embed_batch raises error when all strings are empty."""
        with patch('services.embedding_model.InferenceClient'):
            model = EmbeddingModel(api_key="test_key")
            
            with pytest.raises(ValueError, match="All texts in batch are empty"):
                model.embed_batch(["", "   ", ""])
    
    def test_embed_text_success(self):
        """Test successful single text embedding."""
        mock_client = MagicMock()
        mock_client.feature_extraction.return_value = [0.1, 0.2, 0.3]
        
        with patch('services.embedding_model.InferenceClient', return_value=mock_client):
            model = EmbeddingModel(api_key="test_key")
            result = model.embed_text("test text")
            
            assert result == [0.1, 0.2, 0.3]
            assert mock_client.feature_extraction.called
    
    def test_embed_batch_success(self):
        """Test successful batch embedding."""
        mock_client = MagicMock()
        mock_client.feature_extraction.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        
        with patch('services.embedding_model.InferenceClient', return_value=mock_client):
            model = EmbeddingModel(api_key="test_key")
            result = model.embed_batch(["text1", "text2"])
            
            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]
    
    @patch('time.sleep')  # Mock sleep to speed up test
    def test_retry_on_503_success(self, mock_sleep):
        """Test retry logic succeeds after 503 error."""
        mock_client = MagicMock()
        # First call raises 503 error, second call succeeds
        mock_client.feature_extraction.side_effect = [
            Exception("503 Service Unavailable"),
            [0.1, 0.2, 0.3]
        ]
        
        with patch('services.embedding_model.InferenceClient', return_value=mock_client):
            model = EmbeddingModel(api_key="test_key", initial_delay=0.1)
            result = model.embed_text("test text")
            
            assert result == [0.1, 0.2, 0.3]
            assert mock_sleep.called  # Verify backoff was used
            assert mock_client.feature_extraction.call_count == 2
    
    @patch('time.sleep')
    def test_retry_exhausted_on_503(self, mock_sleep):
        """Test retry logic fails after max retries on 503."""
        mock_client = MagicMock()
        # Always raise 503 error
        mock_client.feature_extraction.side_effect = Exception("503 Service Unavailable")
        
        with patch('services.embedding_model.InferenceClient', return_value=mock_client):
            model = EmbeddingModel(api_key="test_key", max_retries=3, initial_delay=0.1)
            
            with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
                model.embed_text("test text")
            
            assert mock_client.feature_extraction.call_count == 3
    
    def test_rate_limit_error(self):
        """Test handling of 429 rate limit error."""
        mock_client = MagicMock()
        mock_client.feature_extraction.side_effect = Exception("429 Rate limit exceeded")
        
        with patch('services.embedding_model.InferenceClient', return_value=mock_client):
            model = EmbeddingModel(api_key="test_key")
            
            with pytest.raises(RuntimeError, match="Rate limit exceeded"):
                model.embed_text("test text")
    
    def test_authentication_error(self):
        """Test handling of 401 authentication error."""
        mock_client = MagicMock()
        mock_client.feature_extraction.side_effect = Exception("401 Unauthorized")
        
        with patch('services.embedding_model.InferenceClient', return_value=mock_client):
            model = EmbeddingModel(api_key="invalid_key")
            
            with pytest.raises(RuntimeError, match="Invalid API key"):
                model.embed_text("test text")
    
    @patch('time.sleep')
    def test_timeout_with_retry(self, mock_sleep):
        """Test handling of timeout with retry."""
        mock_client = MagicMock()
        # First call times out, second succeeds
        mock_client.feature_extraction.side_effect = [
            Exception("Timeout"),
            [0.1, 0.2, 0.3]
        ]
        
        with patch('services.embedding_model.InferenceClient', return_value=mock_client):
            model = EmbeddingModel(api_key="test_key", initial_delay=0.1)
            result = model.embed_text("test text")
            
            assert result == [0.1, 0.2, 0.3]
            assert mock_sleep.called
    
    @patch('time.sleep')
    def test_network_error_with_retry(self, mock_sleep):
        """Test handling of network error with retry."""
        mock_client = MagicMock()
        # First call has network error, second succeeds
        mock_client.feature_extraction.side_effect = [
            Exception("Network error"),
            [0.1, 0.2, 0.3]
        ]
        
        with patch('services.embedding_model.InferenceClient', return_value=mock_client):
            model = EmbeddingModel(api_key="test_key", initial_delay=0.1)
            result = model.embed_text("test text")
            
            assert result == [0.1, 0.2, 0.3]
            assert mock_sleep.called
    
    def test_warmup_success(self):
        """Test successful model warmup."""
        mock_client = MagicMock()
        mock_client.feature_extraction.return_value = [0.1, 0.2, 0.3]
        
        with patch('services.embedding_model.InferenceClient', return_value=mock_client):
            model = EmbeddingModel(api_key="test_key")
            result = model.warmup()
            
            assert result is True
            assert mock_client.feature_extraction.called
    
    def test_warmup_failure(self):
        """Test model warmup handles failure gracefully."""
        mock_client = MagicMock()
        mock_client.feature_extraction.side_effect = Exception("API error")
        
        with patch('services.embedding_model.InferenceClient', return_value=mock_client):
            model = EmbeddingModel(api_key="test_key")
            result = model.warmup()
            
            assert result is False
    
    @patch('time.sleep')
    def test_exponential_backoff_delays(self, mock_sleep):
        """Test that exponential backoff increases delays correctly."""
        mock_client = MagicMock()
        mock_client.feature_extraction.side_effect = Exception("503 Service Unavailable")
        
        with patch('services.embedding_model.InferenceClient', return_value=mock_client):
            model = EmbeddingModel(api_key="test_key", max_retries=3, initial_delay=2.0)
            
            with pytest.raises(RuntimeError):
                model.embed_text("test text")
            
            # Check that delays increased: 2s, 4s (exponential backoff)
            calls = mock_sleep.call_args_list
            assert len(calls) == 2  # 3 attempts = 2 sleeps
            assert calls[0][0][0] == 2.0  # First delay
            assert calls[1][0][0] == 4.0  # Second delay (doubled)
