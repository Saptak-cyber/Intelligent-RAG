"""Unit tests for LLMClient."""
import sys
sys.path.insert(0, 'backend')

import pytest
from unittest.mock import Mock, patch
from services.llm_client import LLMClient, LLMResponse, LLMError, LLMClientError
from groq import RateLimitError, AuthenticationError, APIError, APITimeoutError


class TestLLMClient:
    """Test suite for LLMClient class."""
    
    def test_initialization_with_api_key(self):
        """Test LLMClient initializes with provided API key."""
        client = LLMClient(api_key="test_key")
        assert client.api_key == "test_key"
    
    def test_initialization_without_api_key_raises_error(self):
        """Test LLMClient raises error when no API key provided."""
        with patch('services.llm_client.GROQ_API_KEY', None):
            with pytest.raises(ValueError, match="GROQ_API_KEY must be provided"):
                LLMClient()
    
    def test_build_prompt_with_context_and_history(self):
        """Test prompt building with context and conversation history."""
        query = "What is the Pro plan price?"
        chunks = ["Pro plan costs $29/month", "Includes 10 users"]
        history = "Previous Q: What is ClearPath? A: A project management tool."
        
        prompt = LLMClient.build_prompt(query, chunks, history)
        
        assert "ClearPath" in prompt
        assert "project management tool" in prompt
        assert query in prompt
        assert "Pro plan costs $29/month" in prompt
        assert "Includes 10 users" in prompt
        assert history in prompt
    
    def test_build_prompt_without_context(self):
        """Test prompt building without context chunks."""
        query = "Hello, who are you?"
        
        prompt = LLMClient.build_prompt(query, None, None)
        
        assert query in prompt
        assert "ClearPath" in prompt
        assert "Context from documentation:" not in prompt
    
    def test_build_prompt_without_history(self):
        """Test prompt building without conversation history."""
        query = "What is pricing?"
        chunks = ["Pro plan costs $29/month"]
        
        prompt = LLMClient.build_prompt(query, chunks, None)
        
        assert query in prompt
        assert "Pro plan costs $29/month" in prompt
        assert "Previous Q:" not in prompt
    
    @patch('services.llm_client.Groq')
    def test_generate_success(self, mock_groq_class):
        """Test successful response generation."""
        # Mock Groq API response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="The Pro plan costs $29/month."))]
        mock_response.usage = Mock(prompt_tokens=150, completion_tokens=12)
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq_class.return_value = mock_client
        
        # Create client and generate
        client = LLMClient(api_key="test_key")
        response = client.generate(
            model="llama-3.1-8b-instant",
            prompt="What is the Pro plan price?"
        )
        
        # Verify response
        assert isinstance(response, LLMResponse)
        assert response.text == "The Pro plan costs $29/month."
        assert response.tokens_input == 150
        assert response.tokens_output == 12
        assert response.model_used == "llama-3.1-8b-instant"
        assert response.latency_ms >= 0
    
    @patch('services.llm_client.Groq')
    def test_generate_tracks_latency(self, mock_groq_class):
        """Test that latency is measured correctly."""
        # Mock Groq API response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Answer"))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=10)
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq_class.return_value = mock_client
        
        client = LLMClient(api_key="test_key")
        response = client.generate(
            model="llama-3.1-8b-instant",
            prompt="Test prompt"
        )
        
        # Latency should be a positive integer
        assert isinstance(response.latency_ms, int)
        assert response.latency_ms >= 0
    
    @patch('services.llm_client.Groq')
    def test_generate_with_different_models(self, mock_groq_class):
        """Test generation with different model names."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Answer"))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=10)
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq_class.return_value = mock_client
        
        client = LLMClient(api_key="test_key")
        
        # Test with simple model
        response1 = client.generate(
            model="llama-3.1-8b-instant",
            prompt="Simple query"
        )
        assert response1.model_used == "llama-3.1-8b-instant"
        
        # Test with complex model
        response2 = client.generate(
            model="llama-3.3-70b-versatile",
            prompt="Complex query"
        )
        assert response2.model_used == "llama-3.3-70b-versatile"
    
    @patch('services.llm_client.Groq')
    def test_generate_handles_api_error(self, mock_groq_class):
        """Test that API errors are properly raised with structured error."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_groq_class.return_value = mock_client
        
        client = LLMClient(api_key="test_key")
        
        with pytest.raises(LLMClientError) as exc_info:
            client.generate(
                model="llama-3.1-8b-instant",
                prompt="Test prompt"
            )
        
        # Verify structured error
        error = exc_info.value.error
        assert error.code == "UNKNOWN_ERROR"
        assert "Unexpected error" in error.message
        assert "model" in error.details
        assert error.details["model"] == "llama-3.1-8b-instant"
    
    @patch('services.llm_client.Groq')
    def test_generate_handles_rate_limit_error(self, mock_groq_class):
        """Test that rate limit errors are handled with retry suggestion."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit exceeded",
            response=Mock(status_code=429),
            body=None
        )
        mock_groq_class.return_value = mock_client
        
        client = LLMClient(api_key="test_key")
        
        with pytest.raises(LLMClientError) as exc_info:
            client.generate(
                model="llama-3.1-8b-instant",
                prompt="Test prompt"
            )
        
        # Verify structured error
        error = exc_info.value.error
        assert error.code == "RATE_LIMIT_ERROR"
        assert "Rate limit exceeded" in error.message
        assert "retry_after" in error.details
        assert error.details["retry_after"] == 60
        assert error.details["model"] == "llama-3.1-8b-instant"
    
    @patch('services.llm_client.Groq')
    def test_generate_handles_authentication_error(self, mock_groq_class):
        """Test that authentication errors are handled properly."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            message="Invalid API key",
            response=Mock(status_code=401),
            body=None
        )
        mock_groq_class.return_value = mock_client
        
        client = LLMClient(api_key="test_key")
        
        with pytest.raises(LLMClientError) as exc_info:
            client.generate(
                model="llama-3.1-8b-instant",
                prompt="Test prompt"
            )
        
        # Verify structured error
        error = exc_info.value.error
        assert error.code == "AUTHENTICATION_ERROR"
        assert "Authentication failed" in error.message
        assert error.details["model"] == "llama-3.1-8b-instant"
    
    @patch('services.llm_client.Groq')
    def test_generate_handles_timeout_error(self, mock_groq_class):
        """Test that timeout errors are handled properly."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = APITimeoutError(request=Mock())
        mock_groq_class.return_value = mock_client
        
        client = LLMClient(api_key="test_key")
        
        with pytest.raises(LLMClientError) as exc_info:
            client.generate(
                model="llama-3.1-8b-instant",
                prompt="Test prompt"
            )
        
        # Verify structured error
        error = exc_info.value.error
        assert error.code == "TIMEOUT_ERROR"
        assert "timed out" in error.message
        assert error.details["model"] == "llama-3.1-8b-instant"
    
    @patch('services.llm_client.Groq')
    def test_generate_handles_generic_api_error(self, mock_groq_class):
        """Test that generic API errors are handled properly."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = APIError(
            message="Service unavailable",
            request=Mock(),
            body=None
        )
        mock_groq_class.return_value = mock_client
        
        client = LLMClient(api_key="test_key")
        
        with pytest.raises(LLMClientError) as exc_info:
            client.generate(
                model="llama-3.1-8b-instant",
                prompt="Test prompt"
            )
        
        # Verify structured error
        error = exc_info.value.error
        assert error.code == "API_ERROR"
        assert "Groq API error" in error.message
        assert error.details["model"] == "llama-3.1-8b-instant"
    
    @patch('services.llm_client.Groq')
    def test_error_includes_latency(self, mock_groq_class):
        """Test that errors include latency measurement."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit exceeded",
            response=Mock(status_code=429),
            body=None
        )
        mock_groq_class.return_value = mock_client
        
        client = LLMClient(api_key="test_key")
        
        with pytest.raises(LLMClientError) as exc_info:
            client.generate(
                model="llama-3.1-8b-instant",
                prompt="Test prompt"
            )
        
        # Verify latency is tracked even on error
        error = exc_info.value.error
        assert "latency_ms" in error.details
        assert isinstance(error.details["latency_ms"], int)
        assert error.details["latency_ms"] >= 0
