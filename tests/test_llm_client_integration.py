"""Integration tests for LLMClient with Groq API.

These tests require a valid GROQ_API_KEY in the environment.
They will be skipped if the API key is not available.
"""
import sys
sys.path.insert(0, 'backend')

import pytest
import os
from services.llm_client import LLMClient, LLMResponse


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set in environment"
)
class TestLLMClientIntegration:
    """Integration tests for LLMClient with real Groq API."""
    
    @pytest.fixture
    def client(self):
        """Create LLMClient instance."""
        return LLMClient()
    
    def test_generate_with_simple_model(self, client):
        """Test generation with llama-3.1-8b-instant model."""
        chunks = ["ClearPath offers three pricing tiers: Basic, Pro, and Enterprise."]
        
        prompt = LLMClient.build_prompt(
            query="What pricing tiers does ClearPath offer?",
            retrieved_chunks=chunks,
            conversation_history=None
        )
        
        response = client.generate(
            model="llama-3.1-8b-instant",
            prompt=prompt,
            max_tokens=50
        )
        
        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert len(response.text) > 0
        assert response.tokens_input > 0
        assert response.tokens_output > 0
        assert response.latency_ms > 0
        assert response.model_used == "llama-3.1-8b-instant"
        
        # Verify the answer mentions pricing tiers
        response_lower = response.text.lower()
        assert any(word in response_lower for word in ["basic", "pro", "enterprise", "tier", "pricing"])
    
    def test_generate_with_complex_model(self, client):
        """Test generation with llama-3.3-70b-versatile model."""
        chunks = [
            "ClearPath is a project management tool.",
            "It helps teams collaborate effectively."
        ]
        
        prompt = LLMClient.build_prompt(
            query="What is ClearPath?",
            retrieved_chunks=chunks,
            conversation_history=None
        )
        
        response = client.generate(
            model="llama-3.3-70b-versatile",
            prompt=prompt,
            max_tokens=100
        )
        
        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert len(response.text) > 0
        assert response.tokens_input > 0
        assert response.tokens_output > 0
        assert response.latency_ms > 0
        assert response.model_used == "llama-3.3-70b-versatile"
        
        # Verify the answer mentions ClearPath
        assert "ClearPath" in response.text or "project management" in response.text.lower()
    
    def test_generate_with_conversation_history(self, client):
        """Test generation with conversation history."""
        history = "Previous Q: What is ClearPath? A: ClearPath is a project management tool."
        
        prompt = LLMClient.build_prompt(
            query="What does it help with?",
            retrieved_chunks=["ClearPath helps teams collaborate and manage projects."],
            conversation_history=history
        )
        
        response = client.generate(
            model="llama-3.1-8b-instant",
            prompt=prompt,
            max_tokens=100
        )
        
        # Verify response
        assert isinstance(response, LLMResponse)
        assert len(response.text) > 0
        assert response.tokens_input > 0
        assert response.tokens_output > 0
    
    def test_token_tracking_accuracy(self, client):
        """Test that token counts are reasonable."""
        prompt = LLMClient.build_prompt(
            query="Hello",
            retrieved_chunks=None,
            conversation_history=None
        )
        
        response = client.generate(
            model="llama-3.1-8b-instant",
            prompt=prompt,
            max_tokens=20
        )
        
        # Token counts should be reasonable
        assert response.tokens_input > 10  # At least the system prompt
        assert response.tokens_output > 0
        assert response.tokens_output <= 20  # Should respect max_tokens
