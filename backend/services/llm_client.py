"""LLM Client for Groq API integration."""
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from groq import Groq
from groq import RateLimitError, AuthenticationError, APIError, APITimeoutError
import logging

from config import GROQ_API_KEY

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    text: str
    tokens_input: int
    tokens_output: int
    latency_ms: int
    model_used: str


@dataclass
class LLMError:
    """Structured error response from LLM operations."""
    code: str
    message: str
    details: Dict[str, Any]


class LLMClientError(Exception):
    """Custom exception for LLM client errors with structured error information."""
    
    def __init__(self, error: LLMError):
        self.error = error
        super().__init__(error.message)


class LLMClient:
    """Client for interfacing with Groq API for text generation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM client with Groq API key.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY from environment)
        """
        self.api_key = api_key or GROQ_API_KEY
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided or set in environment")
        
        self.client = Groq(api_key=self.api_key)
        logger.info("LLMClient initialized successfully")
    
    def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 500
    ) -> LLMResponse:
        """
        Generate response using Groq API.
        
        Args:
            model: Model name (llama-3.1-8b-instant or llama-3.3-70b-versatile)
            prompt: Complete prompt with context and query
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse with text, token counts, and latency
            
        Raises:
            LLMClientError: Structured error with code, message, and details
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Generating response with model: {model}")
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract response text
            text = response.choices[0].message.content
            
            # Extract token usage
            tokens_input = response.usage.prompt_tokens
            tokens_output = response.usage.completion_tokens
            
            logger.info(
                f"Generated response: model={model}, "
                f"input_tokens={tokens_input}, output_tokens={tokens_output}, "
                f"latency={latency_ms}ms"
            )
            
            return LLMResponse(
                text=text,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                latency_ms=latency_ms,
                model_used=model
            )
            
        except RateLimitError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            error = LLMError(
                code="RATE_LIMIT_ERROR",
                message="Rate limit exceeded. Please try again in a few moments.",
                details={
                    "retry_after": 60,  # Suggest retry after 60 seconds
                    "model": model,
                    "latency_ms": latency_ms,
                    "original_error": str(e)
                }
            )
            logger.error(
                f"Rate limit error: model={model}, latency={latency_ms}ms, error={e}",
                exc_info=True,
                extra={"error_code": error.code, "error_details": error.details}
            )
            raise LLMClientError(error)
            
        except AuthenticationError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            error = LLMError(
                code="AUTHENTICATION_ERROR",
                message="Authentication failed. Please check your API key.",
                details={
                    "model": model,
                    "latency_ms": latency_ms,
                    "original_error": str(e)
                }
            )
            logger.error(
                f"Authentication error: model={model}, latency={latency_ms}ms, error={e}",
                exc_info=True,
                extra={"error_code": error.code, "error_details": error.details}
            )
            raise LLMClientError(error)
            
        except APITimeoutError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            error = LLMError(
                code="TIMEOUT_ERROR",
                message="Request timed out. Please try again.",
                details={
                    "model": model,
                    "latency_ms": latency_ms,
                    "original_error": str(e)
                }
            )
            logger.error(
                f"Timeout error: model={model}, latency={latency_ms}ms, error={e}",
                exc_info=True,
                extra={"error_code": error.code, "error_details": error.details}
            )
            raise LLMClientError(error)
            
        except APIError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            error = LLMError(
                code="API_ERROR",
                message=f"Groq API error: {str(e)}",
                details={
                    "model": model,
                    "latency_ms": latency_ms,
                    "original_error": str(e)
                }
            )
            logger.error(
                f"API error: model={model}, latency={latency_ms}ms, error={e}",
                exc_info=True,
                extra={"error_code": error.code, "error_details": error.details}
            )
            raise LLMClientError(error)
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            error = LLMError(
                code="UNKNOWN_ERROR",
                message=f"Unexpected error during generation: {str(e)}",
                details={
                    "model": model,
                    "latency_ms": latency_ms,
                    "original_error": str(e),
                    "error_type": type(e).__name__
                }
            )
            logger.error(
                f"Unexpected error: model={model}, latency={latency_ms}ms, error={e}",
                exc_info=True,
                extra={"error_code": error.code, "error_details": error.details}
            )
            raise LLMClientError(error)
    
    @staticmethod
    def build_prompt(
        query: str,
        retrieved_chunks: Optional[List[str]] = None,
        conversation_history: Optional[str] = None
    ) -> str:
        """
        Build prompt template with context and query.
        
        Args:
            query: User question
            retrieved_chunks: List of retrieved chunk texts
            conversation_history: Formatted conversation history
            
        Returns:
            Complete prompt string
        """
        # Build context section
        context_section = ""
        if retrieved_chunks:
            context_text = "\n\n".join(retrieved_chunks)
            context_section = f"""Context from documentation:
{context_text}

"""
        
        # Build conversation history section
        history_section = ""
        if conversation_history:
            history_section = f"""{conversation_history}

"""
        
        # Build complete prompt
        prompt = f"""You are a helpful customer support assistant for ClearPath, a project management tool.

{context_section}{history_section}User question: {query}

Instructions:
- Answer based on the provided context
- If the context doesn't contain relevant information, say so clearly
- Be concise and helpful
- Cite specific features or details from the documentation when applicable

Answer:"""
        
        return prompt
