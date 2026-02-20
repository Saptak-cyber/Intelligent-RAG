"""Embedding model integration with Hugging Face Inference API."""
import time
import logging
from typing import List
from huggingface_hub import InferenceClient
from config import HUGGINGFACE_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for Hugging Face Inference API embedding model."""
    
    def __init__(
        self,
        api_key: str = HUGGINGFACE_API_KEY,
        model_name: str = EMBEDDING_MODEL,
        max_retries: int = 5,
        initial_delay: float = 5.0,
        timeout: float = 120.0
    ):
        """
        Initialize the embedding model client.
        
        Args:
            api_key: Hugging Face API key
            model_name: Model identifier (default: sentence-transformers/all-mpnet-base-v2)
            max_retries: Maximum number of retry attempts for 503 errors
            initial_delay: Initial delay in seconds for exponential backoff
            timeout: Request timeout in seconds
        """
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY environment variable is required")
        
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.timeout = timeout
        
        # Initialize HuggingFace InferenceClient
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key,
            timeout=timeout
        )
        
        logger.info(f"Initialized EmbeddingModel with model: {model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            ValueError: If text is empty
            RuntimeError: If API request fails after all retries
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        result = self._embed_with_retry([text])
        # feature_extraction returns a list of embeddings, get the first one
        return result[0] if isinstance(result[0], list) else result
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a single API call.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If texts list is empty or contains empty strings
            RuntimeError: If API request fails after all retries
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        # Filter out empty strings and log warning
        valid_texts = [t for t in texts if t and t.strip()]
        if len(valid_texts) < len(texts):
            logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty texts from batch")
        
        if not valid_texts:
            raise ValueError("All texts in batch are empty")
        
        return self._embed_with_retry(valid_texts)
    
    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Internal method to call HF API with exponential backoff retry strategy.
        
        HF free tier models "sleep" and take 15-20s to load on first query.
        This implements aggressive retry with exponential backoff for 503 errors.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If API request fails after all retries
        """
        delay = self.initial_delay
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Use InferenceClient's feature_extraction method
                # Pass single text or list of texts
                input_data = texts[0] if len(texts) == 1 else texts
                embeddings = self.client.feature_extraction(
                    input_data,
                    model=self.model_name
                )
                
                elapsed = time.time() - start_time
                
                # Log successful request with timing
                if elapsed > 10.0:
                    logger.info(
                        f"Model loading delay detected: {elapsed:.1f}s for {len(texts)} texts "
                        f"(attempt {attempt + 1})"
                    )
                else:
                    logger.debug(f"Generated embeddings for {len(texts)} texts in {elapsed:.2f}s")
                
                # Ensure we return a list of embeddings
                if isinstance(embeddings, list) and len(embeddings) > 0:
                    # Check if it's a single embedding or list of embeddings
                    if isinstance(embeddings[0], (int, float)):
                        # Single embedding returned as flat list
                        return [embeddings]
                    else:
                        # List of embeddings
                        return embeddings
                else:
                    # Handle numpy arrays
                    import numpy as np
                    if isinstance(embeddings, np.ndarray):
                        # Convert numpy array to list
                        if embeddings.ndim == 1:
                            # Single embedding
                            return [embeddings.tolist()]
                        else:
                            # Multiple embeddings
                            return embeddings.tolist()
                    else:
                        raise RuntimeError(f"Unexpected embedding format: {type(embeddings)}")
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a model loading error (503)
                if "503" in error_msg or "service unavailable" in error_msg or "loading" in error_msg:
                    logger.warning(
                        f"Model loading (503) on attempt {attempt + 1}/{self.max_retries}. "
                        f"Retrying in {delay}s..."
                    )
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(delay)
                        delay = min(delay * 2, 60.0)  # Exponential backoff, max 60s
                        continue
                    else:
                        last_error = f"Model failed to load after {self.max_retries} attempts"
                        break
                
                # Check for rate limiting
                if "429" in error_msg or "rate limit" in error_msg:
                    logger.error("Rate limit exceeded for Hugging Face API")
                    raise RuntimeError("Rate limit exceeded. Please try again later.")
                
                # Check for authentication errors
                if "401" in error_msg or "unauthorized" in error_msg:
                    logger.error("Authentication failed for Hugging Face API")
                    raise RuntimeError("Invalid API key")
                
                # For other errors, retry with backoff
                last_error = str(e)
                logger.error(f"{last_error} on attempt {attempt + 1}/{self.max_retries}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                    continue
        
        # All retries exhausted
        error_msg = f"Failed to generate embeddings after {self.max_retries} attempts. Last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def warmup(self) -> bool:
        """
        Warm up the model with a dummy query to avoid cold start delays.
        
        This is useful to call at startup to ensure the model is loaded
        before processing real user queries.
        
        Returns:
            True if warmup successful, False otherwise
        """
        try:
            logger.info("Warming up embedding model...")
            start_time = time.time()
            
            # Use a simple dummy text
            self.embed_text("warmup query")
            
            elapsed = time.time() - start_time
            logger.info(f"Model warmup completed in {elapsed:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"Model warmup failed: {str(e)}")
            return False
