"""Embedding model integration with Hugging Face Inference API."""
import time
import logging
from typing import List, Union
import httpx
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
        # Use the new router endpoint as per HF API migration
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
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
        
        return self._embed_with_retry([text])[0]
    
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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": texts,
            "options": {
                "wait_for_model": True  # Wait for model to load if sleeping
            }
        }
        
        delay = self.initial_delay
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        self.api_url,
                        headers=headers,
                        json=payload
                    )
                
                elapsed = time.time() - start_time
                
                # Handle 503 Service Unavailable (model loading)
                if response.status_code == 503:
                    error_data = response.json() if response.text else {}
                    estimated_time = error_data.get("estimated_time", delay)
                    
                    logger.warning(
                        f"Model loading (503) on attempt {attempt + 1}/{self.max_retries}. "
                        f"Estimated time: {estimated_time}s. Retrying in {delay}s..."
                    )
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(delay)
                        delay = min(delay * 2, 60.0)  # Exponential backoff, max 60s
                        continue
                    else:
                        last_error = f"Model failed to load after {self.max_retries} attempts"
                        break
                
                # Handle rate limiting
                if response.status_code == 429:
                    logger.error("Rate limit exceeded for Hugging Face API")
                    raise RuntimeError("Rate limit exceeded. Please try again later.")
                
                # Handle authentication errors
                if response.status_code == 401:
                    logger.error("Authentication failed for Hugging Face API")
                    raise RuntimeError("Invalid API key")
                
                # Handle other errors
                if response.status_code != 200:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # Success - parse embeddings
                embeddings = response.json()
                
                # Log successful request with timing
                if elapsed > 10.0:
                    logger.info(
                        f"Model loading delay detected: {elapsed:.1f}s for {len(texts)} texts "
                        f"(attempt {attempt + 1})"
                    )
                else:
                    logger.debug(f"Generated embeddings for {len(texts)} texts in {elapsed:.2f}s")
                
                return embeddings
                
            except httpx.TimeoutException as e:
                last_error = f"Request timeout after {self.timeout}s"
                logger.error(f"{last_error} on attempt {attempt + 1}/{self.max_retries}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                    continue
                    
            except httpx.RequestError as e:
                last_error = f"Network error: {str(e)}"
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
