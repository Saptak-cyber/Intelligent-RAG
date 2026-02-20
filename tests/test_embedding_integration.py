"""Integration tests for EmbeddingModel with real API (optional)."""
import sys
from pathlib import Path
import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from services.embedding_model import EmbeddingModel
from config import HUGGINGFACE_API_KEY


@pytest.mark.skipif(
    not HUGGINGFACE_API_KEY,
    reason="HUGGINGFACE_API_KEY not set"
)
class TestEmbeddingIntegration:
    """Integration tests with real Hugging Face API."""
    
    def test_real_embed_text(self):
        """Test embedding a single text with real API."""
        model = EmbeddingModel()
        
        result = model.embed_text("This is a test sentence.")
        
        # all-mpnet-base-v2 produces 768-dimensional embeddings
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)
    
    def test_real_embed_batch(self):
        """Test embedding multiple texts with real API."""
        model = EmbeddingModel()
        
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        
        results = model.embed_batch(texts)
        
        assert len(results) == 3
        for embedding in results:
            assert len(embedding) == 768
            assert all(isinstance(x, float) for x in embedding)
    
    def test_real_warmup(self):
        """Test model warmup with real API."""
        model = EmbeddingModel()
        
        result = model.warmup()
        
        assert result is True
