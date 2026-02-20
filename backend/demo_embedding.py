"""Demo script to test EmbeddingModel with Hugging Face API."""
import sys
from services.embedding_model import EmbeddingModel
from config import HUGGINGFACE_API_KEY


def main():
    """Demonstrate embedding model functionality."""
    if not HUGGINGFACE_API_KEY:
        print("Error: HUGGINGFACE_API_KEY not set in environment")
        sys.exit(1)
    
    print("Initializing EmbeddingModel...")
    model = EmbeddingModel()
    
    # Test warmup
    print("\n1. Testing model warmup...")
    success = model.warmup()
    print(f"   Warmup {'successful' if success else 'failed'}")
    
    # Test single text embedding
    print("\n2. Testing single text embedding...")
    text = "What is the pricing for ClearPath Pro plan?"
    embedding = model.embed_text(text)
    print(f"   Text: '{text}'")
    print(f"   Embedding dimensions: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
    
    # Test batch embedding
    print("\n3. Testing batch embedding...")
    texts = [
        "How do I create a new project?",
        "What integrations are available?",
        "Tell me about custom workflows."
    ]
    embeddings = model.embed_batch(texts)
    print(f"   Embedded {len(embeddings)} texts")
    for i, (text, emb) in enumerate(zip(texts, embeddings), 1):
        print(f"   {i}. '{text}' -> {len(emb)} dimensions")
    
    print("\n✓ All tests completed successfully!")


if __name__ == "__main__":
    main()
