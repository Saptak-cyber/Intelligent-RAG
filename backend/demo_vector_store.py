"""Demo script to test VectorStore functionality with Supabase."""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from services.embedding_model import EmbeddingModel
from services.vector_store import VectorStore
from models.chunk import Chunk
from config import HUGGINGFACE_API_KEY, SUPABASE_URL, SUPABASE_KEY
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate VectorStore functionality."""
    try:
        # Initialize embedding model
        logger.info("Initializing embedding model...")
        embedding_model = EmbeddingModel(api_key=HUGGINGFACE_API_KEY)
        
        # Warm up the model
        logger.info("Warming up embedding model...")
        embedding_model.warmup()
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = VectorStore(
            embedding_model=embedding_model,
            supabase_url=SUPABASE_URL,
            supabase_key=SUPABASE_KEY
        )
        
        # Create sample chunks
        logger.info("Creating sample chunks...")
        chunks = [
            Chunk(
                chunk_id="demo_1_0",
                text="ClearPath is a project management tool designed for modern teams. It offers features like task tracking, team collaboration, and real-time updates.",
                document_name="demo_doc.pdf",
                page_number=1,
                token_count=30,
                context_header="[Context: Introduction]"
            ),
            Chunk(
                chunk_id="demo_1_1",
                text="The Pro plan costs $29 per user per month and includes advanced features like custom workflows, priority support, and unlimited projects.",
                document_name="demo_doc.pdf",
                page_number=1,
                token_count=28,
                context_header="[Context: Pricing]"
            ),
            Chunk(
                chunk_id="demo_2_0",
                text="ClearPath integrates with popular tools like Slack, GitHub, and Jira. You can set up webhooks to automate your workflow.",
                document_name="demo_doc.pdf",
                page_number=2,
                token_count=25,
                context_header="[Context: Integrations]"
            )
        ]
        
        # Add chunks to vector store
        logger.info("Adding chunks to vector store...")
        vector_store.add_chunks(chunks)
        
        # Check count
        count = vector_store.count()
        logger.info(f"Total chunks in store: {count}")
        
        # Test search
        logger.info("\nTesting search functionality...")
        query = "What is the pricing for ClearPath?"
        logger.info(f"Query: {query}")
        
        # Generate query embedding
        query_embedding = embedding_model.embed_text(query)
        
        # Search for similar chunks
        results = vector_store.search(query_embedding, top_k=3)
        
        logger.info(f"\nFound {len(results)} results:")
        for i, scored_chunk in enumerate(results, 1):
            logger.info(f"\n--- Result {i} ---")
            logger.info(f"Chunk ID: {scored_chunk.chunk.chunk_id}")
            logger.info(f"Document: {scored_chunk.chunk.document_name}")
            logger.info(f"Page: {scored_chunk.chunk.page_number}")
            logger.info(f"Context: {scored_chunk.chunk.context_header}")
            logger.info(f"Relevance Score: {scored_chunk.relevance_score:.4f}")
            logger.info(f"Text: {scored_chunk.chunk.text[:100]}...")
        
        # Test another query
        logger.info("\n" + "="*60)
        query2 = "How do I integrate with Slack?"
        logger.info(f"Query: {query2}")
        
        query_embedding2 = embedding_model.embed_text(query2)
        results2 = vector_store.search(query_embedding2, top_k=3)
        
        logger.info(f"\nFound {len(results2)} results:")
        for i, scored_chunk in enumerate(results2, 1):
            logger.info(f"\n--- Result {i} ---")
            logger.info(f"Chunk ID: {scored_chunk.chunk.chunk_id}")
            logger.info(f"Relevance Score: {scored_chunk.relevance_score:.4f}")
            logger.info(f"Text: {scored_chunk.chunk.text[:100]}...")
        
        logger.info("\n" + "="*60)
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
