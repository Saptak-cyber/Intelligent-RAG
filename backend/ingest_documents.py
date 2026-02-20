"""
Document Ingestion Script for ClearPath RAG Chatbot.

This script:
1. Clears existing data from Supabase
2. Loads all PDFs from clearpath_docs/
3. Chunks documents with contextual headers
4. Generates embeddings using HuggingFace API
5. Stores everything in Supabase pgvector

Usage:
    python ingest_documents.py
"""
import sys
import logging
from pathlib import Path
from typing import List

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from services.document_loader import DocumentLoader
from services.chunking_engine import ChunkingEngine
from services.embedding_model import EmbeddingModel
from services.vector_store import VectorStore
from models.chunk import Chunk
from config import HUGGINGFACE_API_KEY, SUPABASE_URL, SUPABASE_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_existing_data(vector_store: VectorStore) -> None:
    """
    Clear all existing chunks from the database.
    
    Args:
        vector_store: VectorStore instance
    """
    try:
        logger.info("Clearing existing data from database...")
        count_before = vector_store.count()
        logger.info(f"Found {count_before} existing chunks")
        
        if count_before > 0:
            # Delete all chunks
            vector_store.client.table('document_chunks').delete().neq('chunk_id', '').execute()
            count_after = vector_store.count()
            logger.info(f"Cleared {count_before - count_after} chunks")
        else:
            logger.info("No existing data to clear")
            
    except Exception as e:
        logger.error(f"Error clearing existing data: {e}")
        raise


def main():
    """Main ingestion process."""
    try:
        logger.info("="*60)
        logger.info("Starting ClearPath Document Ingestion")
        logger.info("="*60)
        
        # Step 1: Initialize services
        logger.info("\n[1/6] Initializing services...")
        
        embedding_model = EmbeddingModel(api_key=HUGGINGFACE_API_KEY)
        logger.info("✓ Embedding model initialized")
        
        vector_store = VectorStore(
            embedding_model=embedding_model,
            supabase_url=SUPABASE_URL,
            supabase_key=SUPABASE_KEY
        )
        logger.info("✓ Vector store initialized")
        
        # Determine correct path to clearpath_docs
        script_dir = Path(__file__).parent
        docs_path = script_dir.parent / "clearpath_docs"
        
        document_loader = DocumentLoader(docs_directory=str(docs_path))
        logger.info("✓ Document loader initialized")
        
        chunking_engine = ChunkingEngine()
        logger.info("✓ Chunking engine initialized")
        
        # Step 2: Clear existing data
        logger.info("\n[2/6] Clearing existing data...")
        clear_existing_data(vector_store)
        
        # Step 3: Warm up embedding model
        logger.info("\n[3/6] Warming up embedding model...")
        logger.info("This may take 15-20 seconds on first run (HuggingFace free tier)...")
        embedding_model.warmup()
        logger.info("✓ Model warmed up and ready")
        
        # Step 4: Load documents
        logger.info("\n[4/6] Loading PDF documents...")
        documents = document_loader.load_documents()
        logger.info(f"✓ Loaded {len(documents)} documents")
        
        if len(documents) == 0:
            logger.error("No documents found! Check that clearpath_docs/ directory exists and contains PDFs")
            sys.exit(1)
        
        # Log document names
        for doc in documents:
            logger.info(f"  - {doc.filename} ({len(doc.pages)} pages)")
        
        # Step 5: Chunk documents
        logger.info("\n[5/6] Chunking documents with contextual headers...")
        all_chunks: List[Chunk] = []
        
        for doc in documents:
            logger.info(f"Processing {doc.filename}...")
            chunks = chunking_engine.chunk_documents([doc])
            all_chunks.extend(chunks)
            logger.info(f"  ✓ Created {len(chunks)} chunks")
        
        logger.info(f"✓ Total chunks created: {len(all_chunks)}")
        
        # Step 6: Generate embeddings and store in Supabase
        logger.info("\n[6/6] Generating embeddings and storing in Supabase...")
        logger.info("This will take several minutes depending on the number of chunks...")
        logger.info(f"Processing {len(all_chunks)} chunks in batches...")
        
        # Process in batches to show progress
        batch_size = 10
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            vector_store.add_chunks(batch)
            logger.info(f"  ✓ Batch {batch_num} completed")
        
        # Verify final count
        final_count = vector_store.count()
        logger.info(f"✓ Successfully stored {final_count} chunks in database")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("INGESTION COMPLETE!")
        logger.info("="*60)
        logger.info(f"Documents processed: {len(documents)}")
        logger.info(f"Total chunks created: {len(all_chunks)}")
        logger.info(f"Chunks in database: {final_count}")
        logger.info("\nYour ClearPath RAG Chatbot is now ready to use!")
        logger.info("Start the API with: python main.py")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.warning("\nIngestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nIngestion failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
