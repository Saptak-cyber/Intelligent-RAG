"""Test script for ChunkingEngine."""
import sys
sys.path.insert(0, 'backend')

# Import directly to avoid __init__.py issues
from models.document import Document, Page
from models.chunk import Chunk

def test_chunk_model():
    """Test the Chunk dataclass."""
    print("Testing Chunk model...")
    
    # Create a test chunk
    chunk = Chunk(
        chunk_id="test_doc_1_0",
        text="[Context: Introduction] This is a test chunk with context.",
        document_name="test_doc.pdf",
        page_number=1,
        token_count=10,
        context_header="Introduction"
    )
    
    print(f"✓ Chunk created successfully")
    print(f"  ID: {chunk.chunk_id}")
    print(f"  Document: {chunk.document_name}")
    print(f"  Page: {chunk.page_number}")
    print(f"  Token count: {chunk.token_count}")
    print(f"  Context header: {chunk.context_header}")
    print(f"  Text: {chunk.text}")
    
    # Verify all required fields are present
    assert chunk.chunk_id, "chunk_id is required"
    assert chunk.text, "text is required"
    assert chunk.document_name, "document_name is required"
    assert chunk.page_number, "page_number is required"
    
    print("\n✓ All required fields present")
    print("✓ Chunk model test passed!")

if __name__ == "__main__":
    test_chunk_model()
