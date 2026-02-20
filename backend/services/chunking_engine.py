"""Chunking engine with contextual heading injection."""
import logging
import re
from typing import List, Dict, Tuple, Optional
import fitz  # PyMuPDF
from transformers import AutoTokenizer

from models.document import Document
from models.chunk import Chunk
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class ChunkingEngine:
    """Segments documents into retrievable chunks with contextual headers."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize ChunkingEngine.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize tokenizer for embeddings (all-mpnet-base-v2)
        logger.info("Loading tokenizer for chunking...")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        
        # Separators for recursive splitting (in priority order)
        self.separators = ["\n\n", "\n", ". ", " ", ""]
    
    def chunk_documents(self, documents: List[Document], docs_directory: Optional[str] = None) -> List[Chunk]:
        """
        Chunk documents using token-aware recursive splitting with contextual heading injection.
        
        Args:
            documents: List of loaded documents
            docs_directory: Path to PDF files for header extraction (optional, uses document.source_path if not provided)
            
        Returns:
            List of Chunk objects with text, metadata, and context headers
        """
        all_chunks = []
        
        for document in documents:
            logger.info(f"Chunking document: {document.filename}")
            
            # Determine PDF path for header extraction
            if docs_directory:
                pdf_path = f"{docs_directory}/{document.filename}"
            else:
                # Use source_path from document if available
                pdf_path = getattr(document, 'source_path', None)
                if not pdf_path:
                    # Fallback to relative path
                    from pathlib import Path
                    pdf_path = str(Path(__file__).parent.parent / "clearpath_docs" / document.filename)
            
            # Extract hierarchical headers from PDF
            header_stack_by_page = self._extract_headers(pdf_path)
            
            # Maintain header stack state across pages
            current_header_stack = []
            
            for page in document.pages:
                # Update header stack if new headers found on this page
                if page.page_number in header_stack_by_page:
                    current_header_stack = header_stack_by_page[page.page_number]
                
                # Chunk this page's text
                page_chunks = self._chunk_text(
                    text=page.text,
                    document_name=document.filename,
                    page_number=page.page_number,
                    header_stack=current_header_stack
                )
                
                all_chunks.extend(page_chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def _extract_headers(self, pdf_path: str) -> Dict[int, List[str]]:
        """
        Extract hierarchical headers from PDF using font size detection.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary mapping page number to header stack
        """
        header_stack_by_page = {}
        current_stack = []  # List of (text, level) tuples
        
        try:
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Get text with formatting info
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            font_size = span["size"]
                            
                            # Detect headers by font size (> 12pt)
                            if font_size > 12 and text and len(text) > 2:
                                # Determine hierarchy by font size
                                if font_size > 18:
                                    level = 1  # H1
                                elif font_size > 14:
                                    level = 2  # H2
                                else:
                                    level = 3  # H3
                                
                                # Update header stack based on hierarchy
                                # When we see a header of level N, remove all headers of level >= N
                                current_stack = [(h, l) for h, l in current_stack if l < level]
                                current_stack.append((text, level))
                
                # Store header stack for this page (extract just the text)
                if current_stack:
                    header_stack_by_page[page_num + 1] = [h for h, l in current_stack]
            
            pdf_document.close()
        
        except Exception as e:
            logger.warning(f"Could not extract headers from {pdf_path}: {str(e)}")
        
        return header_stack_by_page
    
    def _chunk_text(
        self,
        text: str,
        document_name: str,
        page_number: int,
        header_stack: List[str]
    ) -> List[Chunk]:
        """
        Chunk text using recursive splitting with header injection.
        
        Args:
            text: Text to chunk
            document_name: Source document filename
            page_number: Source page number
            header_stack: Current hierarchical header stack
            
        Returns:
            List of chunks for this text
        """
        if not text.strip():
            return []
        
        # Build context header
        context_header = None
        if header_stack:
            context_header = " > ".join(header_stack)
        
        # Split text recursively
        chunks_text = self._recursive_split(text, self.chunk_size, self.chunk_overlap)
        
        # Create Chunk objects
        chunks = []
        for idx, chunk_text in enumerate(chunks_text):
            # Prepend context header
            if context_header:
                full_text = f"[Context: {context_header}] {chunk_text}"
            else:
                full_text = chunk_text
            
            # Count tokens
            token_count = len(self.tokenizer.encode(full_text))
            
            chunk = Chunk(
                chunk_id=f"{document_name}_{page_number}_{idx}",
                text=full_text,
                document_name=document_name,
                page_number=page_number,
                token_count=token_count,
                context_header=context_header
            )
            chunks.append(chunk)
        
        return chunks
    
    def _recursive_split(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Recursively split text using separators.
        
        Args:
            text: Text to split
            chunk_size: Target chunk size in tokens
            overlap: Overlap size in tokens
            
        Returns:
            List of text chunks
        """
        # Try each separator in order
        for separator in self.separators:
            if separator in text:
                parts = text.split(separator)
                
                # Rebuild chunks respecting token limits
                chunks = []
                current_chunk = ""
                
                for part in parts:
                    # Add separator back (except for empty string separator)
                    if separator:
                        part = part + separator
                    
                    # Check if adding this part exceeds chunk size
                    test_chunk = current_chunk + part
                    token_count = len(self.tokenizer.encode(test_chunk))
                    
                    if token_count <= chunk_size:
                        current_chunk = test_chunk
                    else:
                        # Save current chunk if not empty
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        
                        # Start new chunk with overlap
                        if chunks and overlap > 0:
                            # Get last N tokens from previous chunk
                            prev_tokens = self.tokenizer.encode(chunks[-1])
                            if len(prev_tokens) > overlap:
                                overlap_tokens = prev_tokens[-overlap:]
                                overlap_text = self.tokenizer.decode(overlap_tokens)
                                current_chunk = overlap_text + part
                            else:
                                current_chunk = part
                        else:
                            current_chunk = part
                
                # Add final chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                return [c for c in chunks if c]
        
        # If no separator worked, return as single chunk
        return [text]
