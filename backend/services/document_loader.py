"""Document loading service for PDF processing."""
import logging
import os
from typing import List
import fitz  # PyMuPDF

from models.document import Document, Page

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Loads and extracts text from PDF files."""
    
    def __init__(self, docs_directory: str = "clearpath_docs"):
        """
        Initialize DocumentLoader.
        
        Args:
            docs_directory: Path to directory containing PDF files
        """
        self.docs_directory = docs_directory
    
    def load_documents(self) -> List[Document]:
        """
        Load all PDF files from the documents directory.
        
        Returns:
            List of Document objects with text, filename, and page numbers
        """
        documents = []
        
        if not os.path.exists(self.docs_directory):
            logger.error(f"Documents directory not found: {self.docs_directory}")
            return documents
        
        # Get all PDF files
        pdf_files = [f for f in os.listdir(self.docs_directory) if f.endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files in {self.docs_directory}")
        
        for filename in sorted(pdf_files):
            filepath = os.path.join(self.docs_directory, filename)
            
            try:
                document = self._load_pdf(filepath, filename)
                if document:
                    documents.append(document)
                    logger.info(f"Loaded {filename}: {document.total_pages} pages")
            except Exception as e:
                logger.error(f"Error loading {filename}: {str(e)}", exc_info=True)
                # Skip corrupted file and continue
                continue
        
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def _load_pdf(self, filepath: str, filename: str) -> Document:
        """
        Load a single PDF file and extract text page-by-page.
        
        Args:
            filepath: Full path to PDF file
            filename: Name of the file
            
        Returns:
            Document object with extracted text
        """
        try:
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(filepath)
            pages = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text = page.get_text()
                
                # Count words
                word_count = len(text.split())
                
                pages.append(Page(
                    page_number=page_num + 1,  # 1-indexed
                    text=text,
                    word_count=word_count
                ))
            
            pdf_document.close()
            
            return Document(
                filename=filename,
                pages=pages,
                total_pages=len(pages)
            )
        
        except Exception as e:
            logger.error(f"Failed to load PDF {filename}: {str(e)}")
            raise
