"""Document data models."""
from dataclasses import dataclass
from typing import List

@dataclass
class Page:
    """Represents a single page from a document."""
    page_number: int
    text: str
    word_count: int

@dataclass
class Document:
    """Represents a loaded PDF document."""
    filename: str
    pages: List[Page]
    total_pages: int

    @property
    def total_words(self) -> int:
        """Calculate the total word count across all pages."""
        return sum(page.word_count for page in self.pages)
