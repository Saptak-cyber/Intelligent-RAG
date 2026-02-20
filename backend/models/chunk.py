"""Chunk data models."""
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

@dataclass
class Chunk:
    """Represents a document chunk for retrieval."""
    chunk_id: str  # Format: "{filename}_{page}_{chunk_index}"
    text: str
    document_name: str
    page_number: int
    embedding: Optional[np.ndarray] = None
    token_count: int = 0
    context_header: Optional[str] = None

@dataclass
class ScoredChunk:
    """Chunk with relevance score from retrieval."""
    chunk: Chunk
    relevance_score: float  # 0.0 to 1.0
