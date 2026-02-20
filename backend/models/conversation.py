"""Conversation data models."""
from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass
class Turn:
    """Represents a single turn in a conversation."""
    query: str
    response: str
    timestamp: datetime

@dataclass
class Conversation:
    """Represents a multi-turn conversation."""
    conversation_id: str
    turns: List[Turn]
    created_at: datetime
