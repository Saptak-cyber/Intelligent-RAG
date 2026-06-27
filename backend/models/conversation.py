"""Conversation data models."""
# NOTE: All fields are intentionally plain Python types to stay
# serialisation-agnostic (no external ORM dependency).
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

    @property
    def total_turns(self) -> int:
        """Returns the total number of turns in this conversation."""
        return len(self.turns)

    @property
    def last_updated_at(self) -> datetime:
        """Returns the timestamp of the most recent turn, or created_at if there are no turns."""
        if self.turns:
            return max(turn.timestamp for turn in self.turns)
        return self.created_at
