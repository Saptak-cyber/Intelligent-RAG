"""Conversation manager for multi-turn conversation support."""
import logging
import uuid
from datetime import datetime
from typing import Optional, List
from supabase import create_client, Client

from models.conversation import Conversation, Turn
from config import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation storage and retrieval using Supabase PostgreSQL."""
    
    def __init__(self):
        """Initialize the conversation manager with Supabase client."""
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("ConversationManager initialized with Supabase")
    
    def get_or_create_conversation(self, conversation_id: Optional[str] = None) -> Conversation:
        """
        Get existing conversation or create new one.
        
        Args:
            conversation_id: Optional existing conversation ID
            
        Returns:
            Conversation object with ID and turns
        """
        if conversation_id:
            # Try to retrieve existing conversation
            try:
                result = self.client.table("conversations").select("*").eq("conversation_id", conversation_id).execute()
                
                if result.data and len(result.data) > 0:
                    # Conversation exists, retrieve turns
                    conv_data = result.data[0]
                    turns = self._get_turns(conversation_id)
                    
                    logger.info(f"Retrieved existing conversation: {conversation_id} with {len(turns)} turns")
                    return Conversation(
                        conversation_id=conv_data["conversation_id"],
                        turns=turns,
                        created_at=datetime.fromisoformat(conv_data["created_at"].replace("Z", "+00:00"))
                    )
                else:
                    logger.warning(f"Conversation {conversation_id} not found, creating new one")
            except Exception as e:
                logger.error(f"Error retrieving conversation {conversation_id}: {e}")
                # Fall through to create new conversation
        
        # Create new conversation
        new_id = self._generate_conversation_id()
        created_at = datetime.now()
        
        try:
            self.client.table("conversations").insert({
                "conversation_id": new_id,
                "created_at": created_at.isoformat()
            }).execute()
            
            logger.info(f"Created new conversation: {new_id}")
            return Conversation(
                conversation_id=new_id,
                turns=[],
                created_at=created_at
            )
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            raise
    
    def add_turn(self, conversation_id: str, query: str, response: str) -> None:
        """
        Add query-response pair to conversation history.
        
        Args:
            conversation_id: ID of the conversation
            query: User query
            response: System response
        """
        timestamp = datetime.now()
        
        try:
            self.client.table("turns").insert({
                "conversation_id": conversation_id,
                "query": query,
                "response": response,
                "timestamp": timestamp.isoformat()
            }).execute()
            
            logger.info(f"Added turn to conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Error adding turn to conversation {conversation_id}: {e}")
            raise
    
    def get_context(self, conversation_id: str, max_turns: int = 3) -> str:
        """
        Get formatted conversation history for prompt.
        
        Args:
            conversation_id: ID of the conversation
            max_turns: Maximum number of recent turns to include
            
        Returns:
            Formatted conversation history string
        """
        turns = self._get_turns(conversation_id, limit=max_turns)
        
        if not turns:
            return ""
        
        # Format turns as conversation history
        context_parts = []
        for turn in turns:
            context_parts.append(f"Previous Q: {turn.query}")
            context_parts.append(f"Previous A: {turn.response}")
        
        context = "\n".join(context_parts)
        logger.debug(f"Retrieved context for conversation {conversation_id}: {len(turns)} turns")
        
        return context
    
    def _get_turns(self, conversation_id: str, limit: Optional[int] = None) -> List[Turn]:
        """
        Retrieve turns for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            limit: Optional limit on number of turns to retrieve (most recent)
            
        Returns:
            List of Turn objects ordered by timestamp
        """
        try:
            query = self.client.table("turns").select("*").eq("conversation_id", conversation_id).order("timestamp", desc=False)
            
            if limit:
                # Get all turns first, then take the last N
                result = query.execute()
                if result.data:
                    # Take the last N turns
                    turn_data = result.data[-limit:] if len(result.data) > limit else result.data
                else:
                    turn_data = []
            else:
                result = query.execute()
                turn_data = result.data if result.data else []
            
            turns = [
                Turn(
                    query=t["query"],
                    response=t["response"],
                    timestamp=datetime.fromisoformat(t["timestamp"].replace("Z", "+00:00"))
                )
                for t in turn_data
            ]
            
            return turns
        except Exception as e:
            logger.error(f"Error retrieving turns for conversation {conversation_id}: {e}")
            return []
    
    def _generate_conversation_id(self) -> str:
        """
        Generate a unique conversation ID.
        
        Returns:
            Unique conversation ID string
        """
        return f"conv_{uuid.uuid4().hex[:12]}"
