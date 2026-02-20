"""Unit tests for ConversationManager."""
import sys
sys.path.insert(0, 'backend')

import pytest
from datetime import datetime
from services.conversation_manager import ConversationManager
from models.conversation import Conversation, Turn


class TestConversationManager:
    """Test suite for ConversationManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a ConversationManager instance."""
        return ConversationManager()
    
    def test_create_new_conversation(self, manager):
        """Test creating a new conversation."""
        conversation = manager.get_or_create_conversation()
        
        assert conversation is not None
        assert conversation.conversation_id.startswith("conv_")
        assert len(conversation.turns) == 0
        assert isinstance(conversation.created_at, datetime)
    
    def test_conversation_id_uniqueness(self, manager):
        """Test that new conversations have unique IDs."""
        conv1 = manager.get_or_create_conversation()
        conv2 = manager.get_or_create_conversation()
        
        assert conv1.conversation_id != conv2.conversation_id
    
    def test_add_turn(self, manager):
        """Test adding a turn to a conversation."""
        conversation = manager.get_or_create_conversation()
        
        query = "What is ClearPath?"
        response = "ClearPath is a project management tool."
        
        # Should not raise an exception
        manager.add_turn(conversation.conversation_id, query, response)
    
    def test_get_existing_conversation(self, manager):
        """Test retrieving an existing conversation."""
        # Create a conversation and add a turn
        conv1 = manager.get_or_create_conversation()
        manager.add_turn(conv1.conversation_id, "Test query", "Test response")
        
        # Retrieve the same conversation
        conv2 = manager.get_or_create_conversation(conv1.conversation_id)
        
        assert conv2.conversation_id == conv1.conversation_id
        assert len(conv2.turns) == 1
        assert conv2.turns[0].query == "Test query"
        assert conv2.turns[0].response == "Test response"
    
    def test_get_context_empty(self, manager):
        """Test getting context from a conversation with no turns."""
        conversation = manager.get_or_create_conversation()
        context = manager.get_context(conversation.conversation_id)
        
        assert context == ""
    
    def test_get_context_with_turns(self, manager):
        """Test getting formatted context from a conversation with turns."""
        conversation = manager.get_or_create_conversation()
        
        # Add multiple turns
        manager.add_turn(conversation.conversation_id, "Query 1", "Response 1")
        manager.add_turn(conversation.conversation_id, "Query 2", "Response 2")
        
        context = manager.get_context(conversation.conversation_id)
        
        assert "Previous Q: Query 1" in context
        assert "Previous A: Response 1" in context
        assert "Previous Q: Query 2" in context
        assert "Previous A: Response 2" in context
    
    def test_get_context_max_turns(self, manager):
        """Test that get_context respects max_turns limit."""
        conversation = manager.get_or_create_conversation()
        
        # Add 5 turns
        for i in range(5):
            manager.add_turn(conversation.conversation_id, f"Query {i+1}", f"Response {i+1}")
        
        # Get context with max_turns=3
        context = manager.get_context(conversation.conversation_id, max_turns=3)
        
        # Should only include the last 3 turns (3, 4, 5)
        assert "Query 3" in context
        assert "Query 4" in context
        assert "Query 5" in context
        assert "Query 1" not in context
        assert "Query 2" not in context
    
    def test_get_nonexistent_conversation(self, manager):
        """Test that requesting a nonexistent conversation creates a new one."""
        fake_id = "conv_nonexistent123"
        conversation = manager.get_or_create_conversation(fake_id)
        
        # Should create a new conversation with a different ID
        assert conversation.conversation_id != fake_id
        assert conversation.conversation_id.startswith("conv_")
        assert len(conversation.turns) == 0
