"""Demo script for ConversationManager."""
import sys
sys.path.insert(0, '.')

from services.conversation_manager import ConversationManager

def main():
    """Demo the ConversationManager functionality."""
    print("=== ConversationManager Demo ===\n")
    
    try:
        # Initialize the manager
        print("1. Initializing ConversationManager...")
        manager = ConversationManager()
        print("✓ ConversationManager initialized\n")
        
        # Create a new conversation
        print("2. Creating a new conversation...")
        conversation = manager.get_or_create_conversation()
        print(f"✓ Created conversation: {conversation.conversation_id}")
        print(f"  - Created at: {conversation.created_at}")
        print(f"  - Initial turns: {len(conversation.turns)}\n")
        
        # Add some turns
        print("3. Adding turns to the conversation...")
        manager.add_turn(
            conversation.conversation_id,
            "What is ClearPath?",
            "ClearPath is a project management tool that helps teams collaborate."
        )
        print("✓ Added turn 1")
        
        manager.add_turn(
            conversation.conversation_id,
            "What are the pricing plans?",
            "ClearPath offers Pro and Enterprise plans with different features."
        )
        print("✓ Added turn 2\n")
        
        # Retrieve the conversation
        print("4. Retrieving the conversation...")
        retrieved = manager.get_or_create_conversation(conversation.conversation_id)
        print(f"✓ Retrieved conversation: {retrieved.conversation_id}")
        print(f"  - Total turns: {len(retrieved.turns)}")
        for i, turn in enumerate(retrieved.turns, 1):
            print(f"  - Turn {i}:")
            print(f"    Q: {turn.query}")
            print(f"    A: {turn.response[:50]}...")
        print()
        
        # Get formatted context
        print("5. Getting formatted context...")
        context = manager.get_context(conversation.conversation_id, max_turns=2)
        print("✓ Context retrieved:")
        print(context)
        print()
        
        # Test max_turns limit
        print("6. Testing max_turns limit...")
        manager.add_turn(
            conversation.conversation_id,
            "How do I integrate with Slack?",
            "You can integrate ClearPath with Slack through the integrations page."
        )
        print("✓ Added turn 3")
        
        context_limited = manager.get_context(conversation.conversation_id, max_turns=2)
        print("✓ Context with max_turns=2 (should only show last 2 turns):")
        print(context_limited)
        print()
        
        # Create another conversation to test uniqueness
        print("7. Testing conversation ID uniqueness...")
        conversation2 = manager.get_or_create_conversation()
        print(f"✓ Created second conversation: {conversation2.conversation_id}")
        print(f"  - First ID:  {conversation.conversation_id}")
        print(f"  - Second ID: {conversation2.conversation_id}")
        print(f"  - Are different: {conversation.conversation_id != conversation2.conversation_id}\n")
        
        print("=== All tests passed! ===")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
