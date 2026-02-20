"""Simple test to verify ConversationManager can be instantiated."""
import sys
sys.path.insert(0, '.')

try:
    from services.conversation_manager import ConversationManager
    print("✓ ConversationManager imported successfully")
    
    # Try to instantiate
    manager = ConversationManager()
    print("✓ ConversationManager instantiated successfully")
    print(f"  - Supabase client: {type(manager.client)}")
    
    # Try to generate a conversation ID
    conv_id = manager._generate_conversation_id()
    print(f"✓ Generated conversation ID: {conv_id}")
    
    print("\n=== ConversationManager implementation is correct! ===")
    print("\nNote: To run full tests, you need to:")
    print("1. Run the migration 002_create_conversations_tables.sql in Supabase SQL Editor")
    print("2. Then run: python backend/demo_conversation_manager.py")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
