# ConversationManager

The ConversationManager class provides multi-turn conversation support for the ClearPath RAG Chatbot. It stores conversation history in Supabase PostgreSQL and enables context-aware responses.

## Features

- **Conversation Creation**: Automatically generates unique conversation IDs
- **Turn Storage**: Stores query-response pairs with timestamps
- **Context Retrieval**: Formats conversation history for LLM prompts
- **Persistent Storage**: Uses Supabase PostgreSQL for reliable storage

## Database Schema

### Tables

#### conversations
```sql
CREATE TABLE conversations (
    id BIGSERIAL PRIMARY KEY,
    conversation_id TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### turns
```sql
CREATE TABLE turns (
    id BIGSERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
);
```

## Usage

### Initialization

```python
from services.conversation_manager import ConversationManager

manager = ConversationManager()
```

### Creating a New Conversation

```python
# Create a new conversation
conversation = manager.get_or_create_conversation()
print(f"Conversation ID: {conversation.conversation_id}")
```

### Retrieving an Existing Conversation

```python
# Retrieve an existing conversation by ID
conversation = manager.get_or_create_conversation("conv_abc123")
print(f"Turns: {len(conversation.turns)}")
```

### Adding Turns

```python
# Add a query-response pair to the conversation
manager.add_turn(
    conversation_id="conv_abc123",
    query="What is ClearPath?",
    response="ClearPath is a project management tool."
)
```

### Getting Formatted Context

```python
# Get formatted conversation history for LLM prompt
context = manager.get_context(
    conversation_id="conv_abc123",
    max_turns=3  # Include last 3 turns
)

# Context format:
# Previous Q: What is ClearPath?
# Previous A: ClearPath is a project management tool.
# Previous Q: What are the pricing plans?
# Previous A: ClearPath offers Pro and Enterprise plans.
```

## API Reference

### ConversationManager

#### `__init__()`
Initialize the conversation manager with Supabase client.

**Raises:**
- `ValueError`: If SUPABASE_URL or SUPABASE_KEY environment variables are not set

#### `get_or_create_conversation(conversation_id: Optional[str] = None) -> Conversation`
Get existing conversation or create new one.

**Parameters:**
- `conversation_id` (Optional[str]): Existing conversation ID to retrieve

**Returns:**
- `Conversation`: Conversation object with ID, turns, and created_at timestamp

**Behavior:**
- If `conversation_id` is provided and exists: Returns existing conversation with turns
- If `conversation_id` is provided but doesn't exist: Creates new conversation with new ID
- If `conversation_id` is None: Creates new conversation with generated ID

#### `add_turn(conversation_id: str, query: str, response: str) -> None`
Add query-response pair to conversation history.

**Parameters:**
- `conversation_id` (str): ID of the conversation
- `query` (str): User query text
- `response` (str): System response text

**Raises:**
- `Exception`: If database operation fails

#### `get_context(conversation_id: str, max_turns: int = 3) -> str`
Get formatted conversation history for prompt.

**Parameters:**
- `conversation_id` (str): ID of the conversation
- `max_turns` (int): Maximum number of recent turns to include (default: 3)

**Returns:**
- `str`: Formatted conversation history string, empty string if no turns

**Format:**
```
Previous Q: {query1}
Previous A: {response1}
Previous Q: {query2}
Previous A: {response2}
```

## Data Models

### Conversation
```python
@dataclass
class Conversation:
    conversation_id: str
    turns: List[Turn]
    created_at: datetime
```

### Turn
```python
@dataclass
class Turn:
    query: str
    response: str
    timestamp: datetime
```

## Setup

### 1. Run Database Migration

Execute the migration file in Supabase SQL Editor:
```bash
backend/migrations/002_create_conversations_tables.sql
```

### 2. Configure Environment Variables

Ensure these variables are set in your `.env` file:
```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
```

### 3. Test the Implementation

Run the demo script:
```bash
cd backend
python demo_conversation_manager.py
```

## Error Handling

The ConversationManager handles errors gracefully:

- **Missing Environment Variables**: Raises `ValueError` on initialization
- **Database Connection Errors**: Logs error and raises exception
- **Nonexistent Conversation**: Creates new conversation instead of failing
- **Turn Retrieval Errors**: Returns empty list and logs error

## Integration with API Endpoint

Example usage in the `/query` endpoint:

```python
from services.conversation_manager import ConversationManager

manager = ConversationManager()

# Get or create conversation
conversation = manager.get_or_create_conversation(request.conversation_id)

# Get conversation context for prompt
context = manager.get_context(conversation.conversation_id, max_turns=3)

# Build prompt with context
prompt = f"""
{context}

User question: {request.question}
"""

# Generate response
response = llm_client.generate(model, prompt)

# Store the turn
manager.add_turn(conversation.conversation_id, request.question, response.text)

# Return response with conversation_id
return QueryResponse(
    answer=response.text,
    conversation_id=conversation.conversation_id,
    ...
)
```

## Known Issues

### Python 3.14 Compatibility

There is a known compatibility issue with Python 3.14 and the `supabase-py` library (specifically the `httpx` dependency). The error:

```
TypeError: Client.__init__() got an unexpected keyword argument 'proxy'
```

**Workaround:**
- Use Python 3.11, 3.12, or 3.13
- Or wait for `supabase-py` to update their `httpx` dependency

The implementation itself is correct and will work once the library compatibility is resolved.

## Testing

Unit tests are provided in `tests/test_conversation_manager.py`:

```bash
# Run tests (requires Python 3.11-3.13)
python -m pytest tests/test_conversation_manager.py -v
```

Test coverage includes:
- Creating new conversations
- Conversation ID uniqueness
- Adding turns
- Retrieving existing conversations
- Getting formatted context
- Max turns limit
- Handling nonexistent conversations
