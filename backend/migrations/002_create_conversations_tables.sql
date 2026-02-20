-- Create conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id BIGSERIAL PRIMARY KEY,
    conversation_id TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on conversation_id for fast lookups
CREATE INDEX IF NOT EXISTS idx_conversation_id ON conversations(conversation_id);

-- Create turns table
CREATE TABLE IF NOT EXISTS turns (
    id BIGSERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
);

-- Create index on conversation_id for fast turn retrieval
CREATE INDEX IF NOT EXISTS idx_turns_conversation_id ON turns(conversation_id);

-- Create index on timestamp for ordering
CREATE INDEX IF NOT EXISTS idx_turns_timestamp ON turns(timestamp);
