# ClearPath Chatbot Frontend

A clean, minimal Next.js chat interface for the ClearPath RAG Chatbot.

## Features

- Clean chat interface with message history
- Real-time conversation with the ClearPath support bot
- Conversation persistence across multiple questions
- Loading states and error handling
- Responsive design with Tailwind CSS

## Setup

1. Install dependencies:
```bash
npm install
```

2. Make sure the backend is running on `http://localhost:8000`

3. Start the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Usage

1. Type your question about ClearPath in the input field
2. Click "Send" or press Enter
3. The chatbot will respond with information from the ClearPath documentation
4. Continue the conversation - the bot remembers previous questions

## API Integration

The frontend connects to the backend API at `http://localhost:8000/query` with the following request format:

```json
{
  "question": "What is the Pro plan price?",
  "conversation_id": "optional-conversation-id"
}
```

The backend returns:

```json
{
  "answer": "The response text...",
  "metadata": {
    "model_used": "llama-3.1-8b-instant",
    "classification": "simple",
    "tokens": { "input": 234, "output": 45 },
    "latency_ms": 342,
    "chunks_retrieved": 2,
    "evaluator_flags": []
  },
  "sources": [...],
  "conversation_id": "conv_abc123"
}
```

## Build for Production

```bash
npm run build
npm start
```

## Technologies

- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
