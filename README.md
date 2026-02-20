# ClearPath RAG Chatbot

A customer support chatbot for ClearPath (a fictional SaaS project management tool) that answers user questions by retrieving relevant content from 30 PDF documentation files and generating responses using LLMs via Groq API.

## Architecture

The system implements a three-layer architecture:

1. **RAG Pipeline**: Retrieves relevant document chunks from 30 PDF files using vector similarity search
2. **Model Router**: Classifies queries and routes them to appropriate LLM models using deterministic rules
3. **Output Evaluator**: Analyzes responses and flags potential quality issues

## Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Next.js (TypeScript)
- **PDF Processing**: PyMuPDF
- **Vector Search**: Supabase pgvector
- **Embeddings**: Hugging Face Inference API (all-mpnet-base-v2)
- **LLM**: Groq API (llama-3.1-8b-instant, llama-3.3-70b-versatile)
- **Storage**: Supabase PostgreSQL

## Setup Instructions

### Prerequisites

- Python 3.10+
- Node.js 18+
- Supabase account (free tier)
- Groq API key (free tier)
- Hugging Face API key (free tier)

### Backend Setup

1. **Create virtual environment and install dependencies:**

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Set up environment variables:**

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
PORT=8000
LOG_LEVEL=INFO
```

3. **Set up Supabase:**

- Create a new Supabase project at https://supabase.com
- Enable the pgvector extension in your database
- Copy your project URL and anon key to `.env`

4. **Run the backend:**

```bash
cd backend
source venv/bin/activate
python main.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Install dependencies:**

```bash
cd frontend
npm install
```

2. **Run the development server:**

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## API Endpoints

### POST /query

Accepts a user question and returns the chatbot's response along with metadata.

**Request:**
```json
{
  "question": "What is the price of the Pro plan?",
  "conversation_id": "optional-conversation-id-for-multi-turn"
}
```

**Response:**
```json
{
  "answer": "The Pro plan pricing appears in the documentation...",
  "metadata": {
    "model_used": "llama-3.3-70b-versatile",
    "classification": "complex",
    "tokens": {
      "input": 1234,
      "output": 156
    },
    "latency_ms": 847,
    "chunks_retrieved": 3,
    "evaluator_flags": []
  },
  "sources": [
    {
      "document": "14_Pricing_Sheet_2024.pdf",
      "page": 1,
      "relevance_score": 0.92
    }
  ],
  "conversation_id": "conv_abc123"
}
```

## Model Routing Rules

The router uses deterministic, rule-based classification:

- **Rule 0 (OOD Filter)**: Greetings/meta-comments → Simple + skip retrieval
- **Rule 1**: Contains complex keywords ("why", "how", "explain", "compare") → Complex
- **Rule 2**: Query length > 15 words → Complex
- **Rule 3**: Multiple question marks (>1) → Complex
- **Rule 4**: Contains comparison words ("versus", "vs", "better") → Complex
- **Rule 5**: Default → Simple

Simple queries route to `llama-3.1-8b-instant`, complex queries to `llama-3.3-70b-versatile`.

## Evaluator Flags

The system flags potentially unreliable outputs:

1. **`no_context`**: LLM answered but no relevant chunks were retrieved
2. **`refusal`**: LLM explicitly refused to answer or said it doesn't know
3. **`unverified_feature`**: LLM mentions features/integrations not in retrieved chunks
4. **`pricing_uncertainty`**: Pricing query with hedging language or conflicting sources

## Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Configuration management
│   ├── logger.py            # Structured logging
│   ├── models/              # Data models
│   ├── services/            # Business logic
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── pages/               # Next.js pages
│   ├── components/          # React components
│   └── package.json         # Node dependencies
├── tests/                   # Test files
├── logs/                    # Log files
├── clearpath_docs/          # 30 PDF documentation files
└── .env.example             # Environment variables template
```

## Development

### Running Tests

```bash
cd backend
source venv/bin/activate
pytest tests/
```

### Running with Docker (Optional)

```bash
docker-compose up
```

## Known Issues and Limitations

1. **Stateless Router**: The router doesn't consider conversation history, which can cause misclassification in multi-turn conversations
2. **HF API Cold Start**: First query may take 15-20s due to model loading on free tier
3. **Groundedness Check**: Catches hallucinations but doesn't prevent "helpful" LLM from inventing features

## Deployment

### Local Deployment

Follow the setup instructions above. The API runs on `localhost:8000` by default.

### Cloud Deployment

**Backend** (Railway/Render):
- Deploy backend as a Python web service
- Set environment variables in platform dashboard
- Ensure Supabase is accessible from deployment

**Frontend** (Vercel):
- Deploy frontend as a Next.js application
- Set API URL environment variable

**Bonus: AWS/GCP Deployment**
- Use AWS Lambda + API Gateway or GCP Cloud Run
- Configure environment variables in cloud console
- Set up VPC for Supabase connection if needed

## License

MIT

## Contact

For questions or issues, please open a GitHub issue.
