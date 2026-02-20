"""Main entry point for ClearPath RAG Chatbot API."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import PORT

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ClearPath RAG Chatbot",
    description="Customer support chatbot for ClearPath project management tool",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Next.js dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "ClearPath RAG Chatbot API"}

@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "service": "clearpath-rag-chatbot",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting ClearPath RAG Chatbot API on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
