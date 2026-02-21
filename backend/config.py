"""Configuration management for ClearPath RAG Chatbot."""
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Server Configuration
PORT = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# CORS Configuration
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", 
    "http://localhost:3000,http://localhost:3001"
).split(",")

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
SIMPLE_MODEL = "llama-3.1-8b-instant"
COMPLEX_MODEL = "llama-3.3-70b-versatile"

# Chunking Configuration
CHUNK_SIZE = 300  # tokens
CHUNK_OVERLAP = 50  # tokens
MAX_CHUNKS = 5

# Retrieval Configuration
RELEVANCE_THRESHOLD = 0.3
DYNAMIC_K_CUTOFF = 0.8  # Only include chunks within 80% of top score

# Logging Configuration
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
