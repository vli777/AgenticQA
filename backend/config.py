# backend/config.py

import os
from dotenv import load_dotenv

# env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(override=True)

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "test"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

_cors_origins_raw = os.getenv("CORS_ORIGINS")
if _cors_origins_raw:
    CORS_ORIGINS = [origin.strip() for origin in _cors_origins_raw.split(",") if origin.strip()]
else:
    CORS_ORIGINS = ["*"]

# Read the full model name from the environment, e.g.:
#   EMBEDDING_MODEL="multilingual-e5-large"  or  "text-embedding-3-small"
_raw = os.getenv("EMBEDDING_MODEL", "").strip()

# Recognize exactly these full model names; otherwise, default to None
#  • "multilingual-e5-large" → local E5
#  • "text-embedding-3-small" → OpenAI
#  • "llama-text-embed-v2" → Pinecone’s LLaMA embedder
# If you want to add more in future (e.g. “something-else”), just include them here.
_SUPPORTED = {
    "multilingual-e5-large",
    "text-embedding-3-small",
    "llama-text-embed-v2",
}

if _raw in _SUPPORTED:
    EMBEDDING_MODEL = _raw
else:
    EMBEDDING_MODEL = None

# Hybrid search configuration
# HYBRID_SEARCH_ALPHA: Weight for BM25 vs vector search (0.0 = vector only, 1.0 = BM25 only, 0.5 = equal)
HYBRID_SEARCH_ALPHA = float(os.getenv("HYBRID_SEARCH_ALPHA", "0.5"))

# CROSS_ENCODER_MODEL: Model to use for re-ranking
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Number of results to retrieve before re-ranking
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "20"))

# Optimization features
# Enable caching for embeddings, LLM responses, and search results
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"

# Enable token streaming for LLM responses
ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "true").lower() == "true"

# Cache configuration
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "10000"))
EMBEDDING_CACHE_TTL = int(os.getenv("EMBEDDING_CACHE_TTL", "7200"))  # 2 hours

SEARCH_CACHE_SIZE = int(os.getenv("SEARCH_CACHE_SIZE", "1000"))
SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "1800"))  # 30 minutes

LLM_CACHE_SIZE = int(os.getenv("LLM_CACHE_SIZE", "500"))
LLM_CACHE_TTL = int(os.getenv("LLM_CACHE_TTL", "3600"))  # 1 hour
