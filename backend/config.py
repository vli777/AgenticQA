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
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

_cors_origins_raw = os.getenv("CORS_ORIGINS")
if _cors_origins_raw:
    CORS_ORIGINS = [origin.strip() for origin in _cors_origins_raw.split(",") if origin.strip()]
else:
    CORS_ORIGINS = ["*"]

# Ensure local dev servers can hit the API even when custom origins are provided
_LOCAL_DEV_ORIGINS = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
}
if "*" not in CORS_ORIGINS:
    for origin in _LOCAL_DEV_ORIGINS:
        if origin not in CORS_ORIGINS:
            CORS_ORIGINS.append(origin)

# Read the full model name from the environment, e.g.:
#   EMBEDDING_MODEL="nvidia-embed"  or  "text-embedding-3-small"
_raw = os.getenv("EMBEDDING_MODEL", "").strip()

# Recognize exactly these full model names; otherwise, default to nvidia/nv-embedqa-e5-v5
#  • "nvidia/nv-embedqa-e5-v5" → NVIDIA E5 (1024-dim, recommended for existing indexes)
#  • "nvidia/nv-embed-v1" → NVIDIA nv-embed-v1 (4096-dim, high quality)
#  • "nvidia/embed-qa-4" → NVIDIA embedding-qa-4 (Q&A optimized)
#  • "text-embedding-3-small" → OpenAI (1536-dim)
# If you want to add more in future, just include them here.
_SUPPORTED = {
    "nvidia/nv-embedqa-e5-v5",
    "nvidia/nv-embed-v1",
    "nvidia/embed-qa-4",
    "nvidia/embedding-qa-4",
    "text-embedding-3-small",
}

if _raw in _SUPPORTED:
    EMBEDDING_MODEL = _raw
else:
    EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"  # Default: 1024-dim, compatible with existing indexes

# Semantic tagging configuration (Hugging Face zero-shot + heuristics)
ENABLE_SEMANTIC_TAGGING = os.getenv("ENABLE_SEMANTIC_TAGGING", "true").lower() == "true"
HUGGINGFACE_ZS_MODEL = os.getenv("HUGGINGFACE_ZS_MODEL", "facebook/bart-large-mnli")

_semantic_labels_raw = os.getenv("SEMANTIC_TAG_LABELS")
if _semantic_labels_raw:
    SEMANTIC_TAG_LABELS = [label.strip().lower() for label in _semantic_labels_raw.split(",") if label.strip()]
else:
    SEMANTIC_TAG_LABELS = [
        "python",
        "java",
        "javascript",
        "typescript",
        "c++",
        "c#",
        "go",
        "rust",
        "sql",
        "data engineering",
        "machine learning",
        "devops",
    ]

SEMANTIC_TAG_THRESHOLD = float(os.getenv("SEMANTIC_TAG_THRESHOLD", "0.6"))
SEMANTIC_TAG_BOOST = float(os.getenv("SEMANTIC_TAG_BOOST", "0.2"))

# Hybrid search configuration (Best Practice Pipeline)
# Keep these internal to avoid accidental overrides
BM25_K = 30
VECTOR_K = 30

# CROSS_ENCODER_MODEL: Model to use for re-ranking
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/stsb-roberta-base")

# Optimization features (internal defaults)
ENABLE_CACHING = True
ENABLE_STREAMING = True

# Cache configuration (internal defaults)
EMBEDDING_CACHE_SIZE = 10000
EMBEDDING_CACHE_TTL = 7200  # 2 hours

SEARCH_CACHE_SIZE = 1000
SEARCH_CACHE_TTL = 1800  # 30 minutes

LLM_CACHE_SIZE = 500
LLM_CACHE_TTL = 3600  # 1 hour

# Chunking configuration (for better context preservation)
CHUNK_SIZE = 2000  # Characters per chunk (~500 tokens)
CHUNK_OVERLAP = 400  # ~20% overlap to preserve context
