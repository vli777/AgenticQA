# backend/config.py

import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path, override=True)

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "test"

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