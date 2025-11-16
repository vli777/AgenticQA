# backend/pinecone_client.py

from pinecone import Pinecone, ServerlessSpec

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_MODEL

_DEFAULT_DIMENSION = 1536
_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "nvidia/llama-3.2-nv-embedqa-1b-v2": 1024,
    "nvidia/nv-embedqa-e5-v5": 1024,
    "nvidia/nv-embed-v1": 4096,
    "nvidia/embed-qa-4": 1024,
    "nvidia/embedding-qa-4": 1024,
    "nvidia-embed": 1024,
}
_PINECONE_HOSTED_MODELS = {
    "llama-text-embed-v2",
    "multilingual-e5-large",
    "pinecone-sparse-english-v0",
}


def _get_dimension(model_name: str | None) -> int:
    if not model_name:
        return _DEFAULT_DIMENSION
    return _MODEL_DIMENSIONS.get(model_name, _DEFAULT_DIMENSION)


pc = Pinecone(api_key=PINECONE_API_KEY)

if not pc.has_index(PINECONE_INDEX_NAME):
    if EMBEDDING_MODEL in _PINECONE_HOSTED_MODELS:
        pc.create_index_for_model(
            name=PINECONE_INDEX_NAME,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            embed={
                "model": EMBEDDING_MODEL,
                "field_map": {"text": "text"},
            },
        )
    else:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=_get_dimension(EMBEDDING_MODEL),
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

index = pc.Index(PINECONE_INDEX_NAME)

# Validate index dimension matches embedding model
try:
    index_stats = index.describe_index_stats()
    index_dim = index_stats.get("dimension")

    # Determine expected dimension based on embedding model
    expected_dim = _get_dimension(EMBEDDING_MODEL)

    if index_dim and index_dim != expected_dim:
        raise RuntimeError(
            f"Pinecone index dimension mismatch: index has {index_dim} dimensions, "
            f"but embedding model '{EMBEDDING_MODEL}' requires {expected_dim} dimensions. "
            f"Either use a different PINECONE_INDEX_NAME for this model or drop/recreate the index."
        )
    print(f"✓ Pinecone index '{PINECONE_INDEX_NAME}' validated: {index_dim} dimensions match {EMBEDDING_MODEL}")
except Exception as e:
    if "mismatch" in str(e).lower():
        raise
    # Don't fail on other errors (e.g., network issues during validation)
    print(f"Warning: Could not validate Pinecone index dimension: {e}")
