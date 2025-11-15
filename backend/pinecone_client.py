# backend/pinecone_client.py

from pinecone import Pinecone

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_MODEL

pc = Pinecone(api_key=PINECONE_API_KEY)

if not pc.has_index(PINECONE_INDEX_NAME):
    if EMBEDDING_MODEL:
        pc.create_index_for_model(
            name=PINECONE_INDEX_NAME,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": EMBEDDING_MODEL,
                "field_map": {"text": "text"}
            }
        )
    else:
        # No EMBEDDING_MODEL -> create a vanilla index so Pinecone’s built-in (text-embedding-3-small) is used        
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,    # default dimension for text-embedding-3-small
            metric="cosine",
            cloud="aws",
            region="us-east-1"
        )

index = pc.Index(PINECONE_INDEX_NAME)

# Validate index dimension matches embedding model
try:
    index_stats = index.describe_index_stats()
    index_dim = index_stats.get("dimension")

    # Determine expected dimension based on embedding model
    if EMBEDDING_MODEL == "text-embedding-3-small":
        expected_dim = 1536
    elif EMBEDDING_MODEL and (EMBEDDING_MODEL.startswith("nvidia/") or EMBEDDING_MODEL.startswith("llama")):
        expected_dim = 1024
    else:
        expected_dim = 1536  # Default

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
