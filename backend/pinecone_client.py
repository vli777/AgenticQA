# backend/pinecone_client.py

from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, VECTOR_DIMENSION

# Use VECTOR_DIMENSION from config (defaults to 2048 for nvidia/llama-3.2-nv-embedqa-1b-v2)
EMBED_DIM = VECTOR_DIMENSION

pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if required (always dense + cosine)
if not pc.has_index(PINECONE_INDEX_NAME):
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )
    print(f"✓ Created Pinecone index '{PINECONE_INDEX_NAME}' ({EMBED_DIM}-dim)")

index = pc.Index(PINECONE_INDEX_NAME)

# Validate index dimension on startup
stats = index.describe_index_stats()
index_dim = stats.get("dimension")

if index_dim != EMBED_DIM:
    raise RuntimeError(
        f"Pinecone index dimension mismatch: index={index_dim}, "
        f"model={EMBED_DIM}. Drop & recreate the index."
    )

print(f"✓ Pinecone index '{PINECONE_INDEX_NAME}' validated ({index_dim}-dim)")
