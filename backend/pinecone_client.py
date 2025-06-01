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
