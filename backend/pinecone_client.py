# backend/pinecone_client.py

from pinecone import Pinecone

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

pc = Pinecone(api_key=PINECONE_API_KEY)

if not pc.has_index(PINECONE_INDEX_NAME):
    pc.create_index_for_model(
        name=PINECONE_INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "multilingual-e5-large",     # Default: Pinecone's built-in embedding
            "field_map": {"text": "chunk_text"}
        }
    )

    # --------- OpenAI Embedding Setup ---------
    #
    # pc.create_index(
    #     name=PINECONE_INDEX_NAME,
    #     dimension=1536,    # Use 1536 for text-embedding-3-small, or 3072 for large
    #     metric="cosine",
    #     cloud="aws",
    #     region="us-east-1"
    # )
    # --------------------------------------------------

index = pc.Index(PINECONE_INDEX_NAME)
