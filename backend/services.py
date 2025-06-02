# backend/services.py

import numpy as np

from backend.utils import get_embedding  
from backend.pinecone_client import index 
from backend.config import EMBEDDING_MODEL

def upsert_doc(doc_text, doc_id, namespace="default"):
    """
    Upsert a document chunk to Pinecone.
    If EMBEDDING_MODEL is set, use BYOE (get embedding and upsert vector).
    Otherwise, use Pinecone's built-in embedding (pass values=None).
    """
    if EMBEDDING_MODEL in {"multilingual-e5-large", "text-embedding-3-small"}:
        # Local or OpenAI path:
        vec = get_embedding(doc_text, EMBEDDING_MODEL)
        # Convert numpy → list if necessary
        try:            
            if isinstance(vec, np.ndarray):
                vec = vec.astype("float32").tolist()
        except ImportError:
            pass

        index.upsert([{
            "id": doc_id,
            "values": vec,
            "metadata": {"text": doc_text}
        }], namespace=namespace)

    else:
        # EMBEDDING_MODEL is either None or "llama-text-embed-v2":
        # let Pinecone embed server-side (default or LLaMA).
        index.upsert([{
            "id": doc_id,
            "values": None,
            "metadata": {"text": doc_text}
        }], namespace=namespace)