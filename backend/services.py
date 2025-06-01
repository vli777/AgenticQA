# services.py

from utils import get_embedding  
from pinecone_client import index 
from config import EMBEDDING_MODEL

def upsert_doc(doc_text, doc_id, namespace="default"):
    """
    Upsert a document chunk to Pinecone.
    If EMBEDDING_MODEL is set, use BYOE (get embedding and upsert vector).
    Otherwise, use Pinecone's built-in embedding (pass values=None).
    """
    if EMBEDDING_MODEL:
        embedding = get_embedding(doc_text, method=EMBEDDING_MODEL)
        index.upsert([{
            "id": doc_id,
            "values": embedding,
            "metadata": {"chunk_text": doc_text}
        }], namespace=namespace)
    else:
        index.upsert([{
            "id": doc_id,
            "values": None,
            "metadata": {"chunk_text": doc_text}
        }], namespace=namespace)
