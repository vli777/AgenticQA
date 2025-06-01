# backend/utils.py

from sentence_transformers import SentenceTransformer
import openai

e5_model = SentenceTransformer('intfloat/e5-base-v2')

def get_embedding(text, method="e5"):
    """
    Returns the embedding for the input text using the specified method.
    method: "e5" (default, local open-source) or "openai"
    """
    if method == "openai":        
        response = openai.embeddings.create(input=[text], model="text-embedding-3-small")
        return response.data[0].embedding
    elif method == "e5":
        return e5_model.encode([text])[0]
    else:
        raise ValueError("Unknown embedding method: " + method)
