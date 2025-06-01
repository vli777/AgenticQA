# backend/utils.py

from sentence_transformers import SentenceTransformer
import openai

_e5 = SentenceTransformer('intfloat/e5-base-v2')

def get_embedding(text: str, model: str):
    """
    Returns a list-of-floats embedding (1536-dim) for `text` using:
      • local E5 if model == "multilingual-e5-large"
      • OpenAI text-embedding-3-small if model == "text-embedding-3-small"
    """
    if model == "multilingual-e5-large":
        vec = _e5.encode([text])[0]
        return vec.astype("float32").tolist()

    if model == "text-embedding-3-small":
        resp = openai.embeddings.create(input=[text], model="text-embedding-3-small")
        return resp.data[0].embedding

    raise ValueError(f"get_embedding(): unsupported model_name={model!r}")