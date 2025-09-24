# backend/runtime.py
from __future__ import annotations
import os
from typing import Optional

from config import OPENAI_API_KEY  # may be None

# model knobs (optional)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-4-maverick-17b-128e-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))


def get_llm():
    """
    Canonical LLM selector:
      - If OPENAI_API_KEY is set -> OpenAI Chat
      - Else -> NVIDIA Chat
    """
    key = (OPENAI_API_KEY or "").strip()

    if key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=OPENAI_MODEL, temperature=LLM_TEMPERATURE, openai_api_key=key
        )

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    return ChatNVIDIA(model=NVIDIA_MODEL, temperature=LLM_TEMPERATURE)


def get_embeddings(prefer: Optional[str] = None):
    """
    Canonical Embeddings selector:
      - If OPENAI_API_KEY is set -> OpenAI (text-embedding-3-small)
      - Else -> HuggingFace e5-large
    You can hint with prefer: "text-embedding-3-small" or "multilingual-e5-large".
    """
    key = (OPENAI_API_KEY or "").strip()

    # Honor explicit preference first
    if prefer == "text-embedding-3-small" and key:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=key)
    if prefer == "multilingual-e5-large":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="intfloat/e5-large")

    # Default rule
    if key:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=key)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="intfloat/e5-large")


def embed_text(text: str, prefer: Optional[str] = None) -> list[float]:
    """Convenience helper used by simple RAG."""
    emb = get_embeddings(prefer=prefer)
    if hasattr(emb, "embed_query"):
        return emb.embed_query(text)
    # fallback (some providers only expose embed_documents)
    vecs = emb.embed_documents([text])
    return vecs[0] if vecs else []


def embed_texts(texts: list[str], prefer: Optional[str] = None) -> list[list[float]]:
    """
    Batch embedding helper:
      - If provider supports .embed_documents, use it.
      - Else fall back to per-item .embed_query.
    """
    emb = get_embeddings(prefer=prefer)
    # Preferred, efficient path
    if hasattr(emb, "embed_documents"):
        return emb.embed_documents(texts)
    # Fallback: per-item (rare)
    out: list[list[float]] = []
    for t in texts:
        if hasattr(emb, "embed_query"):
            out.append(emb.embed_query(t))
        else:
            out.append([])  # shouldn't happen
    return out
