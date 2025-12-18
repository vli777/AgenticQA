# backend/utils.py

import asyncio
import io
from typing import List

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from openai import OpenAI
from pypdf import PdfReader
from docx import Document

from config import EMBEDDING_MODEL, OPENAI_API_KEY, NVIDIA_API_KEY, ENABLE_CACHING

_DEFAULT_NVIDIA_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _is_nvidia_model(model_name: str | None) -> bool:
    return bool(model_name) and (
        model_name.startswith("nvidia/") or model_name.startswith("llama") or model_name == "nvidia-embed"
    )


_nvidia_embeddings = (
    NVIDIAEmbeddings(
        model=EMBEDDING_MODEL if _is_nvidia_model(EMBEDDING_MODEL) else _DEFAULT_NVIDIA_MODEL,
        nvidia_api_key=NVIDIA_API_KEY,
        base_url=_NVIDIA_BASE_URL,
    )
    if NVIDIA_API_KEY
    else None
)
_openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def _compute_embedding(text: str, model: str) -> List[float]:
    """Internal function to compute embedding without caching."""
    if _is_nvidia_model(model):
        if _nvidia_embeddings is None:
            raise ValueError("NVIDIA_API_KEY not configured for NVIDIA embeddings")
        # NVIDIA models have a 512 token limit
        # Very conservative truncation: ~2.0 chars/token observed for PDFs
        max_chars = 1000  # ~500 tokens, with safety margin under 512
        if len(text) > max_chars:
            text = text[:max_chars]
        return _nvidia_embeddings.embed_query(text)
    if model == "text-embedding-3-small":
        if _openai_client is None:
            raise ValueError("OPENAI_API_KEY not configured for text-embedding-3-small")
        resp = _openai_client.embeddings.create(
            input=[text],
            model="text-embedding-3-small",
        )
        return resp.data[0].embedding
    raise ValueError(f"get_embedding(): unsupported model_name={model!r}")


def _get_cached_embedding(text: str, model: str):
    from cache import embedding_cache

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop and running_loop.is_running():
        # Cannot run asyncio.run inside active loop; skip caching for this call
        return None

    return asyncio.run(embedding_cache.get_embedding(text, model, _compute_embedding))


def get_embedding(text: str, model: str = None) -> List[float]:
    """
    Returns a list-of-floats embedding for `text` using:
      • NVIDIA hosted embeddings if model contains "nvidia"
      • OpenAI text-embedding-3-small if model == "text-embedding-3-small"

    Uses caching if enabled in config.
    """
    model = model or EMBEDDING_MODEL

    if ENABLE_CACHING:
        try:
            cached = _get_cached_embedding(text, model)
            if cached is not None:
                return cached
        except Exception as e:
            from logger import logger

            logger.warning(f"Cache lookup failed, computing embedding: {e}")

    return _compute_embedding(text, model)


def get_embeddings_batch(texts: List[str], model: str = None) -> List[List[float]]:
    """
    Get embeddings for multiple texts with batching support.

    Args:
        texts: List of texts to embed
        model: Embedding model to use

    Returns:
        List of embedding vectors
    """
    model = model or EMBEDDING_MODEL

    if _is_nvidia_model(model):
        if _nvidia_embeddings is None:
            raise ValueError("NVIDIA_API_KEY not configured for NVIDIA embeddings")
        # NVIDIA models have a 512 token limit - truncate texts to be safe
        max_chars = 1000  # ~500 tokens, with safety margin under 512
        truncated_texts = [text[:max_chars] if len(text) > max_chars else text for text in texts]
        return _nvidia_embeddings.embed_documents(truncated_texts)
    if model == "text-embedding-3-small":
        if _openai_client is None:
            raise ValueError("OPENAI_API_KEY not configured for text-embedding-3-small")
        resp = _openai_client.embeddings.create(
            input=texts,
            model="text-embedding-3-small",
        )
        return [item.embedding for item in resp.data]
    raise ValueError(f"get_embeddings_batch(): unsupported model_name={model!r}")

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Read all pages from a PDF (given as raw bytes) and return their concatenated text.
    Relies on PyPDF2 (PdfReader) to extract page-by-page.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    """
    Extract text from a DOCX file (given as raw bytes) and return the concatenated text.
    Uses python-docx to read all paragraphs.
    """
    doc = Document(io.BytesIO(docx_bytes))
    text_parts = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text)
    return "\n".join(text_parts)


def chunk_document_text(full_text: str, max_chars: int = 1000) -> list[str]:
    """
    Split a long string into chunks of roughly `max_chars` characters each.
    This version splits on double-newlines when possible; if a paragraph
    would exceed max_chars, it starts a new chunk.
    """
    paragraphs = full_text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph would exceed max_chars, flush current chunk
        if len(current) + len(para) + 2 > max_chars:
            chunks.append(current.strip())
            current = para
        else:
            if current:
                current += "\n\n" + para
            else:
                current = para

    if current:
        chunks.append(current.strip())

    return chunks
