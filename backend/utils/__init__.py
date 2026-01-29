# backend/utils/__init__.py

"""Shared utilities: embeddings, file extraction, text processing."""

from .embeddings import (
    get_embedding,
    get_embeddings_batch,
)
from .file_extraction import (
    extract_text_from_pdf_bytes,
    extract_text_from_docx_bytes,
    chunk_document_text,
)
from .text_processing import (
    clean_text,
    is_meaningful_chunk,
    chunk_text,
)

__all__ = [
    "get_embedding",
    "get_embeddings_batch",
    "extract_text_from_pdf_bytes",
    "extract_text_from_docx_bytes",
    "chunk_document_text",
    "clean_text",
    "is_meaningful_chunk",
    "chunk_text",
]
