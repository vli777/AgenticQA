# backend/document_summary/__init__.py

"""Document summarization: extraction, storage, retrieval, analysis, and chunk operations."""

from .extractor import (
    extract_structured_summary,
    extract_structured_summaries_batch,
)
from .storage import (
    store_document_summary,
    store_document_summaries_batch,
    store_cross_document_summary,
    get_document_summary,
    list_documents_in_namespace,
    search_summaries,
)
from .analyzer import detect_cross_document_overlap
from .chunks import (
    extract_relevant_chunks_from_summary,
    fetch_chunks_by_refs,
    fetch_full_document,
)

__all__ = [
    "extract_structured_summary",
    "extract_structured_summaries_batch",
    "store_document_summary",
    "store_document_summaries_batch",
    "store_cross_document_summary",
    "get_document_summary",
    "list_documents_in_namespace",
    "search_summaries",
    "detect_cross_document_overlap",
    "extract_relevant_chunks_from_summary",
    "fetch_chunks_by_refs",
    "fetch_full_document",
]
