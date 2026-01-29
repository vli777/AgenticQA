# backend/semantic_tags/__init__.py

"""Semantic tagging: LLM-based and zero-shot tag extraction for chunks and queries."""

from .extractor import (
    extract_semantic_tags,
    extract_semantic_tags_batch,
)
from .query import infer_query_tags

__all__ = [
    "extract_semantic_tags",
    "extract_semantic_tags_batch",
    "infer_query_tags",
]
