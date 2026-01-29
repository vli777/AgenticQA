# backend/semantic_tags/query.py

"""Query-time semantic tag inference."""

from __future__ import annotations

from functools import lru_cache
from typing import Set, Tuple

from config import SEMANTIC_TAG_LABELS

from .extractor import _llm_extract_tags, _zero_shot_tags

_DEFAULT_LABELS = set(SEMANTIC_TAG_LABELS or [])


@lru_cache(maxsize=512)
def _infer_query_tags_cached(query: str) -> Tuple[str, ...]:
    """
    Infer tags from query text with caching.

    Strategy:
    1. LLM-based extraction (always on)
    2. Zero-shot classification (optional fallback if no LLM tags and labels defined)
    """
    tags: Set[str] = set()

    # Primary: LLM-based extraction (always enabled)
    tags |= _llm_extract_tags(query)

    # Fallback: Zero-shot (only if no LLM tags or labels are defined)
    if not tags or _DEFAULT_LABELS:
        tags |= _zero_shot_tags(query)

    return tuple(sorted(tags))


def infer_query_tags(query: str) -> Set[str]:
    """Infer semantic tags implied by a user query."""
    normalized = " ".join((query or "").split())
    if not normalized:
        return set()
    return set(_infer_query_tags_cached(normalized.lower()))
