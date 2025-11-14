# backend/semantic_tags.py

"""Semantic tagging utilities for augmenting chunk metadata and queries."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Set, Tuple

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None  # type: ignore

from config import (
    ENABLE_SEMANTIC_TAGGING,
    HUGGINGFACE_API_KEY,
    HUGGINGFACE_ZS_MODEL,
    SEMANTIC_TAG_LABELS,
    SEMANTIC_TAG_THRESHOLD,
)
from logger import logger

# Keywords for quick heuristic tagging (keep lowercase)
_TAG_KEYWORDS = {
    "python": [
        "python",
        "pandas",
        "numpy",
        "torch",
        "pytorch",
        "langchain",
        "fastapi",
        "django",
        "flask",
        "scikit",
        "sklearn",
        "airflow",
    ],
    "javascript": [
        "javascript",
        "node.js",
        "nodejs",
        "react",
        "vue",
        "angular",
        "next.js",
        "typescript",
    ],
    "typescript": [
        "typescript",
        "ts-node",
        "deno",
    ],
    "java": [
        "java",
        "spring",
        "spring boot",
        "jvm",
        "maven",
        "gradle",
    ],
    "go": [
        "golang",
        "go ",
    ],
    "rust": [
        "rust",
        "cargo",
    ],
    "c++": [
        "c++",
        "cplusplus",
        "cuda",
    ],
    "c#": [
        "c#",
        ".net",
        "dotnet",
    ],
    "sql": [
        "sql",
        "postgres",
        "mysql",
        "sqlite",
        "snowflake",
        "redshift",
        "bigquery",
    ],
    "machine learning": [
        "machine learning",
        "ml",
        "deep learning",
        "transformer",
        "bert",
        "llm",
        "gpt",
    ],
    "data engineering": [
        "data pipeline",
        "etl",
        "elt",
        "spark",
        "databricks",
        "kafka",
    ],
    "devops": [
        "kubernetes",
        "docker",
        "terraform",
        "ansible",
        "ci/cd",
    ],
}

_KEYWORD_TO_TAG = {
    kw.lower().strip(): tag
    for tag, keywords in _TAG_KEYWORDS.items()
    for kw in keywords
}

_DEFAULT_LABELS = set(SEMANTIC_TAG_LABELS or [])


def _sanitize_text(sample: str, limit: int = 750) -> str:
    sample = sample.strip()
    if len(sample) > limit:
        sample = sample[:limit]
    return sample


@lru_cache(maxsize=1)
def _hf_client() -> Optional[InferenceClient]:
    if not ENABLE_SEMANTIC_TAGGING or not HUGGINGFACE_API_KEY or InferenceClient is None:
        return None
    try:
        return InferenceClient(
            model=HUGGINGFACE_ZS_MODEL,
            token=HUGGINGFACE_API_KEY,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to initialize Hugging Face client: %s", exc)
        return None


def _keyword_tags(text: str) -> Set[str]:
    lowered = text.lower()
    tags: Set[str] = set()
    for keyword, tag in _KEYWORD_TO_TAG.items():
        if keyword in lowered:
            tags.add(tag)
    # Also include exact label matches (e.g., query literally says "python")
    for label in _DEFAULT_LABELS:
        if label and label in lowered:
            tags.add(label)
    return tags


def _zero_shot_tags(text: str) -> Set[str]:
    client = _hf_client()
    if not client or not _DEFAULT_LABELS or not ENABLE_SEMANTIC_TAGGING:
        return set()
    sample = _sanitize_text(text)
    if not sample:
        return set()
    try:
        response = client.zero_shot_classification(
            sample,
            list(_DEFAULT_LABELS),
            multi_label=True,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Hugging Face zero-shot classification failed: %s", exc)
        return set()

    tags: Set[str] = set()
    try:
        labels = response.get("labels") or []
        scores = response.get("scores") or []
        for label, score in zip(labels, scores):
            if score >= SEMANTIC_TAG_THRESHOLD:
                tags.add(label.lower())
    except AttributeError:
        logger.warning("Unexpected zero-shot payload: %s", response)
    return tags


def extract_semantic_tags(text: str) -> List[str]:
    """
    Infer semantic tags for a chunk of text. Uses keyword heuristics plus
    optional Hugging Face zero-shot classification.
    """
    if not text:
        return []
    tags = _keyword_tags(text)
    tags |= _zero_shot_tags(text)
    return sorted(tags)


@lru_cache(maxsize=512)
def _infer_query_tags_cached(query: str) -> Tuple[str, ...]:
    tags = _keyword_tags(query)
    if not tags:
        tags |= _zero_shot_tags(query)
    return tuple(sorted(tags))


def infer_query_tags(query: str) -> Set[str]:
    """Infer semantic tags implied by a user query."""
    normalized = " ".join((query or "").split())
    if not normalized:
        return set()
    return set(_infer_query_tags_cached(normalized.lower()))
