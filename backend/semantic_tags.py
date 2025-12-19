# backend/semantic_tags.py

"""
Semantic tagging utilities for augmenting chunk metadata and queries.

Uses a hybrid approach:
1. LLM-based extraction (primary) - domain-agnostic keyword extraction
2. Zero-shot classification (fallback) - when LLM is unavailable or for specific labels
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Set, Tuple
import json

from pydantic import BaseModel, Field

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None  # type: ignore

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
except ImportError:
    ChatNVIDIA = None  # type: ignore

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

from config import (
    LLM_TAG_MODEL,
    LLM_TAG_COUNT,
    NVIDIA_API_KEY,
    OPENAI_API_KEY,
    HUGGINGFACE_API_KEY,
    HUGGINGFACE_ZS_MODEL,
    SEMANTIC_TAG_LABELS,
    SEMANTIC_TAG_THRESHOLD,
)
from logger import logger
from exceptions import CircuitBreakerOpenError
from structured_output import get_tag_schema, extract_with_nvidia_guided_json, extract_with_openai_structured_output

_DEFAULT_LABELS = set(SEMANTIC_TAG_LABELS or [])

# Circuit breaker for failed LLM calls
_llm_failure_count = 0
_llm_circuit_open = False
FAILURE_THRESHOLD = 3


# ---------------------------
# Pydantic Model for Structured Output
# ---------------------------

class TagList(BaseModel):
    """Structured output for semantic tag extraction."""
    tags: List[str] = Field(
        description="Array of lowercase keyword strings representing key concepts, topics, technologies, domains, or categories mentioned in the text"
    )


class BatchTagList(BaseModel):
    """Structured output for batch semantic tag extraction."""
    chunks: List[TagList] = Field(
        description="Array of tag lists, one for each input chunk"
    )


def _sanitize_text(sample: str, limit: int = 750) -> str:
    sample = sample.strip()
    if len(sample) > limit:
        sample = sample[:limit]
    return sample


@lru_cache(maxsize=1)
def _llm_client():
    """Initialize LLM client for tag extraction (NVIDIA or OpenAI)."""
    # Prefer NVIDIA if available
    if NVIDIA_API_KEY and ChatNVIDIA is not None:
        try:
            return ChatNVIDIA(model=LLM_TAG_MODEL, temperature=0.0)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to initialize NVIDIA LLM client: %s", exc)

    # Fallback to OpenAI
    if OPENAI_API_KEY and OpenAI is not None:
        try:
            return OpenAI(api_key=OPENAI_API_KEY)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to initialize OpenAI client: %s", exc)

    return None


@lru_cache(maxsize=1)
def _hf_client() -> Optional[InferenceClient]:
    """Initialize HuggingFace client for zero-shot classification."""
    if not HUGGINGFACE_API_KEY or InferenceClient is None:
        return None
    try:
        return InferenceClient(
            model=HUGGINGFACE_ZS_MODEL,
            token=HUGGINGFACE_API_KEY,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to initialize Hugging Face client: %s", exc)
        return None


def _llm_extract_tags(text: str) -> Set[str]:
    """
    Extract semantic tags using LLM with Pydantic structured output (always enabled, domain-agnostic).
    Lets the LLM determine relevant keywords/topics from any content.
    """
    global _llm_failure_count, _llm_circuit_open

    # Circuit breaker: if too many failures, abort processing
    if _llm_circuit_open:
        logger.error(f"LLM circuit breaker open ({_llm_failure_count} consecutive failures), aborting processing")
        raise CircuitBreakerOpenError("An error occurred during document indexing")

    client = _llm_client()
    if not client:
        return set()

    sample = _sanitize_text(text, limit=1000)
    if not sample:
        return set()

    prompt = f"""Extract {LLM_TAG_COUNT} relevant keywords or topic tags from the following text.
Focus on: key concepts, topics, technologies, domains, or categories mentioned.

Text: {sample}"""

    try:
        # Use NVIDIA's guided_json for structured output
        if hasattr(client, "invoke"):
            tag_schema = get_tag_schema(LLM_TAG_COUNT)
            parsed = extract_with_nvidia_guided_json(client, prompt, tag_schema)

            if "tags" in parsed and isinstance(parsed["tags"], list):
                # Success - reset failure counter
                _llm_failure_count = 0
                _llm_circuit_open = False
                return {tag.lower().strip() for tag in parsed["tags"] if isinstance(tag, str) and tag.strip()}

            logger.warning("Invalid tags format in response")
            _llm_failure_count += 1
            if _llm_failure_count >= FAILURE_THRESHOLD:
                _llm_circuit_open = True
                logger.error(f"LLM circuit breaker opened after {_llm_failure_count} consecutive failures")
            return set()

        # Handle OpenAI - use native structured output
        elif hasattr(client, "chat"):
            tag_schema = get_tag_schema(LLM_TAG_COUNT)
            parsed = extract_with_openai_structured_output(client, prompt, tag_schema, model="gpt-4")

            if "tags" in parsed and isinstance(parsed["tags"], list):
                # Success - reset failure counter
                _llm_failure_count = 0
                _llm_circuit_open = False
                return {tag.lower().strip() for tag in parsed["tags"] if isinstance(tag, str) and tag.strip()}

            logger.warning("Invalid tags format in OpenAI response")
            _llm_failure_count += 1
            if _llm_failure_count >= FAILURE_THRESHOLD:
                _llm_circuit_open = True
                logger.error(f"LLM circuit breaker opened after {_llm_failure_count} consecutive failures")
            return set()
        else:
            logger.warning("Unknown LLM client type: %s", type(client))
            return set()

        return set()

    except Exception as exc:  # pragma: no cover
        logger.warning("LLM tag extraction failed: %s", exc)
        _llm_failure_count += 1
        if _llm_failure_count >= FAILURE_THRESHOLD:
            _llm_circuit_open = True
            logger.error(f"LLM circuit breaker opened after {_llm_failure_count} consecutive failures")
        return set()


def _zero_shot_tags(text: str) -> Set[str]:
    """
    Classify text using zero-shot classification (fallback method).
    Only used when SEMANTIC_TAG_LABELS are defined.
    """
    client = _hf_client()
    if not client or not _DEFAULT_LABELS:
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
    Extract semantic tags for a chunk of text.

    Strategy:
    1. LLM-based extraction (always on, domain-agnostic)
    2. Zero-shot classification (optional fallback if SEMANTIC_TAG_LABELS defined)
    """
    if not text:
        return []

    tags: Set[str] = set()

    # Primary: LLM-based extraction (always enabled)
    tags |= _llm_extract_tags(text)

    # Fallback: Zero-shot classification (only if labels are defined)
    if not tags or _DEFAULT_LABELS:
        tags |= _zero_shot_tags(text)

    return sorted(tags)


def extract_semantic_tags_batch(texts: List[str], batch_size: int = 10) -> List[List[str]]:
    """
    Extract semantic tags for multiple chunks in batches (optimization).

    Args:
        texts: List of text chunks to extract tags from
        batch_size: Number of chunks to process in each LLM call

    Returns:
        List of tag lists, one for each input text
    """
    global _llm_failure_count, _llm_circuit_open

    if not texts:
        return []

    # Circuit breaker: if open, fall back to empty tags for all
    if _llm_circuit_open:
        logger.error(f"LLM circuit breaker open, returning empty tags for {len(texts)} chunks")
        raise CircuitBreakerOpenError("An error occurred during document indexing")

    client = _llm_client()
    if not client:
        # No LLM client available, return empty tags
        return [[] for _ in texts]

    all_results: List[List[str]] = []

    # Process in batches
    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]
        batch_results = []

        try:
            # Sanitize and prepare batch
            sanitized = [_sanitize_text(t, limit=500) for t in batch_texts]

            # Build prompt for batch
            chunks_text = ""
            for i, chunk in enumerate(sanitized):
                chunks_text += f"\n\nCHUNK {i+1}:\n{chunk}"

            prompt = f"""Extract {LLM_TAG_COUNT} relevant keywords or topic tags from each of the following text chunks.
Focus on: key concepts, topics, technologies, domains, or categories mentioned.

Return a JSON array with one entry per chunk, where each entry contains a 'tags' array.
{chunks_text}"""

            # Use NVIDIA's guided_json for structured output
            if hasattr(client, "invoke"):
                # Create schema for batch processing
                batch_schema = {
                    "type": "object",
                    "properties": {
                        "chunks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "tags": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "minItems": 1,
                                        "maxItems": LLM_TAG_COUNT
                                    }
                                },
                                "required": ["tags"]
                            },
                            "minItems": len(batch_texts),
                            "maxItems": len(batch_texts)
                        }
                    },
                    "required": ["chunks"]
                }

                parsed = extract_with_nvidia_guided_json(client, prompt, batch_schema)

                if "chunks" in parsed and isinstance(parsed["chunks"], list):
                    # Success - reset failure counter
                    _llm_failure_count = 0
                    _llm_circuit_open = False

                    for chunk_data in parsed["chunks"]:
                        if "tags" in chunk_data and isinstance(chunk_data["tags"], list):
                            tags = sorted({tag.lower().strip() for tag in chunk_data["tags"] if isinstance(tag, str) and tag.strip()})
                            batch_results.append(tags)
                        else:
                            batch_results.append([])
                else:
                    logger.warning("Invalid batch tags format in response")
                    batch_results = [[] for _ in batch_texts]

            # Handle OpenAI
            elif hasattr(client, "chat"):
                batch_schema = {
                    "type": "object",
                    "properties": {
                        "chunks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "tags": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "minItems": 1,
                                        "maxItems": LLM_TAG_COUNT
                                    }
                                },
                                "required": ["tags"]
                            }
                        }
                    },
                    "required": ["chunks"]
                }

                parsed = extract_with_openai_structured_output(client, prompt, batch_schema, model="gpt-4")

                if "chunks" in parsed and isinstance(parsed["chunks"], list):
                    # Success - reset failure counter
                    _llm_failure_count = 0
                    _llm_circuit_open = False

                    for chunk_data in parsed["chunks"]:
                        if "tags" in chunk_data and isinstance(chunk_data["tags"], list):
                            tags = sorted({tag.lower().strip() for tag in chunk_data["tags"] if isinstance(tag, str) and tag.strip()})
                            batch_results.append(tags)
                        else:
                            batch_results.append([])
                else:
                    logger.warning("Invalid batch tags format in OpenAI response")
                    batch_results = [[] for _ in batch_texts]
            else:
                logger.warning("Unknown LLM client type: %s", type(client))
                batch_results = [[] for _ in batch_texts]

        except Exception as exc:
            logger.warning(f"Batch tag extraction failed for batch starting at {batch_start}: {exc}")
            _llm_failure_count += 1
            if _llm_failure_count >= FAILURE_THRESHOLD:
                _llm_circuit_open = True
                logger.error(f"LLM circuit breaker opened after {_llm_failure_count} consecutive failures")
            batch_results = [[] for _ in batch_texts]

        all_results.extend(batch_results)

    return all_results


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
