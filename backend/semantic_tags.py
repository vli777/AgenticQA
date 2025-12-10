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

_DEFAULT_LABELS = set(SEMANTIC_TAG_LABELS or [])


# ---------------------------
# Pydantic Model for Structured Output
# ---------------------------

class TagList(BaseModel):
    """Structured output for semantic tag extraction."""
    tags: List[str] = Field(
        description="Array of lowercase keyword strings representing key concepts, topics, technologies, domains, or categories mentioned in the text"
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
        # Handle ChatNVIDIA (LangChain) - supports with_structured_output
        if hasattr(client, "invoke") and hasattr(client, "with_structured_output"):
            structured_llm = client.with_structured_output(TagList)
            response = structured_llm.invoke(prompt)
            return {tag.lower().strip() for tag in response.tags if tag.strip()}

        # Handle OpenAI - use native structured output via response_format
        elif hasattr(client, "chat"):
            try:
                # Try OpenAI's native structured output (requires newer SDK)
                response = client.beta.chat.completions.parse(
                    model="gpt-5.1",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format=TagList,
                )
                parsed = response.choices[0].message.parsed
                if parsed:
                    return {tag.lower().strip() for tag in parsed.tags if tag.strip()}
            except (AttributeError, Exception) as e:
                # Fallback to JSON mode if structured output not available
                logger.info("OpenAI structured output not available, using JSON mode: %s", e)
                response = client.chat.completions.create(
                    model="gpt-5.1",
                    messages=[{"role": "user", "content": prompt + "\n\nReturn ONLY a JSON array of lowercase strings."}],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                parsed = json.loads(content)
                if "tags" in parsed:
                    return {tag.lower().strip() for tag in parsed["tags"] if isinstance(tag, str) and tag.strip()}
        else:
            logger.warning("Unknown LLM client type: %s", type(client))
            return set()

        return set()

    except Exception as exc:  # pragma: no cover
        logger.warning("LLM tag extraction failed: %s", exc)
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
