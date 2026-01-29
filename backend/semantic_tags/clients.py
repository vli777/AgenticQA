# backend/semantic_tags/clients.py

"""LLM and HuggingFace client initialization for tag extraction."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

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
    NVIDIA_API_KEY,
    OPENAI_API_KEY,
    HUGGINGFACE_API_KEY,
    HUGGINGFACE_ZS_MODEL,
)
from logger import logger


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
