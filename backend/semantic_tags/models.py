# backend/semantic_tags/models.py

"""Pydantic models and text helpers for semantic tagging."""

from typing import List

from pydantic import BaseModel, Field


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
