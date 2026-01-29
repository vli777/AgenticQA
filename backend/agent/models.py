# backend/agent/models.py

"""Type definitions and constants for the QA agent."""

from typing import List, Optional, Literal, TypedDict

from pydantic import BaseModel, Field

# --- Thresholds aligned with qa.py ---
PRIMARY_SIMILARITY_THRESHOLD = 0.6
FALLBACK_SIMILARITY_THRESHOLD = 0.4
MAX_MATCHES_TO_RETURN = 3


class AgentOutput(TypedDict):
    answer: str
    reasoning: List[str]
    sources: Optional[List[str]]


class QueryPlan(BaseModel):
    """Structured output for query planning with multiple search variations."""
    queries: List[str] = Field(
        description="List of concise keyword variations for document search (synonyms, related terminology, abbreviations). Each variation should be under 20 words."
    )


class VerificationVerdict(BaseModel):
    """Structured output for answer verification against evidence."""
    verdict: Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED"] = Field(
        description=(
            "SUPPORTED: answer is explicitly stated or directly implied in evidence. "
            "PARTIAL: some parts match but important details are missing/unclear. "
            "UNSUPPORTED: answer goes beyond evidence, contradicts it, or lacks access."
        )
    )


class AnswerWithCitations(BaseModel):
    """Structured output for answer with source citations."""
    answer: str = Field(
        description="The answer to the question based on the provided documents"
    )
    sources_used: List[str] = Field(
        description="List of document filenames that were actually used to formulate the answer (only include documents that contributed to the answer)"
    )
