# backend/agent/planner.py

"""Query planning: generate search variations for broader coverage."""

from typing import List

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from logger import logger
from .models import QueryPlan


def _plan_search_queries(
    llm: ChatNVIDIA,
    question: str,
    max_variations: int = 4,
) -> List[str]:
    """
    Ask the LLM to propose related keyword variations so search can cover synonyms/concepts.
    Returns the original query plus up to (max_variations-1) alternates.
    Uses Pydantic structured output for reliable parsing.
    """
    prompt = (
        "You are a query planner for document search. Given a user's question, list concise keyword "
        "variations that capture different angles (synonyms, related terminology, abbreviations). "
        "Keep each variation under 20 words and prioritize distinct wording.\n\n"
        f"Question: {question}"
    )

    planned_variations: List[str] = []
    try:
        # Use structured output with Pydantic model
        structured_llm = llm.with_structured_output(QueryPlan)
        response = structured_llm.invoke(prompt)
        planned_variations = response.queries
    except Exception as e:
        # If structured output fails, fall back to original query only
        logger.warning(f"Query planning failed with structured output: {e}, using original query only")
        planned_variations = []

    deduped: List[str] = []
    seen = set()
    for candidate in [question] + planned_variations:
        cleaned = " ".join((candidate or "").split())
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(cleaned)
        if len(deduped) >= max_variations:
            break

    return deduped or [question]
