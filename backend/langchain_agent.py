# backend/langchain_agent.py

import re
import json
from typing import List, Any, Dict, TypedDict, Optional

from langchain.tools import Tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.runnables import RunnableLambda

from config import BM25_K, VECTOR_K
from services import clean_text  # reuse the shared cleaner

# --- Keep thresholds aligned with qa.py ---
PRIMARY_SIMILARITY_THRESHOLD = 0.6
FALLBACK_SIMILARITY_THRESHOLD = 0.4
MAX_MATCHES_TO_RETURN = 3


class AgentOutput(TypedDict):
    answer: str
    reasoning: List[str]
    sources: Optional[List[str]]


# ---------------------------
# Pinecone search tool — identical logic to RAG
# ---------------------------
def _pinecone_search_tool(namespace: str = "default") -> Tool:
    """
    Hybrid search tool using BM25 + vector embeddings with cross-encoder re-ranking.
    Returns top-ranked evidence snippets with provenance tags.
    """

    def search_fn(query: str) -> str:
        # Use hybrid search with re-ranking
        # Use sync wrapper to avoid event loop conflicts
        from hybrid_search import hybrid_search_sync

        reranked_results = hybrid_search_sync(
            query=query,
            namespace=namespace,
            top_k=MAX_MATCHES_TO_RETURN,
            bm25_k=BM25_K,
            vector_k=VECTOR_K
        )

        if not reranked_results:
            return "No results."

        # Format provenance-tagged snippets
        lines: List[str] = []
        for result in reranked_results:
            md = result.get("metadata") or {}
            text = clean_text(md.get("text") or "")
            if not text or len(text) < 50:
                continue
            source = md.get("source") or md.get("file_name") or "unknown"
            section = md.get("section_index") or md.get("chunk_id")
            score = result.get("rerank_score", 0.0)
            prov = f"{source}::section-{section}" if section is not None else source

            # Trim to ~1000 chars on sentence boundaries
            if len(text) > 1000:
                pieces = re.split(r"([.!?])\s+", text)
                buf = ""
                for i in range(0, len(pieces), 2):
                    s = pieces[i] + (pieces[i + 1] if i + 1 < len(pieces) else "")
                    if len(buf) + len(s) > 1000:
                        break
                    buf += s + " "
                text = buf.strip()

            lines.append(f"[{prov}] (rerank_score: {score:.3f}) {text}")

        return "\n\n".join(lines) if lines else "No meaningful text chunks found."

    return Tool(
        name="semantic_search",
        func=search_fn,
        description="Search the indexed knowledge base using hybrid search (BM25 + embeddings) with cross-encoder re-ranking. Returns up to 3 evidence snippets with provenance like [source::section-X] …",
    )


# ---------------------------
# Agent construction (Zero-shot ReAct) — manages scratchpad internally
# ---------------------------
def _extract_sources(obs: str, limit: int = 5) -> List[str]:
    if not isinstance(obs, str):
        return []
    out: List[str] = []
    seen = set()
    for ln in obs.splitlines():
        ln = ln.strip()
        if ln.startswith("["):
            if ln not in seen:
                out.append(ln)
                seen.add(ln)
            if len(out) >= limit:
                break
    return out


def get_agent(namespace: str = "default", tools: list = None):
    """
    Deterministic agentic controller that:
      1) Uses the LLM to expand the user query into multiple focused search variations
      2) Runs the Pinecone hybrid search tool for each variation and aggregates evidence
      3) Synthesizes a single final answer with provenance
    Returns: {"answer": str, "reasoning": [str], "sources": [str]}
    """
    # Reuse our Pinecone-backed tool (mirrors RAG query behavior & thresholds)
    if tools is None:
        tools = [_pinecone_search_tool(namespace=namespace)]

    # Grab the search function
    search_tool = None
    for t in tools:
        if t.name == "semantic_search":
            search_tool = t.func
            break
    if search_tool is None:
        raise RuntimeError("semantic_search tool not found")

    llm = ChatNVIDIA(model="meta/llama-4-maverick-17b-128e-instruct", temperature=0.0)

    def _plan_search_queries(question: str, max_variations: int = 4) -> List[str]:
        """
        Ask the LLM to propose related keyword variations so search can cover synonyms/concepts.
        Returns the original query plus up to (max_variations-1) alternates.
        """
        prompt = (
            "You are a query planner for document search. Given a user's question, list concise keyword\n"
            "variations that capture different angles (synonyms, related terminology, abbreviations).\n"
            "Return STRICT JSON such as {\"queries\": [\"variation 1\", \"variation 2\"]}. "
            "Keep each variation under 20 words and prioritize distinct wording.\n\n"
            f"Question: {question}"
        )

        planned_variations: List[str] = []
        try:
            response = llm.invoke(prompt)
            raw = (response.content or "").strip()
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                planned_variations = parsed.get("queries") or []
            elif isinstance(parsed, list):
                planned_variations = parsed
        except Exception:
            # Fall back to a simple heuristic rephrase if parsing fails
            q2 = re.sub(r"\b(knows?|know|knowledge of)\b", "skills with", question, flags=re.I)
            if q2 != question:
                planned_variations = [q2]
            else:
                planned_variations = [f"{question} skills resume profile"]

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

    def _compose_answer(question: str, evidence: str) -> str:
        """Single-shot composition; no loops, no scratchpad."""
        # If we truly found nothing, answer gracefully
        if not evidence or evidence.strip().lower() in {"no results.", "no meaningful text chunks found."}:
            prompt = (
                "Question:\n"
                f"{question}\n\n"
                "Evidence:\n"
                "(none)\n\n"
                "You found no evidence in the knowledge base. Reply concisely that no information is available."
            )
            return llm.invoke(prompt).content

        prompt = (
            "You are answering based ONLY on the evidence below. "
            "If the evidence supports a clear answer, answer it directly and cite the sources in parentheses "
            "using the bracketed provenance lines.\n\n"
            f"Question:\n{question}\n\n"
            f"Evidence snippets (each starts with a bracketed provenance like [source::section-X]):\n{evidence}\n\n"
            "Write a concise answer in 1–2 sentences. If relevant, include the most helpful provenance lines in parentheses."
        )
        return llm.invoke(prompt).content

    def _invoke(inputs: Dict[str, Any] | str) -> AgentOutput:
        # Normalize input
        if isinstance(inputs, str):
            question = inputs
        elif isinstance(inputs, dict):
            question = inputs.get("input") or inputs.get("question")
            if not isinstance(question, str):
                # Fallback: first string value
                question = next((v for v in inputs.values() if isinstance(v, str)), "")
        else:
            raise ValueError("Unsupported input type for agent")
        question = (question or "").strip()
        if not question:
            return {
                "answer": "I couldn't find a definitive answer.",
                "reasoning": ["No question text was provided."],
                "sources": [],
            }

        # Phase 1: multi-query search using LLM-generated variations
        reasoning: List[str] = []
        observations: List[str] = []

        planned_queries = _plan_search_queries(question, max_variations=4)
        for query in planned_queries:
            obs = search_tool(query)
            observations.append(obs)
            reasoning.append(f"Used semantic_search with: {query}")

        # Merge evidence (prefer the first search’s top lines)
        evidence_blocks = []
        for obs in observations:
            if isinstance(obs, str) and obs.strip() and obs.strip().lower() not in {"no results.", "no meaningful text chunks found."}:
                evidence_blocks.append(obs.strip())
        merged_evidence = "\n\n".join(evidence_blocks[:3])

        sources = _extract_sources(merged_evidence, limit=5)

        # Phase 2: compose final answer (single LLM call)
        final_answer = _compose_answer(question, merged_evidence)

        return {
            "answer": final_answer or "I couldn't find a definitive answer.",
            "reasoning": reasoning or ["Searched the knowledge base and summarized the best-matching evidence."],
            "sources": sources,
        }

    return RunnableLambda(_invoke)
