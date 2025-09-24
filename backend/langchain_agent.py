# backend/langchain_agent.py

import re
import json
from typing import List, Any, Dict, TypedDict, Optional

from langchain.tools import Tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import initialize_agent, AgentType
from langchain_core.runnables import RunnableLambda

from config import EMBEDDING_MODEL
from utils import get_embedding
from pinecone_client import index
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
    Query Pinecone using the SAME branching as /ask (RAG):
      • If EMBEDDING_MODEL in {"multilingual-e5-large","text-embedding-3-small"}:
          compute embedding locally, then index.query(...)
      • Else (Pinecone-managed embeddings, e.g. llama-text-embed-v2 or default):
          index.search(namespace=..., query={top_k, inputs:{text}, model?})
    Apply the same thresholds and format provenance-tagged snippets.
    """

    def search_fn(query: str) -> str:
        # ---- Perform query exactly like qa.py ----
        if EMBEDDING_MODEL in {"multilingual-e5-large", "text-embedding-3-small"}:
            q_embed = get_embedding(text=query, model=EMBEDDING_MODEL)
            response = index.query(
                vector=q_embed,
                top_k=20,                    # mirror /ask
                include_metadata=True,
                namespace=namespace,
            )
            results = response.to_dict() if hasattr(response, "to_dict") else response

        else:
            # Pinecone-managed embeddings (default or llama-text-embed-v2)
            query_body = {"top_k": 20, "inputs": {"text": query}}
            if EMBEDDING_MODEL == "llama-text-embed-v2":
                query_body["model"] = "llama-text-embed-v2"

            response = index.search(
                namespace=namespace,
                query=query_body,
            )
            results = response.to_dict() if hasattr(response, "to_dict") else response

        matches = (results or {}).get("matches") or []
        if not matches:
            return "No results."

        # Attach `metadata.score` (RAG does this so downstream code can read a stable field)
        for m in matches:
            md = m.setdefault("metadata", {})
            if "score" not in md:
                md["score"] = m.get("score")

        # --- Apply the same threshold logic as RAG ---
        def _filter(th: float):
            return [m for m in matches if (m.get("score") or 0) >= th]

        filtered = _filter(PRIMARY_SIMILARITY_THRESHOLD)
        if not filtered:
            filtered = _filter(FALLBACK_SIMILARITY_THRESHOLD)
        if not filtered:
            filtered = matches[:MAX_MATCHES_TO_RETURN]

        # Sort & trim
        filtered.sort(key=lambda m: m.get("score", 0), reverse=True)
        filtered = filtered[:MAX_MATCHES_TO_RETURN]

        # Format provenance-tagged snippets
        lines: List[str] = []
        for m in filtered:
            md = m.get("metadata") or {}
            text = clean_text(md.get("text") or "")
            if not text or len(text) < 50:
                continue
            source = md.get("source") or md.get("file_name") or "unknown"
            section = md.get("section_index")
            score = md.get("score") or 0.0
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

            lines.append(f"[{prov}] (similarity: {score:.2f}) {text}")

        return "\n\n".join(lines) if lines else "No meaningful text chunks found."

    return Tool(
        name="semantic_search",
        func=search_fn,
        description="Search the indexed knowledge base and return up to 3 evidence snippets with provenance like [source::section-X] …",
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
    Deterministic agentic controller:
      1) Run Pinecone search (same logic as RAG) for the original query and a rephrase
      2) Synthesize one final answer with the LLM (no ReAct loop to get 'stuck')
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

        # Phase 1: search (original + simple rephrase)
        reasoning: List[str] = []
        observations: List[str] = []

        q1 = question
        q2 = re.sub(r"\b(knows?|know|knowledge of)\b", "skills with", question, flags=re.I)
        q2 = q2 if q2 != q1 else f"{question} skills resume profile"

        obs1 = search_tool(q1)
        observations.append(obs1)
        reasoning.append(f'Used semantic_search with: {q1}')

        # If first search is thin, try the rephrase
        if (not obs1) or ("No results" in obs1) or (obs1.strip().count("\n[") + obs1.strip().startswith("[") * 1) < 1:
            obs2 = search_tool(q2)
            observations.append(obs2)
            reasoning.append(f'Used semantic_search with: {q2}')

        # Merge evidence (prefer the first search’s top lines)
        evidence_blocks = [o for o in observations if isinstance(o, str) and o.strip()]
        merged_evidence = "\n\n".join(evidence_blocks[:2])

        # Collect sources from bracketed lines
        def _extract_sources(text: str, limit: int = 5) -> List[str]:
            out, seen = [], set()
            for ln in (text or "").splitlines():
                s = ln.strip()
                if s.startswith("[") and s not in seen:
                    out.append(s)
                    seen.add(s)
                if len(out) >= limit:
                    break
            return out

        sources = _extract_sources(merged_evidence, limit=5)

        # Phase 2: compose final answer (single LLM call)
        final_answer = _compose_answer(question, merged_evidence)

        return {
            "answer": final_answer or "I couldn't find a definitive answer.",
            "reasoning": reasoning or ["Searched the knowledge base and summarized the best-matching evidence."],
            "sources": sources,
        }

    return RunnableLambda(_invoke)
