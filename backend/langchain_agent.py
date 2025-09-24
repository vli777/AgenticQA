# backend/langchain_agent.py
from __future__ import annotations
import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from langchain.tools import Tool, BaseTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

# Vector store wrapper
from langchain_pinecone import Pinecone as LC_Pinecone

# --- Optional LLM backends (we’ll pick based on env) ---
_OPENAI_OK = True
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception:
    _OPENAI_OK = False
    ChatOpenAI = None  # type: ignore
    OpenAIEmbeddings = None  # type: ignore

_NVIDIA_OK = True
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
except Exception:
    _NVIDIA_OK = False
    ChatNVIDIA = None  # type: ignore

_HF_EMBED_OK = True
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    _HF_EMBED_OK = False
    HuggingFaceEmbeddings = None  # type: ignore

# ==============================
# Environment / Config
# ==============================
# OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "").strip()
# OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-5").strip()
NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL", "meta/llama-4-maverick-17b-128e-instruct"
).strip()
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()


# ==============================
# Output schema (for validation)
# ==============================
class AgentOutput(BaseModel):
    answer: str = Field(description="Final answer or 'I don't know' if insufficient")
    reasoning: List[str] = Field(description="2–4 short bullets")
    sources: List[str] = Field(description="Exact lines returned by the search tool")


# Back-compat alias used earlier
AgentJSON = AgentOutput


# ==============================
# Embeddings / Vectorstore
# ==============================
def _get_embeddings(
    namespace: str = "default",
):  # signature doesn't matter outside this file
    from runtime import get_embeddings

    return get_embeddings()


def _get_vectorstore(namespace: str = "default") -> LC_Pinecone:
    if not PINECONE_INDEX_NAME:
        raise RuntimeError("PINECONE_INDEX_NAME env var is required.")
    embeddings = _get_embeddings()
    return LC_Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace=namespace,
        text_key="text",
    )


# ==============================
# RAG Tool (semantic_search)
# ==============================
def _pinecone_search_tool(
    namespace: str = "default", similarity_threshold: float = 0.30
) -> Tool:
    """
    Returns up to 5 readable, single-line snippets as:
    [source::section] (similarity: 0.00-1.00) snippet...
    """
    vectorstore = _get_vectorstore(namespace)

    def _clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if text and not text.endswith((".", "!", "?")):
            text += "."
        return text

    def _to_similarity(raw_score: float) -> float:
        """
        Normalize diverse scores into [0,1]:
        - [-1,1] => cosine similarity
        - [0,2]  => cosine distance -> 1 - d
        """
        try:
            s = float(raw_score)
        except Exception:
            return 0.0
        if -1.0 <= s <= 1.0:
            sim = s
        elif 0.0 <= s <= 2.0:
            sim = 1.0 - s
        else:
            sim = s
        return max(0.0, min(1.0, sim))

    def search_fn(query: str) -> str:
        hits: List[Tuple[Any, float]] = vectorstore.similarity_search_with_score(
            query, k=20
        )

        ranked: List[Tuple[Any, float]] = []
        qualified: List[Tuple[Any, float]] = []

        for doc, raw in hits:
            sim = _to_similarity(raw)
            ranked.append((doc, sim))
            if sim >= similarity_threshold:
                qualified.append((doc, sim))

        low_conf = False
        if not qualified and ranked:
            low_conf = True
            ranked.sort(key=lambda x: x[1], reverse=True)
            qualified = ranked[:5]

        if not qualified:
            return f"No results found above the {similarity_threshold} similarity threshold."

        # Group by doc_id for stable provenance
        by_doc: Dict[str, List[Tuple[Any, float]]] = {}
        for d, s in qualified:
            meta = d.metadata or {}
            doc_id = str(meta.get("doc_id", "unknown"))
            by_doc.setdefault(doc_id, []).append((d, s))

        lines: List[str] = []
        for doc_id, chunks in by_doc.items():
            chunks.sort(key=lambda x: x[0].metadata.get("chunk_id", 0))
            source = (
                chunks[0][0].metadata.get("source")
                or chunks[0][0].metadata.get("file_name")
                or "unknown"
            )
            for chunk, sim in chunks:
                text = _clean_text(
                    getattr(chunk, "page_content", "") or chunk.metadata.get("text", "")
                )
                if len(text) < 50 or not re.search(r"[.!?]", text):
                    continue
                section = chunk.metadata.get("section_index")
                prov = (
                    f"{source}::section-{section}"
                    if section is not None
                    else f"{source}::{doc_id}"
                )
                lines.append(f"[{prov}] (similarity: {sim:.2f}) {text}")

        if low_conf and lines:
            lines.insert(
                0, "Results below similarity threshold; showing best available matches."
            )

        if not lines:
            return "No meaningful text chunks found in the results."

        return "\n".join(lines[:5])

    return Tool(
        name="semantic_search",
        func=search_fn,
        description=(
            "Search the knowledge base for the user's query and return up to five "
            "highly relevant snippets. Format each result as a single line: "
            "[source::section] (similarity: 0.00-1.00) snippet... "
            "Copy these lines directly into the 'sources' array."
        ),
    )


# ==============================
# Prompts / Agent
# ==============================
SYSTEM_GUARDRAIL = (
    "You are a careful research agent. Before answering ANY user question, you must "
    "call the `semantic_search` tool at least once. If the first attempt is weak, "
    "try one alternate query. Do not include chain-of-thought in the answer."
)


def _build_agent_executor(llm: BaseChatModel, tools: List[BaseTool]) -> AgentExecutor:
    """
    Model-agnostic tool-calling agent (works with OpenAI, NVIDIA, Anthropic, etc.).
    Requires a Tool named 'semantic_search'.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=SYSTEM_GUARDRAIL
                + "\n\nYou have access to these tools:\n{tools}"
            ),
            ("human", "Answer the user's question using the tool results.\n\n{input}"),
            # REQUIRED by create_tool_calling_agent to show prior tool calls/thoughts
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=6,
        early_stopping_method="generate",
    )


# ==============================
# Tiny JSON post-processor
# ==============================
def _extract_sources(observations_text: str) -> List[str]:
    lines = []
    for raw in observations_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and "]" in line:
            lines.append(line)
    return lines[:5]


def _first_tool_call_happened(intermediate_steps: Any) -> bool:
    try:
        for action, _obs in intermediate_steps:
            if getattr(action, "tool", None):
                return True
    except Exception:
        pass
    return False


def _gather_observations(intermediate_steps: Any) -> str:
    parts: List[str] = []
    try:
        for _action, obs in intermediate_steps:
            if isinstance(obs, str) and obs.strip():
                parts.append(obs.strip())
    except Exception:
        pass
    return "\n".join(parts)


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


# ==============================
# Public API expected by qa.py
# ==============================
class _AgentChain:
    """
    Minimal wrapper so qa.py can do:
        chain = get_agent(namespace=...)
        result = chain.invoke({"input": question})
    and get a dict with answer/reasoning/sources.
    """

    def __init__(self, llm: BaseChatModel, namespace: str):
        self.llm = llm
        self.namespace = namespace
        self.tools: List[BaseTool] = [
            _pinecone_search_tool(namespace=namespace, similarity_threshold=0.30)
        ]
        self.executor = _build_agent_executor(self.llm, self.tools)

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs.get("input") if isinstance(inputs, dict) else str(inputs)
        result: Dict[str, Any] = self.executor.invoke({"input": question})

        # Did the agent call any tool? If not, force one programmatic search
        observations_text = _gather_observations(result.get("intermediate_steps", []))
        if not _first_tool_call_happened(result.get("intermediate_steps", [])):
            try:
                extra = self.tools[0].func(question)
                if extra:
                    observations_text = (observations_text + "\n" + extra).strip()
            except Exception:
                pass

        # Build a stable JSON using a tiny follow-up prompt (no special parser)
        # Keep it minimal to avoid version issues.
        summarize_prompt = (
            "Return ONLY a JSON object with exactly these keys: "
            '"answer" (string), "reasoning" (array of 2-4 short strings), '
            '"sources" (array of strings). '
            "Do not add other keys or text.\n\n"
            f"Question:\n{question}\n\n"
            "Context (tool observations and your draft answer):\n"
            f"{(result.get('output') or '').strip()}\n\n"
            f"{observations_text}\n"
        )
        raw = self.llm.invoke(summarize_prompt)
        content = getattr(raw, "content", raw)

        data = _safe_json_loads(content) or {}
        # Normalize & validate
        try:
            obj = AgentOutput(
                **{
                    "answer": data.get("answer", ""),
                    "reasoning": data.get("reasoning", []),
                    "sources": data.get("sources", []),
                }
            )
        except Exception:
            # Fallback: salvage sources from observations
            srcs = data.get("sources") or _extract_sources(observations_text)
            obj = AgentOutput(
                answer=data.get("answer") or "I don't know",
                reasoning=(
                    data.get("reasoning")
                    or ["Searched the knowledge base", "Insufficient context"]
                )[:4],
                sources=srcs[:5],
            )

        # If sources are still empty, extract from observations
        if not obj.sources:
            obj.sources = _extract_sources(observations_text)

        # Return as a plain dict for qa.py
        return obj.dict()


def get_agent(namespace: str = "default"):
    from runtime import get_llm

    llm = get_llm()
    return _AgentChain(llm=llm, namespace=namespace)
