# backend/langchain_agent.py

import re
from typing import List, Any, Dict, TypedDict, Optional, Literal

from pydantic import BaseModel, Field
from langchain_core.tools import Tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.runnables import RunnableLambda

from config import VECTOR_K
from services import clean_text  # reuse the shared cleaner
from document_summary import (
    list_documents_in_namespace,
    get_document_summary,
    search_summaries,
    fetch_full_document
)
from logger import logger

# --- Keep thresholds aligned with qa.py ---
PRIMARY_SIMILARITY_THRESHOLD = 0.6
FALLBACK_SIMILARITY_THRESHOLD = 0.4
MAX_MATCHES_TO_RETURN = 3


class AgentOutput(TypedDict):
    answer: str
    reasoning: List[str]
    sources: Optional[List[str]]


# ---------------------------
# Pydantic Models for Structured Outputs
# ---------------------------

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


# ---------------------------
# Agent Tools
# ---------------------------

def _list_documents_tool(namespace: str = "default") -> Tool:
    """
    Tool to list all documents in the namespace.
    Useful for answering "what files do you have?" or "what documents are available?"
    """
    def list_fn(query: str = "") -> str:
        docs = list_documents_in_namespace(namespace, limit=50)

        if not docs:
            return "No documents found in this namespace."

        # Format as readable list
        lines = [f"Found {len(docs)} document(s):\n"]
        for i, doc in enumerate(docs, 1):
            topics_str = ", ".join(doc.get("topics", [])[:3])
            lines.append(
                f"{i}. {doc['source']} (ID: {doc['doc_id']}, Type: {doc.get('document_type', 'unknown')}, "
                f"Topics: {topics_str})"
            )

        return "\n".join(lines)

    return Tool(
        name="list_documents",
        func=list_fn,
        description="List all available documents in the knowledge base. Use this when the user asks 'what files do you have?' or 'what documents are available?' or 'show me what's uploaded'."
    )


def _get_document_summary_tool(namespace: str = "default") -> Tool:
    """
    Tool to get structured summary with citations for a specific document.
    This should be the PRIMARY tool for broad questions about a document.
    """
    def summary_fn(doc_id: str) -> str:
        summary = get_document_summary(doc_id, namespace)

        if not summary:
            return f"No summary found for document '{doc_id}'. Use list_documents to see available documents."

        # Format summary with citations
        lines = [
            f"Document: {summary['source']}",
            f"Type: {summary.get('document_type', 'unknown')}",
            f"Subject: {summary.get('primary_subject', 'N/A')}",
            f"\nTopics: {', '.join(summary.get('topics', []))}",
            "\nKey Concepts:"
        ]

        for concept in summary.get("key_concepts", [])[:10]:
            chunk_refs = concept.get("chunk_refs", [])
            context = concept.get("context", "N/A")
            refs_str = f"[chunks: {', '.join(map(str, chunk_refs))}]" if chunk_refs else ""
            lines.append(f"  - {concept.get('concept', 'N/A')}: {context} {refs_str}")

        lines.append("\nKey Facts:")
        for fact in summary.get("key_facts", [])[:15]:
            chunk_refs = fact.get("chunk_refs", [])
            refs_str = f"[chunks: {', '.join(map(str, chunk_refs))}]" if chunk_refs else ""
            lines.append(f"  - {fact.get('fact', 'N/A')} {refs_str}")

        lines.append(f"\nTotal chunks: {summary.get('chunk_count', 0)}")

        return "\n".join(lines)

    return Tool(
        name="get_document_summary",
        func=summary_fn,
        description="Get a structured summary of a specific document with key facts and entities. Use this for broad questions like 'what is this about?', 'summarize this document', or 'what are the main points?'. Input should be the doc_id from list_documents."
    )


def _pinecone_search_tool(namespace: str = "default") -> Tool:
    """
    Vector search tool using NVIDIA Llama 3.2 embeddings with reranking.
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
        description="Search the indexed knowledge base using NVIDIA Llama 3.2 vector embeddings with reranking. Returns up to 3 evidence snippets with provenance like [source::section-X] …",
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
    Enhanced agentic controller with document-aware capabilities:
      1) Can list and summarize documents to answer meta-questions
      2) Uses structured summaries for broad questions (fast, evidence-based)
      3) Falls back to semantic search for specific details
      4) Synthesizes answers with provenance
    Returns: {"answer": str, "reasoning": [str], "sources": [str]}
    """
    # Provide all tools: list, summary, and semantic search
    if tools is None:
        tools = [
            _list_documents_tool(namespace=namespace),
            _get_document_summary_tool(namespace=namespace),
            _pinecone_search_tool(namespace=namespace)
        ]

    # Build tool lookup
    tool_map = {t.name: t.func for t in tools}

    if "semantic_search" not in tool_map:
        raise RuntimeError("semantic_search tool not found")

    search_tool = tool_map["semantic_search"]
    list_tool = tool_map.get("list_documents")
    summary_tool = tool_map.get("get_document_summary")

    llm = ChatNVIDIA(model="meta/llama-4-maverick-17b-128e-instruct", temperature=0.0)

    def _plan_search_queries(question: str, max_variations: int = 4) -> List[str]:
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

    def _compose_answer(question: str, evidence: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Compose a natural, conversational answer grounded in evidence."""
        history_section = ""
        if chat_history:
            history_lines = "\n".join(f"{item['role']}: {item['content']}" for item in chat_history)
            history_section = f"\n\nRecent conversation:\n{history_lines}\n"
        if not evidence or evidence.strip().lower() in {"no results.", "no meaningful text chunks found."}:
            return "I don't see information about that in the documents."

        prompt = (
            "You are a helpful assistant that answers questions using the provided evidence.\n"
            "You DO have access to all of the evidence text shown below; treat it as your only source of truth.\n"
            "Rules:\n"
            "1. Directly answer the user's question using ONLY the evidence.\n"
            "2. Do NOT mention tools, PDFs, file formats, or that you \"don't have access\" to something.\n"
            "3. Do NOT describe how search or retrieval was done.\n"
            "4. If evidence clearly answers the question: give a direct, confident answer.\n"
            "5. If evidence is partial or uncertain: say so explicitly, but still answer as best you can.\n"
            "6. If there is no relevant evidence at all, say exactly: \"I don't see information about that in the documents.\"\n"
            "7. Keep answers concise (1-3 sentences) and natural.\n\n"
            f"Question: {question}\n"
            f"{history_section}\n"
            f"Evidence:\n{evidence}\n\n"
            "Answer:"
        )
        response = llm.invoke(prompt)
        text = getattr(response, "content", "").strip()
        if not text:
            return "I don't see information about that in the documents."
        return text

    def _verify_answer(question: str, evidence: str, answer: str) -> str:
        """
        Verify whether the answer is supported by the evidence.
        Uses Pydantic structured output for reliable verdict parsing.
        """
        if not answer:
            return "UNSUPPORTED"

        lower = answer.strip().lower()

        # Treat meta/limitation answers as unsupported
        if any(phrase in lower for phrase in [
            "i don't have access",
            "i do not have access",
            "i can't access",
            "i cannot access",
            "i don't have the content",
            "i do not have the content",
            "i would need access",
            "i need access to",
            "the pdf",
            "the document is not",
        ]):
            logger.warning(f"Answer contains meta-talk about access/tools, marking UNSUPPORTED: '{answer[:100]}'")
            return "UNSUPPORTED"

        if lower == "the documents do not clearly specify this.":
            return "UNSUPPORTED"

        prompt = (
            "You are a strict fact checker. "
            "Given a question, an answer, and evidence snippets, determine the verification verdict.\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Evidence:\n{evidence}\n"
        )

        try:
            # Use structured output with Pydantic model
            structured_llm = llm.with_structured_output(VerificationVerdict)
            response = structured_llm.invoke(prompt)
            return response.verdict
        except Exception as e:
            # Fallback to PARTIAL if structured output fails
            logger.warning(f"Verification failed with structured output: {e}, defaulting to PARTIAL")
            return "PARTIAL"

    def _invoke(inputs: Dict[str, Any] | str) -> AgentOutput:
        # Normalize input
        history: List[Dict[str, str]] = []
        if isinstance(inputs, str):
            question = inputs
        elif isinstance(inputs, dict):
            question = inputs.get("input") or inputs.get("question")
            history = inputs.get("chat_history") or []
            if not isinstance(question, str):
                # Fallback: first string value
                question = next((v for v in inputs.values() if isinstance(v, str)), "")
        else:
            raise ValueError("Unsupported input type for agent")
        question = (question or "").strip()

        # Strip surrounding quotes if LLM added them during rewriting
        if question.startswith('"') and question.endswith('"'):
            question = question[1:-1].strip()
        if question.startswith("'") and question.endswith("'"):
            question = question[1:-1].strip()

        if not question:
            return {
                "answer": "I couldn't find a definitive answer.",
                "reasoning": ["No question text was provided."],
                "sources": [],
            }

        reasoning: List[str] = []
        observations: List[str] = []
        sources: List[str] = []

        # Detect query type and route to appropriate tools
        question_lower = question.lower()
        logger.info(f"Agent processing question: '{question}' (cleaned from quotes)")

        # Meta-questions: list documents, what's available, etc.
        if any(phrase in question_lower for phrase in [
            "what files", "what documents", "list", "show me", "what do you have",
            "what's uploaded", "available documents"
        ]):
            if list_tool:
                obs = list_tool("")
                observations.append(obs)
                reasoning.append("Used list_documents to show available files")
                return {
                    "answer": obs,
                    "reasoning": reasoning,
                    "sources": []
                }

        # Broad summary questions: what's this about, summarize, main points
        if any(phrase in question_lower for phrase in [
            "what is this about", "what's this about", "what is this doc about",
            "what's this doc about", "what is this document about",
            "summarize", "summary", "main points", "key points", "overview",
            "what is the file about", "what's the file about",
            "what is the document about", "what's the document about"
        ]):
            logger.info("Detected summary question, using document summary tool")
            if list_tool and summary_tool:
                # First, list documents to find the most recent
                docs_list = list_tool("")
                reasoning.append("Listed documents to identify target")

                # Parse doc_id from list (get first doc)
                docs = list_documents_in_namespace(namespace, limit=1)
                if docs:
                    doc_id = docs[0]["doc_id"]
                    logger.info(f"Found document {doc_id}, retrieving summary")
                    obs = summary_tool(doc_id)
                    observations.append(obs)
                    reasoning.append(f"Retrieved structured summary for {doc_id}")

                    # Format answer from summary
                    final_answer = f"Here's what I found:\n\n{obs}"
                    sources = [f"Document summary: {docs[0]['source']}"]

                    return {
                        "answer": final_answer,
                        "reasoning": reasoning,
                        "sources": sources
                    }
                else:
                    logger.warning("No documents found in namespace for summary")
                    # Fall through to semantic search

        # DOCUMENT-FIRST STRATEGY:
        # 1. Search summaries to identify relevant documents
        # 2. Fetch full document text and let LLM reason over it
        # 3. Fall back to semantic search only if no documents found

        logger.info(f"Starting document search for: '{question}'")

        # Step 1: Search summaries to identify relevant documents
        summary_results = search_summaries(query=question, namespace=namespace, top_k=3)

        available_sources = []  # Track all available sources for citation

        if summary_results:
            logger.info(f"Found {len(summary_results)} relevant documents")
            reasoning.append(f"Found {len(summary_results)} relevant document(s)")

            # Step 2: Fetch full document text for top matches
            all_document_texts = []
            for result in summary_results[:2]:  # Top 2 documents max to avoid context overflow
                doc_id = result["doc_id"]
                source = result["source"]

                logger.info(f"Fetching full text for {source}")
                reasoning.append(f"Reading full document: {source}")

                # Fetch entire document
                full_text = fetch_full_document(doc_id, namespace, max_chunks=50)

                if full_text:
                    all_document_texts.append(f"[Document: {source}]\n\n{full_text}")
                    available_sources.append(source)

            if all_document_texts:
                # Combine full documents as evidence
                merged_evidence = "\n\n".join(all_document_texts)
                logger.info(f"Successfully retrieved {len(all_document_texts)} full document(s) for LLM reasoning")
            else:
                logger.info("Failed to fetch full documents, falling back to semantic search")
                merged_evidence = ""
        else:
            logger.info("No summaries found, using semantic search")
            merged_evidence = ""

        # TIER 2: Fall back to full semantic search if summary search didn't yield results
        if not merged_evidence:
            logger.info("Falling back to full semantic search")
            reasoning.append("Summary search yielded no results, using detailed semantic search")

            planned_queries = _plan_search_queries(question, max_variations=4)
            for query in planned_queries:
                obs = search_tool(query)
                observations.append(obs)
                reasoning.append(f"Used semantic_search with: {query}")

            # Merge evidence (prefer the first search's top lines)
            evidence_blocks = []
            for obs in observations:
                if isinstance(obs, str) and obs.strip() and obs.strip().lower() not in {"no results.", "no meaningful text chunks found."}:
                    evidence_blocks.append(obs.strip())
            merged_evidence = "\n\n".join(evidence_blocks[:3])
            sources = _extract_sources(merged_evidence, limit=5)

        # Phase 2: compose and verify answer
        draft_answer = _compose_answer(question, merged_evidence, history)
        verdict = _verify_answer(question, merged_evidence, draft_answer)

        # Use more natural fallback language
        if verdict == "UNSUPPORTED":
            final_answer = "I don't see information about that in the documents."
            sources = []
        elif verdict == "PARTIAL":
            # Still use the draft answer but acknowledge uncertainty
            final_answer = draft_answer
            # Only add caveat if the answer doesn't already express uncertainty
            if "not" not in draft_answer.lower() and "unclear" not in draft_answer.lower():
                final_answer = f"Based on the available information: {draft_answer}"
        else:  # SUPPORTED
            final_answer = draft_answer

        # Extract which sources were actually used (only if we have document sources available)
        if available_sources and verdict != "UNSUPPORTED":
            try:
                # Ask LLM which documents it actually used
                citation_prompt = (
                    f"You previously answered this question: {question}\n\n"
                    f"Your answer was: {final_answer}\n\n"
                    f"Available documents: {', '.join(available_sources)}\n\n"
                    "Which of these documents did you actually use to formulate your answer? "
                    "Only list documents that contributed information to the answer."
                )
                structured_llm = llm.with_structured_output(AnswerWithCitations)
                citation_response = structured_llm.invoke(citation_prompt)
                sources = [f"[{src}]" for src in citation_response.sources_used if src]
                logger.info(f"LLM cited sources: {sources}")
            except Exception as e:
                logger.warning(f"Failed to extract source citations: {e}, using all available sources")
                sources = [f"[{src}]" for src in available_sources]
        elif not available_sources:
            # For semantic search results, use extracted sources
            sources = _extract_sources(merged_evidence, limit=5)

        if history:
            reasoning.append("Considered recent conversation context when forming query and answer.")
        reasoning.append(f"Verification verdict: {verdict}.")

        return {
            "answer": final_answer or "I don't see information about that in the documents.",
            "reasoning": reasoning or ["Searched the knowledge base and summarized the best-matching evidence."],
            "sources": sources,
        }

    return RunnableLambda(_invoke)


async def get_streaming_agent(namespace: str = "default"):
    """
    Streaming version of the agentic controller.
    Yields reasoning steps and final answer with inline citations.

    Yields events:
    - {"type": "reasoning", "content": "status message"}
    - {"type": "answer", "content": "answer with [1] [2] citations", "sources": [...]}
    """
    # Get tools
    tools = [
        _list_documents_tool(namespace=namespace),
        _get_document_summary_tool(namespace=namespace),
        _pinecone_search_tool(namespace=namespace)
    ]

    tool_map = {t.name: t.func for t in tools}
    search_tool = tool_map["semantic_search"]
    list_tool = tool_map.get("list_documents")
    summary_tool = tool_map.get("get_document_summary")

    llm = ChatNVIDIA(model="meta/llama-4-maverick-17b-128e-instruct", temperature=0.0)

    async def stream_invoke(inputs: Dict[str, Any] | str):
        # Normalize input
        history: List[Dict[str, str]] = []
        search_query = None
        if isinstance(inputs, str):
            question = inputs
        elif isinstance(inputs, dict):
            question = inputs.get("input") or inputs.get("question")
            search_query = inputs.get("search_query")  # Optional: separate query for search
            history = inputs.get("chat_history") or []
            if not isinstance(question, str):
                question = next((v for v in inputs.values() if isinstance(v, str)), "")
        else:
            raise ValueError("Unsupported input type for agent")

        question = (question or "").strip()
        # If no separate search query provided, use the main question
        if not search_query:
            search_query = question

        # Strip quotes
        if question.startswith('"') and question.endswith('"'):
            question = question[1:-1].strip()
        if question.startswith("'") and question.endswith("'"):
            question = question[1:-1].strip()

        if not question:
            yield {"type": "answer", "content": "I couldn't find a definitive answer.", "sources": []}
            return

        question_lower = question.lower()
        logger.info(f"Streaming agent processing: '{question}'")

        # Check for meta-questions (list documents)
        if any(phrase in question_lower for phrase in [
            "what files", "what documents", "list", "show me", "what do you have",
            "what's uploaded", "available documents"
        ]):
            yield {"type": "reasoning", "content": "Listing available documents..."}
            if list_tool:
                obs = list_tool("")
                yield {"type": "answer", "content": obs, "sources": []}
                return

        # Check for summary questions
        if any(phrase in question_lower for phrase in [
            "what is this about", "what's this about", "what is this doc about",
            "what's this doc about", "what is this document about",
            "summarize", "summary", "main points", "key points", "overview",
            "what is the file about", "what's the file about",
            "what is the document about", "what's the document about"
        ]):
            yield {"type": "reasoning", "content": "Retrieving document summary..."}
            if list_tool and summary_tool:
                docs = list_documents_in_namespace(namespace, limit=1)
                if docs:
                    doc_id = docs[0]["doc_id"]
                    obs = summary_tool(doc_id)
                    yield {"type": "answer", "content": f"Here's what I found:\n\n{obs}", "sources": [f"Document summary: {docs[0]['source']}"]}
                    return

        # DOCUMENT-FIRST SEARCH with inline citations
        yield {"type": "reasoning", "content": "Searching for relevant documents..."}

        summary_results = search_summaries(query=search_query, namespace=namespace, top_k=3)

        full_document_texts = []
        sources_list = []

        if summary_results:
            yield {"type": "reasoning", "content": f"Found {len(summary_results)} document(s), reading full content..."}

            for result in summary_results[:2]:  # Top 2 documents max
                doc_id = result["doc_id"]
                source = result["source"]

                # Fetch full document text
                full_text = fetch_full_document(doc_id, namespace, max_chunks=50)

                if full_text:
                    full_document_texts.append(f"=== Document: {source} ===\n\n{full_text}")
                    sources_list.append(source)

        # Fallback to semantic search if no full documents retrieved
        if not full_document_texts:
            yield {"type": "reasoning", "content": "Using semantic search..."}

            obs = search_tool(search_query)

            # Use semantic search results as fallback
            if isinstance(obs, str) and obs.strip() and obs.strip().lower() not in {"no results.", "no meaningful text chunks found."}:
                full_document_texts.append(obs)
                sources_list.append("semantic_search")

        if not full_document_texts:
            yield {"type": "answer", "content": "I don't see information about that in the documents.", "sources": []}
            return

        # Compose answer with the full document context
        yield {"type": "reasoning", "content": "Reading and analyzing content..."}

        # Build evidence text from full documents with source labels
        evidence_parts = []
        for i, (doc_text, source) in enumerate(zip(full_document_texts, sources_list)):
            evidence_parts.append(f"[Document {i+1}: {source}]\n{doc_text}")
        evidence_text = "\n\n".join(evidence_parts)

        prompt = (
            "You are a helpful assistant that answers questions based on document content.\n"
            "You DO have access to all of the document content shown below; treat it as your only source of truth.\n"
            "Rules:\n"
            "1. Directly answer the user's question using ONLY the document content.\n"
            "2. Do NOT mention tools, PDFs, file formats, or that you \"don't have access\" to something.\n"
            "3. Do NOT describe how search or retrieval was done.\n"
            "4. If the document content clearly answers the question: give a direct, confident answer.\n"
            "5. If unclear or not found: say exactly 'I don't see information about that in the documents.'\n"
            "6. Keep answers concise (1-3 sentences) and natural.\n"
            "7. In sources_used, ONLY list the document filenames that you actually used to formulate your answer.\n\n"
            f"Question: {question}\n\n"
            f"Document Content:\n{evidence_text}\n\n"
            "Provide your answer and cite which documents you actually used."
        )

        try:
            # Use structured output to get answer with citations
            structured_llm = llm.with_structured_output(AnswerWithCitations)
            response = structured_llm.invoke(prompt)
            answer = response.answer.strip()
            actual_sources_used = response.sources_used or []
        except Exception as e:
            logger.warning(f"Structured output failed for citations: {e}, falling back")
            # Fallback to plain response
            response = llm.invoke(prompt)
            answer = response.content.strip()
            actual_sources_used = sources_list  # Use all sources as fallback

        if not answer:
            answer = "I don't see information about that in the documents."

        # Guard against meta-talk about access/tools
        lower = answer.lower()
        if any(phrase in lower for phrase in [
            "i don't have access",
            "i do not have access",
            "i can't access",
            "i cannot access",
            "i don't have the content",
            "i would need access",
            "the pdf",
        ]):
            logger.warning(f"Streaming answer contains meta-talk, replacing: '{answer[:100]}'")
            answer = "I don't see information about that in the documents."
            actual_sources_used = []

        # Only include sources that were actually used by the LLM
        sources_log = [{"source": source} for source in actual_sources_used if source]
        logger.info(f"[Sources] {sources_log}")

        yield {"type": "answer", "content": answer, "sources": sources_log}

    return stream_invoke
