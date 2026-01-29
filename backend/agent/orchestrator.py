# backend/agent/orchestrator.py

"""
Agent orchestration: sync and streaming entry points for the QA pipeline.

Coordinates tools, planner, and composer to execute the full QA workflow.
"""

from typing import List, Any, Dict, Optional

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.runnables import RunnableLambda

from logger import logger
from document_summary import (
    list_documents_in_namespace,
    search_summaries,
    fetch_full_document,
)

from .models import AgentOutput, AnswerWithCitations
from .tools import _list_documents_tool, _get_document_summary_tool, _pinecone_search_tool
from .planner import _plan_search_queries
from .composer import _compose_answer, _verify_answer, _extract_sources


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

            planned_queries = _plan_search_queries(llm, question, max_variations=4)
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
        draft_answer = _compose_answer(llm, question, merged_evidence, history)
        verdict = _verify_answer(llm, question, merged_evidence, draft_answer)

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
