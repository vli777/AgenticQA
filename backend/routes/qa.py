# backend/qa.py

import re
import json
from typing import Dict, List
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langchain_openai import ChatOpenAI

from logger import logger
from config import (
    VECTOR_K,
    ENABLE_CACHING, ENABLE_STREAMING
)
from models import AskRequest
from langchain_agent import get_agent
from langchain_core.output_parsers.json import JsonOutputParser
from hybrid_search import hybrid_search_engine
from memory import conversation_memory_manager

if ENABLE_CACHING:
    from cache import search_cache, get_all_cache_stats, clear_all_caches

json_parser = JsonOutputParser()

# Tune retrieval sensitivity
PRIMARY_SIMILARITY_THRESHOLD = 0.6
FALLBACK_SIMILARITY_THRESHOLD = 0.4
MAX_MATCHES_TO_RETURN = 3

# Initialize LLM - using NVIDIA by default
llm = ChatNVIDIA(model="meta/llama-4-maverick-17b-128e-instruct", temperature=0.0)

# if OPENAI_API_KEY:
#     llm = ChatOpenAI(
#         model="gpt-4",
#         temperature=0.0,
#         openai_api_key=OPENAI_API_KEY
#     )

router = APIRouter()


def _history_to_text(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""
    return "\n".join(f"{entry['role']}: {entry['content']}" for entry in history if entry.get("content"))


def _rewrite_query(question: str, history: List[Dict[str, str]]) -> str:
    """
    Rewrite query to resolve pronouns, with guards to prevent over-rewriting by weaker LLMs.
    """
    history_text = _history_to_text(history)
    if not history_text:
        return question

    # Skip rewriting if question already looks standalone
    PRONOUNS = {"this", "that", "those", "these", "it", "they", "he", "she"}
    q_lower = question.lower()

    # If question is short and has no pronouns, it's likely already standalone
    if len(question) < 40 and not any(p in q_lower for p in PRONOUNS):
        logger.debug(f"Skipping rewrite - question appears standalone: '{question}'")
        return question

    prompt = (
        "Rewrite the user's latest question into a short standalone search query "
        "(max 20 words). Use the conversation history only to resolve pronouns. "
        "Do not add new information or expand the query.\n\n"
        f"History:\n{history_text}\n\n"
        f"Question: {question}\n\n"
        "Standalone query:"
    )
    try:
        response = llm.invoke(prompt)
        rewritten = (response.content or "").strip()

        # Hard cap length to prevent verbose rewrites
        if rewritten and len(rewritten.split()) <= 25:
            logger.debug(f"Rewrote query: '{question}' -> '{rewritten}'")
            return rewritten
        elif rewritten:
            logger.warning(f"Rewritten query too long ({len(rewritten.split())} words), using original")
    except Exception as exc:
        logger.warning(f"Query rewrite failed: {exc}")
    return question

def clean_text(text: str) -> str:
    """Clean and format text for better readability."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Fix spacing around numbers
    text = re.sub(r'(\d+)\s+', r'\1 ', text)
    # Fix spacing after punctuation
    text = re.sub(r'([.!?])\s+', r'\1 ', text)
    # Remove page numbers at the end
    text = re.sub(r'\s+\d+$', '', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Ensure proper sentence structure
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in '.!?':
        text += '.'
    return text

async def improve_grammar(text: str) -> str:
    """Use LLM to improve grammar and formatting."""
    prompt = f"""Please improve the grammar and formatting of this text while preserving its meaning. 
    Fix any broken sentences, add proper spacing, and ensure proper capitalization.
    Only return the improved text, nothing else.

    Text: {text}"""
    
    try:
        response = await llm.ainvoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error improving grammar: {str(e)}")
        return text  # Return original text if LLM fails

@router.post("/")
async def ask(req: AskRequest):
    """
    Vector search endpoint using NVIDIA Llama 3.2 embeddings with reranking.
    Mounted at POST /ask/ because main.py uses prefix="/ask".
    """
    conversation_id = req.conversation_id or req.namespace
    conversation_memory_manager.add_user_message(conversation_id, req.question)
    topic_history = conversation_memory_manager.get_topic_context(conversation_id)
    standalone_question = _rewrite_query(req.question, topic_history)
    if standalone_question != req.question:
        logger.info(f"Rewrote question '{req.question}' -> '{standalone_question}' for retrieval context")
    else:
        logger.info(f"Using vector search with re-ranking for question: '{req.question}'")

    # Check cache if enabled
    if ENABLE_CACHING:
        cached = await search_cache.get(
            query=standalone_question,
            namespace=req.namespace,
            top_k=MAX_MATCHES_TO_RETURN,
            vector_k=VECTOR_K
        )
        if cached is not None:
            logger.info("Returning cached search results")
            return {"results": {"matches": cached}, "cached": True}

    # NVIDIA OPTIMIZED RETRIEVAL PIPELINE (86.83% recall vs 13.01% for BM25):
    # 1. Vector search using llama-3.2-nv-embedqa-1b-v2 (top 60)
    # 2. Re-rank ALL using nvidia/llama-3.2-nv-rerankqa-1b-v2
    # 3. Return top-3
    reranked_results = await hybrid_search_engine.hybrid_search_with_rerank(
        query=standalone_question,
        namespace=req.namespace,
        top_k=MAX_MATCHES_TO_RETURN,
        vector_k=VECTOR_K
    )

    # Convert to the expected format
    matches = []
    for result in reranked_results:
        match = {
            "id": result["id"],
            "score": result["rerank_score"],  # Use re-rank score as primary score
            "metadata": result["metadata"]
        }

        # Attach all scores to metadata for transparency
        match["metadata"]["rerank_score"] = result["rerank_score"]
        match["metadata"]["original_score"] = result.get("original_score", 0.0)
        match["metadata"]["vector_score"] = result.get("vector_score", 0.0)

        # Clean and improve text
        if "text" in match["metadata"]:
            text = clean_text(match["metadata"]["text"])
            text = await improve_grammar(text)
            match["metadata"]["text"] = text

        matches.append(match)

    results = {"matches": matches}

    # Cache results if enabled
    if ENABLE_CACHING:
        await search_cache.set(
            query=standalone_question,
            namespace=req.namespace,
            top_k=MAX_MATCHES_TO_RETURN,
            results=matches,
            vector_k=VECTOR_K
        )

    assistant_memory = "\n\n".join(
        match["metadata"].get("text", "")
        for match in matches[:2]
    ) or "No matching evidence returned."
    conversation_memory_manager.add_assistant_message(conversation_id, assistant_memory)

    logger.info(f"Vector search returned {len(matches)} re-ranked results")

    return {"results": results, "cached": False}


@router.post("/agentic")
async def ask_agentic(req: AskRequest):
    """
    Agentic retrieval-augmented QA. Mounted at POST /ask/agentic.
    Returns structured output with answer, reasoning, and sources.
    """
    conversation_id = req.conversation_id or req.namespace
    conversation_memory_manager.add_user_message(conversation_id, req.question)
    topic_history = conversation_memory_manager.get_topic_context(conversation_id)
    standalone_question = _rewrite_query(req.question, topic_history)
    if standalone_question != req.question:
        logger.info(f"Rewrote agentic question '{req.question}' -> '{standalone_question}'")

    chain = get_agent(namespace=req.namespace)
    try:
        # Pass the input as a dictionary
        result = chain.invoke({"input": standalone_question, "chat_history": topic_history})
        logger.info(f"Chain output: {result}")

        # Ensure the result has the required fields
        if not isinstance(result, dict):
            result = {"answer": str(result), "reasoning": "Direct response", "sources": []}

        # Ensure all required fields exist
        if "answer" not in result:
            result["answer"] = str(result)
        if "reasoning" not in result:
            result["reasoning"] = "No reasoning provided"
        if "sources" not in result:
            result["sources"] = []

        conversation_memory_manager.add_assistant_message(conversation_id, result.get("answer", ""))
        return result
    except Exception as e:
        logger.error(f"Error in agentic QA: {str(e)}")
        logger.error(f"Full error details: {type(e).__name__}: {str(e)}")
        fallback = {
            "answer": "An error occurred while processing your question",
            "reasoning": f"{type(e).__name__}: {str(e)}",
            "sources": []
        }
        conversation_memory_manager.add_assistant_message(conversation_id, fallback["answer"])
        return fallback


@router.post("/agentic/stream")
async def ask_agentic_stream(req: AskRequest):
    """
    Streaming agentic QA with real-time reasoning updates and inline citations.
    Reasoning steps are streamed as single-line status updates (disappear in UI).
    Final answer includes inline citations [1], [2], etc.
    Sources are logged on backend only, not sent to frontend.
    """
    import json
    from fastapi.responses import StreamingResponse

    async def generate():
        conversation_id = req.conversation_id or req.namespace
        conversation_memory_manager.add_user_message(conversation_id, req.question)
        topic_history = conversation_memory_manager.get_topic_context(conversation_id)

        # Rewrite query
        yield f"data: {json.dumps({'type': 'reasoning', 'content': 'Processing question...'})}\n\n"

        standalone_question = _rewrite_query(req.question, topic_history)
        if standalone_question != req.question:
            logger.info(f"Rewrote agentic question '{req.question}' -> '{standalone_question}'")

        # Get streaming agent
        from langchain_agent import get_streaming_agent

        try:
            agent_generator = get_streaming_agent(namespace=req.namespace)

            # Stream reasoning steps and final answer
            async for event in agent_generator({"input": standalone_question, "chat_history": topic_history}):
                if event["type"] == "reasoning":
                    # Single-line status update (will be replaced in UI)
                    yield f"data: {json.dumps({'type': 'reasoning', 'content': event['content']})}\n\n"
                    logger.info(f"[Reasoning] {event['content']}")

                elif event["type"] == "answer":
                    # Final answer with inline citations
                    answer = event["content"]
                    sources_log = event.get("sources", [])

                    # Log sources on backend (not sent to frontend)
                    logger.info(f"[Sources] {json.dumps(sources_log, indent=2)}")

                    # Send only the answer (with inline citations)
                    yield f"data: {json.dumps({'type': 'answer', 'content': answer})}\n\n"

                    # Store in memory
                    conversation_memory_manager.add_assistant_message(conversation_id, answer)

            # Signal completion
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Streaming agentic QA failed: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/stream")
async def ask_stream(req: AskRequest):
    """
    Streaming endpoint for real-time token-by-token responses.
    Returns Server-Sent Events (SSE) stream.
    """
    if not ENABLE_STREAMING:
        raise HTTPException(status_code=501, detail="Streaming is not enabled")

    conversation_id = req.conversation_id or req.namespace
    conversation_memory_manager.add_user_message(conversation_id, req.question)
    topic_history = conversation_memory_manager.get_topic_context(conversation_id)
    standalone_question = _rewrite_query(req.question, topic_history)
    if standalone_question != req.question:
        logger.info(f"Rewrote streaming question '{req.question}' -> '{standalone_question}'")

    async def generate():
        """Generate streaming response."""
        try:
            # First, get search results
            reranked_results = await hybrid_search_engine.hybrid_search_with_rerank(
                query=standalone_question,
                namespace=req.namespace,
                top_k=MAX_MATCHES_TO_RETURN,
                vector_k=VECTOR_K
            )

            if not reranked_results:
                message = "No relevant documents found"
                yield json.dumps({"type": "error", "content": message}) + "\n"
                conversation_memory_manager.add_assistant_message(conversation_id, message)
                return

            # Format context from results
            context_parts = []
            for result in reranked_results:
                text = result.get("metadata", {}).get("text", "")
                source = result.get("metadata", {}).get("source", "unknown")
                context_parts.append(f"[{source}] {text[:500]}")

            context = "\n\n".join(context_parts)

            # Create streaming LLM
            streaming_llm = ChatNVIDIA(
                model="meta/llama-4-maverick-17b-128e-instruct",
                temperature=0.0,
                streaming=True
            )

            prompt = f"""Based on the following context, answer the question concisely.

Context:
{context}

Question: {standalone_question}

Answer:"""

            # Send sources first
            sources = [r.get("metadata", {}).get("source", "unknown") for r in reranked_results]
            yield json.dumps({"type": "sources", "content": sources}) + "\n"

            # Stream tokens
            assistant_reply = []
            async for chunk in streaming_llm.astream(prompt):
                if hasattr(chunk, "content"):
                    assistant_reply.append(chunk.content)
                    yield json.dumps({"type": "token", "content": chunk.content}) + "\n"

            # Send completion signal
            yield json.dumps({"type": "done"}) + "\n"

            if assistant_reply:
                conversation_memory_manager.add_assistant_message(conversation_id, "".join(assistant_reply))
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"
            conversation_memory_manager.add_assistant_message(conversation_id, f"Streaming error: {str(e)}")

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/cache/stats")
async def get_cache_stats():
    """Get statistics for all caches."""
    if not ENABLE_CACHING:
        return {"caching_enabled": False}

    stats = get_all_cache_stats()
    stats["caching_enabled"] = True
    return stats


@router.post("/cache/clear")
async def clear_cache():
    """Clear all caches."""
    if not ENABLE_CACHING:
        raise HTTPException(status_code=501, detail="Caching is not enabled")

    await clear_all_caches()
    return {"status": "success", "message": "All caches cleared"}
