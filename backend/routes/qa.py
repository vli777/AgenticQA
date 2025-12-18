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

DEFAULT_NAMESPACE = "default"
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


@router.get("/")
async def ask(
    question: str,
    namespace: str = DEFAULT_NAMESPACE,
    conversation_id: str = None
):
    """
    Main QA endpoint with streaming responses.
    Returns Server-Sent Events (SSE) stream with reasoning updates and inline citations.
    """

    async def generate():
        conv_id = conversation_id or namespace
        conversation_memory_manager.add_user_message(conv_id, question)
        topic_history = conversation_memory_manager.get_topic_context(conv_id)

        # Rewrite query
        yield f"data: {json.dumps({'type': 'reasoning', 'content': 'Processing question...'})}\n\n"

        standalone_question = _rewrite_query(question, topic_history)
        if standalone_question != question:
            logger.info(f"Rewrote agentic question '{question}' -> '{standalone_question}'")

        # Get streaming agent
        from langchain_agent import get_streaming_agent

        try:
            agent_generator = await get_streaming_agent(namespace=namespace)

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
                    conversation_memory_manager.add_assistant_message(conv_id, answer)

            # Signal completion
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Streaming agentic QA failed: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

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
