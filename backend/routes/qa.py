# backend/qa.py

import re
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langchain_openai import ChatOpenAI

from logger import logger
from config import (
    EMBEDDING_MODEL, OPENAI_API_KEY, BM25_K, VECTOR_K,
    ENABLE_CACHING, ENABLE_STREAMING
)
from models import AskRequest
from utils import get_embedding
from pinecone_client import index
from langchain_agent import get_agent, AgentOutput
from langchain_core.output_parsers.json import JsonOutputParser
from hybrid_search import hybrid_search_engine

if ENABLE_CACHING:
    from cache import search_cache, llm_cache, get_all_cache_stats, clear_all_caches

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
    if text and not text[-1] in '.!?':
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
    Hybrid search endpoint using BM25 + vector embeddings with cross-encoder re-ranking.
    Mounted at POST /ask/ because main.py uses prefix="/ask".
    """
    logger.info(f"Using hybrid search with re-ranking for question: '{req.question}'")

    # Check cache if enabled
    if ENABLE_CACHING:
        cached = await search_cache.get(
            query=req.question,
            namespace=req.namespace,
            top_k=MAX_MATCHES_TO_RETURN,
            bm25_k=BM25_K,
            vector_k=VECTOR_K
        )
        if cached is not None:
            logger.info("Returning cached search results")
            return {"results": {"matches": cached}, "cached": True}

    # CORRECT HYBRID RAG PIPELINE:
    # 1. BM25 top-30 (lexical)
    # 2. Vector top-30 (semantic)
    # 3. Merge + dedupe (up to 60)
    # 4. Cross-encoder re-rank ALL
    # 5. Return top-3
    reranked_results = await hybrid_search_engine.hybrid_search_with_rerank(
        query=req.question,
        namespace=req.namespace,
        top_k=MAX_MATCHES_TO_RETURN,
        bm25_k=BM25_K,
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
        match["metadata"]["bm25_score"] = result.get("bm25_score", 0.0)
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
            query=req.question,
            namespace=req.namespace,
            top_k=MAX_MATCHES_TO_RETURN,
            results=matches,
            bm25_k=BM25_K,
            vector_k=VECTOR_K
        )

    logger.info(f"Hybrid search returned {len(matches)} re-ranked results")

    return {"results": results, "cached": False}


@router.post("/agentic")
async def ask_agentic(req: AskRequest):
    """
    Agentic retrieval-augmented QA. Mounted at POST /ask/agentic.
    Returns structured output with answer, reasoning, and sources.
    """
    chain = get_agent(namespace=req.namespace)
    try:
        # Pass the input as a dictionary
        result = chain.invoke({"input": req.question})
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

        return result
    except Exception as e:
        logger.error(f"Error in agentic QA: {str(e)}")
        logger.error(f"Full error details: {type(e).__name__}: {str(e)}")
        return {
            "answer": "An error occurred while processing your question",
            "reasoning": f"{type(e).__name__}: {str(e)}",
            "sources": []
        }


@router.post("/stream")
async def ask_stream(req: AskRequest):
    """
    Streaming endpoint for real-time token-by-token responses.
    Returns Server-Sent Events (SSE) stream.
    """
    if not ENABLE_STREAMING:
        raise HTTPException(status_code=501, detail="Streaming is not enabled")

    async def generate():
        """Generate streaming response."""
        try:
            # First, get search results
            reranked_results = await hybrid_search_engine.hybrid_search_with_rerank(
                query=req.question,
                namespace=req.namespace,
                top_k=MAX_MATCHES_TO_RETURN,
                bm25_k=BM25_K,
                vector_k=VECTOR_K
            )

            if not reranked_results:
                yield json.dumps({"type": "error", "content": "No relevant documents found"}) + "\n"
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

Question: {req.question}

Answer:"""

            # Send sources first
            sources = [r.get("metadata", {}).get("source", "unknown") for r in reranked_results]
            yield json.dumps({"type": "sources", "content": sources}) + "\n"

            # Stream tokens
            async for chunk in streaming_llm.astream(prompt):
                if hasattr(chunk, "content"):
                    yield json.dumps({"type": "token", "content": chunk.content}) + "\n"

            # Send completion signal
            yield json.dumps({"type": "done"}) + "\n"

        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

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
