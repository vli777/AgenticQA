# backend/qa.py

import re
import json
from fastapi import APIRouter, HTTPException
from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langchain_openai import ChatOpenAI

from logger import logger
from config import EMBEDDING_MODEL, OPENAI_API_KEY, HYBRID_SEARCH_ALPHA, RETRIEVAL_K
from models import AskRequest
from utils import get_embedding
from pinecone_client import index
from langchain_agent import get_agent, AgentOutput
from langchain_core.output_parsers.json import JsonOutputParser
from hybrid_search import hybrid_search_engine

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

    # Use hybrid search with re-ranking
    # retrieval_k from config (default 20) means we get results from hybrid search before re-ranking
    # top_k=MAX_MATCHES_TO_RETURN (3) is the final number after re-ranking
    # alpha from config (default 0.5) controls weight for BM25 vs vector search
    reranked_results = await hybrid_search_engine.hybrid_search_with_rerank(
        query=req.question,
        namespace=req.namespace,
        top_k=MAX_MATCHES_TO_RETURN,
        retrieval_k=RETRIEVAL_K,
        alpha=HYBRID_SEARCH_ALPHA
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

    logger.info(f"Hybrid search returned {len(matches)} re-ranked results")

    return {"results": results}


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
