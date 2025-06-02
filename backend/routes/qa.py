# backend/qa.py

import re
from fastapi import APIRouter, HTTPException

from logger import logger  
from config import EMBEDDING_MODEL
from models import AskRequest
from utils import get_embedding
from pinecone_client import index
from langchain_agent import get_agent
from langchain_core.output_parsers.json import JsonOutputParser

json_parser = JsonOutputParser()

router = APIRouter()

@router.post("/")
async def ask(req: AskRequest):    
    """
    Simple semantic-search endpoint using vanilla RAG. Mounted at POST /ask/ because main.py uses prefix="/ask".
    """
    if EMBEDDING_MODEL in {"multilingual-e5-large", "text-embedding-3-small"}:
        logger.info(f"Using embedding model: {EMBEDDING_MODEL}")
        # Local/E5 or OpenAI models: compute vector locally, then do a vector query        
        q_embed = get_embedding(text=req.question, model=EMBEDDING_MODEL)
        response = index.query(
            vector=q_embed,
            top_k=5,
            include_metadata=True,
            namespace=req.namespace
        )

    else:
        # Pinecone-managed embeddings (either default or LLaMA)
        if EMBEDDING_MODEL == "llama-text-embed-v2":
            logger.info(f"Using embedding model: {EMBEDDING_MODEL}")
        else:
            logger.info("Using embedding model: text-embedding-3-small")
            
        query_body = {"top_k": 5, "inputs": {"text": req.question}}

        if EMBEDDING_MODEL == "llama-text-embed-v2":            
            query_body["model"] = "llama-text-embed-v2"

        response = index.search(
            namespace=req.namespace,
            query=query_body
        )
        
    results = response.to_dict()
    return {"results": results}


@router.post("/agentic")
async def ask_agentic(req: AskRequest):
    """
    Agentic retrieval-augmented QA. Mounted at POST /ask/agentic.
    Returns the final answer parsed from the agent's output.
    """
    agent = get_agent(namespace=req.namespace)
    try:
        result = agent.invoke(req.question)
        # Just return everything as a string, no parsing at all
        return {"output": str(result)}
    except Exception as e:
        # Even on error, return what you have
        return {"output": str(result) if result else None, "error": str(e)}
        # raise HTTPException(status_code=500, detail=str(e))