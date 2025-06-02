# backend/qa.py

from fastapi import APIRouter, HTTPException

from logger import logger  
from config import EMBEDDING_MODEL
from models import AskRequest
from utils import get_embedding
from pinecone_client import index
from langchain_agent import get_agent

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
    Returns both the LLM's reasoning trace and the final answer.
    """
    try:
        agent = get_agent(namespace=req.namespace)
        # `agent.run(...)` returns a single string containing both chain‐of‐thought and final answer
        output: str = agent.invoke(req.question)
        parts = output.split("Answer:")
        if len(parts) > 1:
            thought_trace = parts[0].strip()
            final_answer = parts[1].strip()
        else:
            # Fallback: treat the entire string as the answer, and leave thoughts empty
            thought_trace = ""
            final_answer = output.strip()
        return {
            "thoughts": thought_trace,
            "answer": final_answer
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))