# backend/qa.py

import re
import json
from fastapi import APIRouter, HTTPException
from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langchain_openai import ChatOpenAI

from logger import logger  
from config import EMBEDDING_MODEL, OPENAI_API_KEY
from models import AskRequest
from utils import get_embedding
from pinecone_client import index
from langchain_agent import get_agent, AgentOutput
from langchain_core.output_parsers.json import JsonOutputParser

json_parser = JsonOutputParser()

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
    
    # Clean up the text in each match
    if "matches" in results:
        for match in results["matches"]:
            if "metadata" in match and "text" in match["metadata"]:
                # First do basic cleaning
                text = clean_text(match["metadata"]["text"])
                # Then improve grammar with LLM
                text = await improve_grammar(text)
                match["metadata"]["text"] = text
    
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