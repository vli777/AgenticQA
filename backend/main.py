# backend/main.py

from fastapi import FastAPI, Body
import logging

from pinecone_client import pc, index
from config import DEBUG, EMBEDDING_MODEL
from models import AskRequest
from utils import get_embedding

logger = logging.getLogger("agenticqa")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = FastAPI(debug=DEBUG)

if DEBUG:
    from routes.debug import router as debug_router
    app.include_router(debug_router, prefix="/debug")

embedding = None
if EMBEDDING_MODEL:
    embedding = get_embedding(
        text="Embedding model set to " + EMBEDDING_MODEL,
        model=EMBEDDING_MODEL
    )
    
@app.get("/")
def root():
    return {"message": "AgenticQA running"}

@app.post("/ask")
async def ask(req: AskRequest):    
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
        results = response.to_dict()

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