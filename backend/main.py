# backend/main.py

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import DEBUG, EMBEDDING_MODEL, CORS_ORIGINS
from utils import get_embedding
from routes.upload import router as upload_router
from routes.qa import router as qa_router
from routes.debug_summary import router as debug_summary_router
from logger import logger

app = FastAPI(debug=DEBUG)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)
app.include_router(upload_router, prefix="/upload", tags=["upload"])
app.include_router(qa_router, prefix="/ask", tags=["qa"])
app.include_router(debug_summary_router, prefix="/debug/summary", tags=["debug"])

if DEBUG:
    from routes.debug import router as debug_router
    app.include_router(debug_router, prefix="/debug")


@app.on_event("startup")
async def init_embeddings():
    if not EMBEDDING_MODEL:
        logger.info("No embedding model configured; skipping embedding init")
        return

    try:
        await asyncio.to_thread(
            get_embedding,
            text="Embedding model set to " + EMBEDDING_MODEL,
            model=EMBEDDING_MODEL,
        )
        logger.info("Embedding model initialized: %s", EMBEDDING_MODEL)
    except Exception:
        logger.exception("Embedding warm-up failed; requests will compute on demand")
    
@app.get("/")
def root():
    return {"message": "AgenticQA running"}

