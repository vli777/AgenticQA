# backend/main.py

from fastapi import FastAPI, Body

from backend.config import DEBUG, EMBEDDING_MODEL
from backend.utils import get_embedding
from backend.routes.upload import router as upload_router
from backend.routes.qa import router as qa_router

app = FastAPI(debug=DEBUG)
app.include_router(upload_router, prefix="/upload", tags=["upload"])
app.include_router(qa_router, prefix="/ask", tags=["qa"])

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

