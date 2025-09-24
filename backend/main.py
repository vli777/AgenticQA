# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import DEBUG, EMBEDDING_MODEL, CORS_ORIGINS
from utils import get_embedding
from routes.upload import router as upload_router
from routes.qa import router as qa_router

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

if DEBUG:
    from routes.debug import router as debug_router

    app.include_router(debug_router, prefix="/debug")

embedding = None
if EMBEDDING_MODEL:
    embedding = get_embedding(
        text="Embedding model set to " + EMBEDDING_MODEL, model=EMBEDDING_MODEL
    )


@app.get("/")
def root():
    return {"message": "AgenticQA running"}
