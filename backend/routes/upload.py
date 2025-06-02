from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

from backend.utils import extract_text_from_pdf_bytes, chunk_document_text
from backend.services import upsert_doc

router = APIRouter()

@router.post("/")
async def upload_documents(
    files: List[UploadFile] = File(...),
    namespace: str = "default"
):
    total_chunks = 0

    for uploaded in files:
        name = uploaded.filename.lower()
        if name.endswith(".pdf"):
            data = await uploaded.read()
            try:
                text = extract_text_from_pdf_bytes(data)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"PDF parse error: {e}")
        elif name.endswith(".txt"):
            data = await uploaded.read()
            text = data.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(status_code=415, detail="Only PDF or TXT allowed")

        chunks = chunk_document_text(text)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{uploaded.filename}__chunk{idx}"
            upsert_doc(chunk, doc_id=chunk_id, namespace=namespace)
            total_chunks += 1

    return {"indexed_chunks": total_chunks}
