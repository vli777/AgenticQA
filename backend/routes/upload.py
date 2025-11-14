# backend/upload.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import re

from services import upsert_doc
from logger import logger
from utils import extract_text_from_pdf_bytes

router = APIRouter()

@router.post("/")
async def upload_documents(
    files: List[UploadFile] = File(...),
    namespace: str = "default"
):
    """Upload endpoint that extracts text and lets the backend chunk it consistently."""
    total_chunks = 0
    total_upserted = 0

    logger.info(
        "Upload request received (files=%s, namespace=%s)",
        [uploaded.filename for uploaded in files],
        namespace,
    )

    for uploaded in files:
        name = uploaded.filename
        data = await uploaded.read()

        logger.info(
            "Processing file '%s' (bytes=%s, namespace=%s)",
            name,
            len(data),
            namespace,
        )

        lower = name.lower()
        try:
            if lower.endswith(".pdf"):
                text = extract_text_from_pdf_bytes(data)
            elif lower.endswith(".txt"):
                text = data.decode("utf-8", errors="ignore")
            else:
                raise HTTPException(status_code=415, detail="Only PDF or TXT allowed")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Failed to extract text from file '%s'", name)
            raise HTTPException(status_code=400, detail=f"Text extraction error: {e}")

        text = text.strip()
        if not text:
            logger.warning("No text extracted from '%s' (namespace=%s)", name, namespace)
            continue

        safe_doc_id = re.sub(r'[^a-zA-Z0-9]+', '_', name).strip('_').lower() or "document"

        result = upsert_doc(
            text,
            doc_id=safe_doc_id,
            source=name,
            namespace=namespace,
            metadata_extra={
                "file_name": name,
                "file_id": safe_doc_id,
            },
        )

        file_chunks_indexed = result.get("chunks", 0)
        file_vectors_upserted = result.get("upserted", 0)

        total_chunks += file_chunks_indexed
        total_upserted += file_vectors_upserted

        logger.info(
            "Indexed file '%s' (namespace=%s, chunks_indexed=%s, pinecone_vectors=%s)",
            name,
            namespace,
            file_chunks_indexed,
            file_vectors_upserted,
        )

    # Clear BM25 cache ONCE after all documents uploaded
    try:
        from hybrid_search import hybrid_search_engine
        hybrid_search_engine.clear_cache(namespace)
        logger.info(f"Cleared BM25 cache for namespace '{namespace}' after uploading {len(files)} file(s)")
    except Exception as e:
        logger.warning(f"Failed to clear BM25 cache: {str(e)}")

    logger.info(
        "Upload request complete (namespace=%s, total_chunks=%s, total_vectors=%s)",
        namespace,
        total_chunks,
        total_upserted,
    )

    return {"indexed_chunks": total_chunks}
