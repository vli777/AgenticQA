# backend/upload.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import re

from services import upsert_doc, chunk_text
from logger import logger
from utils import extract_text_from_pdf_bytes
from document_summary import (
    extract_structured_summary,
    store_document_summary,
    detect_cross_document_overlap,
    store_cross_document_summary
)

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

        # Chunk the text first (needed for both upsert and summary)
        from services import clean_text
        text = clean_text(text)
        chunks = chunk_text(text)

        # Generate structured summary BEFORE upserting (so we have chunks)
        logger.info(f"Extracting structured summary for '{name}'...")
        summary = extract_structured_summary(
            doc_id=safe_doc_id,
            chunks=chunks,
            source=name,
            namespace=namespace
        )

        # Store summary in Pinecone
        store_document_summary(summary, namespace)

        # Check for cross-document overlap
        cross_summary = detect_cross_document_overlap(summary, namespace)
        if cross_summary:
            logger.info(
                f"Cross-document overlap detected between {safe_doc_id} and "
                f"{cross_summary['related_docs']}"
            )
            store_cross_document_summary(cross_summary, namespace)

        # Now upsert the document chunks as usual
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
            "Indexed file '%s' (namespace=%s, chunks_indexed=%s, pinecone_vectors=%s, summary_extracted=True)",
            name,
            namespace,
            file_chunks_indexed,
            file_vectors_upserted,
        )

    # Clear cache ONCE after all documents uploaded (legacy BM25 cache, kept for backward compatibility)
    try:
        from hybrid_search import hybrid_search_engine
        hybrid_search_engine.clear_cache(namespace)
        logger.info(f"Cleared search cache for namespace '{namespace}' after uploading {len(files)} file(s)")
    except Exception as e:
        logger.warning(f"Failed to clear search cache: {str(e)}")

    logger.info(
        "Upload request complete (namespace=%s, total_chunks=%s, total_vectors=%s)",
        namespace,
        total_chunks,
        total_upserted,
    )

    return {"indexed_chunks": total_chunks}
