# backend/upload.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from typing import List
import re
import json

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
async def upload_documents_stream(
    files: List[UploadFile] = File(...),
    namespace: str = "default"
):
    """Streaming upload endpoint that sends progress updates via SSE."""

    # Read all files upfront before starting the stream
    file_data = []
    for uploaded in files:
        name = uploaded.filename
        data = await uploaded.read()
        file_data.append((name, data))

    async def generate_events():
        total_chunks = 0
        total_files = len(file_data)

        logger.info(
            "Streaming upload request received (files=%s, namespace=%s)",
            [name for name, _ in file_data],
            namespace,
        )

        try:
            for idx, (name, data) in enumerate(file_data, start=1):

                # Send progress update: starting file
                yield f"data: {json.dumps({'type': 'progress', 'current': idx, 'total': total_files, 'file_name': name, 'status': 'Extracting text...'})}\n\n"

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
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Only PDF or TXT allowed: {name}'})}\n\n"
                        continue
                except Exception as e:
                    logger.exception("Failed to extract text from file '%s'", name)
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Text extraction error for {name}: {str(e)}'})}\n\n"
                    continue

                text = text.strip()
                if not text:
                    logger.warning("No text extracted from '%s' (namespace=%s)", name, namespace)
                    yield f"data: {json.dumps({'type': 'progress', 'current': idx, 'total': total_files, 'file_name': name, 'status': 'Skipped (no text found)'})}\n\n"
                    continue

                safe_doc_id = re.sub(r'[^a-zA-Z0-9]+', '_', name).strip('_').lower() or "document"

                # Send progress: chunking
                yield f"data: {json.dumps({'type': 'progress', 'current': idx, 'total': total_files, 'file_name': name, 'status': 'Chunking text...'})}\n\n"

                from services import clean_text
                text = clean_text(text)
                chunks = chunk_text(text)

                # Send progress: generating summary
                yield f"data: {json.dumps({'type': 'progress', 'current': idx, 'total': total_files, 'file_name': name, 'status': 'Generating summary...'})}\n\n"

                logger.info(f"Extracting structured summary for '{name}'...")
                summary = extract_structured_summary(
                    doc_id=safe_doc_id,
                    chunks=chunks,
                    source=name,
                    namespace=namespace
                )

                # Send progress: storing summary
                yield f"data: {json.dumps({'type': 'progress', 'current': idx, 'total': total_files, 'file_name': name, 'status': 'Storing summary...'})}\n\n"

                store_document_summary(summary, namespace)

                # Send progress: checking cross-doc overlap
                yield f"data: {json.dumps({'type': 'progress', 'current': idx, 'total': total_files, 'file_name': name, 'status': 'Checking cross-document overlap...'})}\n\n"

                cross_summary = detect_cross_document_overlap(summary, namespace)
                if cross_summary:
                    logger.info(
                        f"Cross-document overlap detected between {safe_doc_id} and "
                        f"{cross_summary['related_docs']}"
                    )
                    store_cross_document_summary(cross_summary, namespace)

                # Send progress: indexing chunks
                yield f"data: {json.dumps({'type': 'progress', 'current': idx, 'total': total_files, 'file_name': name, 'status': 'Indexing chunks...'})}\n\n"

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

                logger.info(
                    "Indexed file '%s' (namespace=%s, chunks_indexed=%s, pinecone_vectors=%s, summary_extracted=True)",
                    name,
                    namespace,
                    file_chunks_indexed,
                    file_vectors_upserted,
                )

                # Send progress: file complete
                yield f"data: {json.dumps({'type': 'progress', 'current': idx, 'total': total_files, 'file_name': name, 'status': f'Complete ({file_chunks_indexed} chunks indexed)'})}\n\n"

            logger.info(
                "Upload request complete (namespace=%s, total_chunks=%s)",
                namespace,
                total_chunks,
            )

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete', 'indexed_chunks': total_chunks})}\n\n"

        except Exception as e:
            logger.exception("Error during streaming upload")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
