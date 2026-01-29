# backend/indexing/upsert.py

import time
from typing import List, Dict, Any, Optional

import numpy as np

from utils.embeddings import get_embedding, get_embeddings_batch
from pinecone_client import index
from config import EMBEDDING_MODEL
from logger import logger
from semantic_tags import extract_semantic_tags_batch
from exceptions import CircuitBreakerOpenError


def upsert_doc(
    chunks: List[str],
    doc_id: str,
    source: str = "unknown",
    namespace: str = "default",
    metadata_extra: Optional[Dict[str, Any]] = None,
):
    """
    Upsert a document to Pinecone with proper chunking and metadata.

    Args:
        chunks: Pre-computed text chunks
        doc_id: Unique identifier for the document
        source: Source of the document (e.g., filename)
        namespace: Pinecone namespace
        metadata_extra: Additional metadata to add to all chunks
    """

    # Prepare vectors and metadata
    vectors = []
    circuit_breaker_triggered = False

    # Generate embeddings for all chunks at once (parallel optimization)
    embeddings = []
    embedding_time = 0
    if EMBEDDING_MODEL and (EMBEDDING_MODEL.startswith("nvidia") or EMBEDDING_MODEL.startswith("llama") or EMBEDDING_MODEL == "text-embedding-3-small"):
        logger.info(f"Generating embeddings for {len(chunks)} chunks using batch processing")
        start_time = time.time()
        try:
            embeddings = get_embeddings_batch(chunks, EMBEDDING_MODEL)
            # Convert numpy arrays to lists if necessary
            embeddings = [
                emb.astype("float32").tolist() if isinstance(emb, np.ndarray) else emb
                for emb in embeddings
            ]
            embedding_time = time.time() - start_time
            logger.info(f"Batch embeddings completed in {embedding_time:.2f}s ({len(chunks)/embedding_time:.1f} chunks/sec)")
        except Exception as e:
            logger.warning(f"Batch embedding failed, falling back to sequential: {e}")
            # Fallback to sequential if batch fails
            for chunk in chunks:
                vec = get_embedding(chunk, EMBEDDING_MODEL)
                if isinstance(vec, np.ndarray):
                    vec = vec.astype("float32").tolist()
                embeddings.append(vec)
            embedding_time = time.time() - start_time
    else:
        # Let Pinecone handle embedding
        embeddings = [None] * len(chunks)

    # Extract semantic tags for all chunks in batch (optimization)
    all_tags = []
    tag_time = 0
    try:
        logger.info(f"Extracting semantic tags for {len(chunks)} chunks using batch processing")
        start_time = time.time()
        all_tags = extract_semantic_tags_batch(chunks, batch_size=10)
        tag_time = time.time() - start_time
        logger.info(f"Batch tags completed in {tag_time:.2f}s ({len(chunks)/tag_time:.1f} chunks/sec)")
    except CircuitBreakerOpenError:
        logger.warning(f"Tag extraction circuit breaker triggered for {doc_id}, continuing without tags")
        circuit_breaker_triggered = True
        all_tags = [[] for _ in chunks]
    except Exception as e:
        logger.warning(f"Batch tag extraction failed, skipping tags: {e}")
        all_tags = [[] for _ in chunks]

    # Build vectors with embeddings and metadata
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"

        # Create metadata
        metadata = {
            "text": chunk,
            "source": source,
            "doc_id": doc_id,
            "chunk_id": i,
            "total_chunks": len(chunks)
        }

        if metadata_extra:
            metadata.update(metadata_extra)

        # Add semantic tags if available
        if i < len(all_tags) and all_tags[i]:
            metadata["semantic_tags"] = all_tags[i]

        vectors.append({
            "id": chunk_id,
            "values": embeddings[i],
            "metadata": metadata
        })

    # Upsert in batches of 100
    upsert_start_time = time.time()
    batch_size = 100
    upserted_total = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            response = index.upsert(batch, namespace=namespace)
        except Exception as exc:
            logger.error(
                "Pinecone upsert failed (namespace=%s, batch_index=%s, batch_size=%s, doc_id=%s)",
                namespace,
                i // batch_size,
                len(batch),
                doc_id,
                exc_info=exc,
            )
            raise

        upserted_count = None
        if hasattr(response, "upserted_count"):
            upserted_count = getattr(response, "upserted_count")
        elif isinstance(response, dict):
            upserted_count = response.get("upserted_count") or response.get("upserted")

        if isinstance(upserted_count, int):
            upserted_total += upserted_count
        else:
            upserted_total += len(batch)

        logger.info(
            "Pinecone upsert succeeded (namespace=%s, batch_index=%s, batch_size=%s, upserted=%s)",
            namespace,
            i // batch_size,
            len(batch),
            upserted_count if upserted_count is not None else len(batch),
        )

    upsert_time = time.time() - upsert_start_time
    logger.info(f"Pinecone upsert completed in {upsert_time:.2f}s")

    return {
        "chunks": len(vectors),
        "upserted": upserted_total,
        "circuit_breaker_triggered": circuit_breaker_triggered,
        "embedding_time": embedding_time,
        "tag_time": tag_time,
        "upsert_time": upsert_time,
        "chunks_per_sec_embedding": len(chunks) / embedding_time if embedding_time > 0 else 0,
        "chunks_per_sec_tags": len(chunks) / tag_time if tag_time > 0 else 0
    }
