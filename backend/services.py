# backend/services.py

import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque

from utils import get_embedding, get_embeddings_batch
from pinecone_client import index
from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from logger import logger
from semantic_tags import extract_semantic_tags, extract_semantic_tags_batch
from exceptions import CircuitBreakerOpenError
from patterns import (
    WHITESPACE_PATTERN,
    SPECIAL_CHARS_PATTERN,
    PARAGRAPH_SPLIT_PATTERN,
    SENTENCE_SPLIT_PATTERN,
    NUMBERS_ONLY_PATTERN,
    NAME_PATTERN,
    URL_PATTERN,
    ARXIV_PATTERN,
    DIGITS_ONLY_PATTERN,
    PUNCTUATION_PATTERN,
    WORD_PATTERN,
)

def clean_text(text: str) -> str:
    """Clean and normalize text while keeping symbols like +/# for skills."""
    text = WHITESPACE_PATTERN.sub(' ', text)
    text = SPECIAL_CHARS_PATTERN.sub('', text)
    return text.strip()

def is_meaningful_chunk(text: str) -> bool:
    """Check if a chunk contains meaningful content."""
    text = text.strip()
    if not text:
        return False

    if NUMBERS_ONLY_PATTERN.match(text):
        return False
    if NAME_PATTERN.match(text):
        return False
    if URL_PATTERN.match(text):
        return False
    if ARXIV_PATTERN.match(text):
        return False
    if DIGITS_ONLY_PATTERN.match(text):
        return False

    # Allow short bullet lists / skill inventories without punctuation
    if not PUNCTUATION_PATTERN.search(text):
        tokens = WORD_PATTERN.findall(text)
        unique_terms = len(set(t.lower() for t in tokens))
        capitalized_terms = len([t for t in tokens if t and t[0].isupper()])
        if unique_terms == 0 and capitalized_terms == 0:
            return False

    return True

def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """
    Split text into sentence-aligned chunks with overlap.

    Args:
        chunk_size: Characters per chunk (default from config: 800)
        chunk_overlap: Overlap between chunks (default from config: 300, ~37%)
    """
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP

    logger.info(
        "Chunking with size=%s, overlap=%s (%.1f%%)",
        chunk_size,
        chunk_overlap,
        chunk_overlap / chunk_size * 100,
    )

    # Clean text once before splitting
    text = clean_text(text)

    raw_sections = PARAGRAPH_SPLIT_PATTERN.split(text)
    sentences: List[str] = []
    for section in raw_sections:
        if not section:
            continue
        # Split into sentences but keep bullet-like fragments
        parts = SENTENCE_SPLIT_PATTERN.split(section)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1  # account for space
        if current_len + sentence_len > chunk_size and current_sentences:
            chunk_text_value = " ".join(current_sentences).strip()
            if is_meaningful_chunk(chunk_text_value):
                chunks.append(chunk_text_value)

            # Build overlap from the tail sentences (using deque for O(1) insertions)
            overlap_sentences = deque()
            overlap_len = 0
            for prev_sentence in reversed(current_sentences):
                prev_len = len(prev_sentence) + 1
                if overlap_len + prev_len > chunk_overlap:
                    break
                overlap_sentences.appendleft(prev_sentence)
                overlap_len += prev_len

            current_sentences = list(overlap_sentences)
            current_len = overlap_len

        current_sentences.append(sentence)
        current_len += sentence_len

    if current_sentences:
        chunk_text_value = " ".join(current_sentences).strip()
        if is_meaningful_chunk(chunk_text_value):
            chunks.append(chunk_text_value)

    return chunks

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
        import time
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
    import time
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
