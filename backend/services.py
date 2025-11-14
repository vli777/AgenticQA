# backend/services.py

import numpy as np
import re
from typing import List, Dict, Any, Optional

from utils import get_embedding
from pinecone_client import index
from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from logger import logger
from semantic_tags import extract_semantic_tags

def clean_text(text: str) -> str:
    """Clean and normalize text while keeping symbols like +/# for skills."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?+#/-]', '', text)
    return text.strip()

def is_meaningful_chunk(text: str) -> bool:
    """Check if a chunk contains meaningful content."""
    text = text.strip()
    if not text:
        return False

    if re.match(r'^[\d\s.,]+$', text):
        return False
    if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', text):
        return False
    if re.match(r'^.*\d{4}\.?\s+URL\s+https?://.*$', text):
        return False
    if re.match(r'^.*arXiv:?\d{4}\.\d{4,5}.*$', text):
        return False
    if re.match(r'^\d+$', text):
        return False

    # Allow short bullet lists / skill inventories without punctuation
    if not re.search(r'[.!?]', text):
        tokens = re.findall(r'\b[\w+#]+\b', text)
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

    raw_sections = re.split(r'\n\s*\n', text)
    sentences: List[str] = []
    for section in raw_sections:
        section = clean_text(section)
        if not section:
            continue
        # Split into sentences but keep bullet-like fragments
        parts = re.split(r'(?<=[.!?])\s+|•\s+', section)
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

            # Build overlap from the tail sentences
            overlap_sentences: List[str] = []
            overlap_len = 0
            for prev_sentence in reversed(current_sentences):
                prev_len = len(prev_sentence) + 1
                if overlap_len + prev_len > chunk_overlap:
                    break
                overlap_sentences.insert(0, prev_sentence)
                overlap_len += prev_len

            current_sentences = overlap_sentences.copy()
            current_len = overlap_len

        current_sentences.append(sentence)
        current_len += sentence_len

    if current_sentences:
        chunk_text_value = " ".join(current_sentences).strip()
        if is_meaningful_chunk(chunk_text_value):
            chunks.append(chunk_text_value)

    return chunks

def upsert_doc(
    doc_text: str,
    doc_id: str,
    source: str = "unknown",
    namespace: str = "default",
    metadata_extra: Optional[Dict[str, Any]] = None,
):
    """
    Upsert a document to Pinecone with proper chunking and metadata.
    
    Args:
        doc_text: The text content of the document
        doc_id: Unique identifier for the document
        source: Source of the document (e.g., filename)
        namespace: Pinecone namespace
    """
    # Clean the text
    doc_text = clean_text(doc_text)
    
    # Split into chunks
    chunks = chunk_text(doc_text)
    
    # Prepare vectors and metadata
    vectors = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        
        # Get embedding
        if EMBEDDING_MODEL in {"multilingual-e5-large", "text-embedding-3-small"}:
            vec = get_embedding(chunk, EMBEDDING_MODEL)
            # Convert numpy → list if necessary
            if isinstance(vec, np.ndarray):
                vec = vec.astype("float32").tolist()
        else:
            vec = None  # Let Pinecone handle embedding
            
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

        tags = extract_semantic_tags(chunk)
        if tags:
            metadata["semantic_tags"] = tags
        
        vectors.append({
            "id": chunk_id,
            "values": vec,
            "metadata": metadata
        })
    
    # Upsert in batches of 100
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

    return {"chunks": len(vectors), "upserted": upserted_total}
