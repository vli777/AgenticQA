# backend/services.py

import numpy as np
import re
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import get_embedding
from pinecone_client import index
from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from logger import logger

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def is_meaningful_chunk(text: str) -> bool:
    """Check if a chunk contains meaningful content."""
    text = text.strip()
    if len(text) < 50:
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

    # Allow bullet lists that may lack sentence punctuation
    if not re.search(r'[.!?]', text):
        tokens = re.findall(r'\b\w+\b', text)
        unique_terms = len(set(t.lower() for t in tokens))
        if unique_terms < 5:
            return False

    return True

def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """
    Split text into meaningful chunks with overlap for better context.

    Args:
        chunk_size: Characters per chunk (default from config: 800)
        chunk_overlap: Overlap between chunks (default from config: 300, ~37%)

    Best practices:
    - Smaller chunks (512-1024 chars) work better for retrieval
    - Higher overlap (30-40%) preserves context across boundaries
    - Current defaults: 800 chars, 300 overlap (37.5%)
    """
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP

    logger.info(f"Chunking with size={chunk_size}, overlap={chunk_overlap} ({chunk_overlap/chunk_size*100:.1f}%)")
    # First split by double newlines to preserve document structure
    sections = re.split(r'\n\s*\n', text)
    
    meaningful_chunks = []
    current_chunk = ""
    
    for section in sections:
        # Clean the section
        section = clean_text(section)
        if not section:
            continue
            
        # If section is meaningful on its own, keep it as a chunk
        if is_meaningful_chunk(section):
            if len(current_chunk) + len(section) <= chunk_size:
                current_chunk += " " + section
            else:
                if current_chunk:
                    meaningful_chunks.append(current_chunk.strip())
                current_chunk = section
            continue
            
        # Otherwise, split the section into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n", ".", "!", "?", ",", " ", ""]
        )
        
        chunks = text_splitter.split_text(section)
        for chunk in chunks:
            if is_meaningful_chunk(chunk):
                if len(current_chunk) + len(chunk) <= chunk_size:
                    current_chunk += " " + chunk
                else:
                    if current_chunk:
                        meaningful_chunks.append(current_chunk.strip())
                    current_chunk = chunk
    
    if current_chunk:
        meaningful_chunks.append(current_chunk.strip())
    
    return meaningful_chunks

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

    # Clear BM25 cache for this namespace since we added new documents
    try:
        from hybrid_search import hybrid_search_engine
        hybrid_search_engine.clear_cache(namespace)
        logger.info(f"Cleared BM25 cache for namespace '{namespace}' after document upsert")
    except Exception as e:
        logger.warning(f"Failed to clear BM25 cache: {str(e)}")

    return {"chunks": len(vectors), "upserted": upserted_total}
