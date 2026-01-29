# backend/document_summary/chunks.py

"""Chunk-level retrieval and extraction operations."""

from typing import List, Dict, Any, Optional

from logger import logger
from pinecone_client import index
from .storage import get_document_summary


def extract_relevant_chunks_from_summary(
    summary: Dict[str, Any],
    query: str,
    max_chunks: int = 5
) -> List[int]:
    """
    Given a summary and query, extract the most relevant chunk references.

    Uses flexible keyword matching on facts, entities, and technical skills.
    If no matches found, returns ALL chunks (let reranker sort it out).

    Args:
        summary: Document summary dict
        query: Original query
        max_chunks: Maximum chunk IDs to return

    Returns:
        List of chunk indices to fetch
    """
    query_lower = query.lower()
    query_words = [w.lower() for w in query_lower.split() if len(w) > 2]

    chunk_refs = set()
    chunk_scores = {}  # Track which chunks match most keywords

    # Extract chunk refs from matching key concepts (works for any document type)
    for concept in summary.get("key_concepts", []):
        concept_text = concept.get("concept", "").lower()
        context_text = concept.get("context", "").lower()
        refs = concept.get("chunk_refs", [])

        # Check both concept and context
        full_text = f"{concept_text} {context_text}"

        # Flexible bidirectional matching
        matches = sum(1 for word in query_words if word in full_text or concept_text in word)
        if matches > 0:
            chunk_refs.update(refs)
            for ref in refs:
                chunk_scores[ref] = chunk_scores.get(ref, 0) + matches * 2  # Weight concepts highly

    # Extract chunk refs from matching facts
    for fact in summary.get("key_facts", []):
        fact_text = fact.get("fact", "").lower()
        refs = fact.get("chunk_refs", [])

        # Count matching keywords
        matches = sum(1 for word in query_words if word in fact_text)
        if matches > 0:
            chunk_refs.update(refs)
            for ref in refs:
                chunk_scores[ref] = chunk_scores.get(ref, 0) + matches

    # If no matches, return first few chunks (let semantic search handle it)
    if not chunk_refs:
        logger.warning(f"No keyword matches in summary for '{query}', returning first chunks")
        total_chunks = summary.get("chunk_count", 0)
        return list(range(min(max_chunks, total_chunks)))

    # Sort by score (most matches first), then return top N
    sorted_refs = sorted(chunk_refs, key=lambda x: chunk_scores.get(x, 0), reverse=True)
    return sorted_refs[:max_chunks]


def fetch_chunks_by_refs(
    doc_id: str,
    chunk_refs: List[int],
    namespace: str = "default"
) -> List[Dict[str, Any]]:
    """
    Fetch specific chunks by their indices.

    Args:
        doc_id: Document identifier
        chunk_refs: List of chunk indices to fetch
        namespace: Pinecone namespace

    Returns:
        List of chunk dicts with text and metadata
    """
    chunk_ids = [f"{doc_id}_chunk_{ref}" for ref in chunk_refs]

    try:
        response = index.fetch(ids=chunk_ids, namespace=namespace)

        chunks = []
        for chunk_id, vector_data in (response.vectors or {}).items():
            # Handle both dict and object-style Vector data (SDK compatibility)
            if hasattr(vector_data, 'metadata'):
                metadata = getattr(vector_data, 'metadata', {}) or {}
            else:
                metadata = vector_data.get("metadata", {})

            chunks.append({
                "id": chunk_id,
                "text": metadata.get("text", ""),
                "metadata": metadata,
                "chunk_index": int(chunk_id.split("_chunk_")[-1])
            })

        # Sort by chunk index
        chunks.sort(key=lambda x: x["chunk_index"])

        logger.info(f"Fetched {len(chunks)}/{len(chunk_ids)} chunks for {doc_id}")
        return chunks

    except Exception as e:
        logger.error(f"Failed to fetch chunks for {doc_id}: {e}")
        return []


def fetch_full_document(
    doc_id: str,
    namespace: str = "default",
    max_chunks: int = 50
) -> Optional[str]:
    """
    Fetch the entire document text by retrieving all chunks.

    This provides the full context for LLM reasoning instead of relying on
    semantic search snippets.

    Args:
        doc_id: Document identifier
        namespace: Pinecone namespace
        max_chunks: Maximum number of chunks to fetch (safety limit)

    Returns:
        Full document text or None if failed
    """
    try:
        # Get summary to know chunk count
        summary = get_document_summary(doc_id, namespace)
        if not summary:
            logger.warning(f"No summary found for {doc_id}, cannot determine chunk count")
            return None

        chunk_count = summary.get("chunk_count", 0)
        if chunk_count == 0:
            logger.warning(f"Document {doc_id} has 0 chunks")
            return None

        # Fetch all chunks (up to max_chunks)
        chunks_to_fetch = min(chunk_count, max_chunks)
        chunk_refs = list(range(chunks_to_fetch))

        chunks = fetch_chunks_by_refs(doc_id, chunk_refs, namespace)

        if not chunks:
            logger.error(f"Failed to fetch any chunks for {doc_id}")
            return None

        # Combine all chunk text
        full_text = "\n\n".join(chunk["text"] for chunk in chunks)

        logger.info(f"Fetched full document for {doc_id}: {len(chunks)} chunks, {len(full_text)} chars")
        return full_text

    except Exception as e:
        logger.error(f"Failed to fetch full document for {doc_id}: {e}")
        return None
