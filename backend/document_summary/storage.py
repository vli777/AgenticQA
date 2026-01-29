# backend/document_summary/storage.py

"""Pinecone CRUD operations for document summaries."""

import json
from typing import List, Dict, Any, Optional

from logger import logger
from pinecone_client import index, EMBED_DIM
from config import EMBEDDING_MODEL
from utils.embeddings import get_embedding, get_embeddings_batch


def store_document_summary(summary: Dict[str, Any], namespace: str = "default") -> None:
    """
    Store document summary in Pinecone as a special metadata-only vector.

    Uses a summary embedding (embedding of the primary_subject + topics) so it can be
    retrieved semantically if needed, but primarily accessed by ID.
    """
    doc_id = summary["doc_id"]
    summary_id = f"{doc_id}__SUMMARY__"

    # Create embedding for the summary (for potential semantic retrieval)
    summary_text = f"{summary['primary_subject']} Topics: {', '.join(summary['topics'])}"
    summary_embedding = get_embedding(summary_text, EMBEDDING_MODEL)

    # Store in Pinecone with special ID
    try:
        index.upsert(
            vectors=[{
                "id": summary_id,
                "values": summary_embedding,
                "metadata": {
                    "type": "document_summary",
                    "doc_id": doc_id,
                    "summary_json": json.dumps(summary)  # Store full summary as JSON string
                }
            }],
            namespace=namespace
        )
        logger.info(f"Stored summary for {doc_id} in namespace '{namespace}'")
    except Exception as e:
        logger.error(f"Failed to store summary for {doc_id}: {e}")


def store_document_summaries_batch(summaries: List[Dict[str, Any]], namespace: str = "default") -> None:
    """
    Store multiple document summaries in Pinecone with batch embedding generation (optimization).

    Args:
        summaries: List of summary dicts
        namespace: Pinecone namespace
    """
    if not summaries:
        return

    # Generate embeddings for all summaries in batch
    summary_texts = [
        f"{s['primary_subject']} Topics: {', '.join(s['topics'])}"
        for s in summaries
    ]

    try:
        summary_embeddings = get_embeddings_batch(summary_texts, EMBEDDING_MODEL)
    except Exception as e:
        logger.warning(f"Batch embedding generation failed for summaries, falling back to sequential: {e}")
        summary_embeddings = [get_embedding(text, EMBEDDING_MODEL) for text in summary_texts]

    # Build vectors for batch upsert
    vectors = []
    for summary, embedding in zip(summaries, summary_embeddings):
        doc_id = summary["doc_id"]
        summary_id = f"{doc_id}__SUMMARY__"

        vectors.append({
            "id": summary_id,
            "values": embedding,
            "metadata": {
                "type": "document_summary",
                "doc_id": doc_id,
                "summary_json": json.dumps(summary)
            }
        })

    # Batch upsert to Pinecone
    try:
        index.upsert(vectors=vectors, namespace=namespace)
        logger.info(f"Batch stored {len(summaries)} summaries in namespace '{namespace}'")
    except Exception as e:
        logger.error(f"Failed to batch store summaries: {e}")


def store_cross_document_summary(
    cross_summary: Dict[str, Any],
    namespace: str = "default"
) -> None:
    """Store cross-document summary in Pinecone."""
    summary_id = f"__CROSS_DOC__{cross_summary['primary_doc']}"

    # Create embedding from shared topics
    summary_text = f"Cross-document summary. Shared topics: {', '.join(cross_summary['shared_topics'])}"
    summary_embedding = get_embedding(summary_text, EMBEDDING_MODEL)

    try:
        index.upsert(
            vectors=[{
                "id": summary_id,
                "values": summary_embedding,
                "metadata": {
                    "type": "cross_document_summary",
                    "summary_json": json.dumps(cross_summary)
                }
            }],
            namespace=namespace
        )
        logger.info(f"Stored cross-document summary for {cross_summary['primary_doc']}")
    except Exception as e:
        logger.error(f"Failed to store cross-document summary: {e}")


def get_document_summary(doc_id: str, namespace: str = "default") -> Optional[Dict[str, Any]]:
    """
    Retrieve document summary from Pinecone.

    Args:
        doc_id: Document identifier
        namespace: Pinecone namespace

    Returns:
        Summary dict or None if not found
    """
    summary_id = f"{doc_id}__SUMMARY__"

    try:
        # Fetch by ID
        response = index.fetch(ids=[summary_id], namespace=namespace)

        if not response.vectors or summary_id not in response.vectors:
            logger.warning(f"No summary found for {doc_id} in namespace '{namespace}'")
            return None

        vector_data = response.vectors[summary_id]

        # Handle both dict and object-style Vector data (SDK compatibility)
        if hasattr(vector_data, 'metadata'):
            metadata = getattr(vector_data, 'metadata', {}) or {}
        else:
            metadata = vector_data.get("metadata", {})

        if "summary_json" in metadata:
            summary = json.loads(metadata["summary_json"])
            return summary
        else:
            logger.warning(f"Summary metadata missing for {doc_id}")
            return None

    except Exception as e:
        logger.error(f"Failed to retrieve summary for {doc_id}: {e}")
        return None


def list_documents_in_namespace(namespace: str = "default", limit: int = 100) -> List[Dict[str, str]]:
    """
    List all documents in a namespace by finding summary entries.

    Returns:
        List of dicts with doc_id and source
    """
    try:
        # Query for all summary entries using a dummy vector
        # Use EMBED_DIM from pinecone_client (controlled by VECTOR_DIMENSION env var)
        dummy_vector = [0.0] * EMBED_DIM

        response = index.query(
            vector=dummy_vector,
            top_k=limit,
            include_metadata=True,
            namespace=namespace,
            filter={"type": "document_summary"}
        )

        documents = []
        for match in response.get("matches", []):
            metadata = match.get("metadata", {})
            if "summary_json" in metadata:
                summary = json.loads(metadata["summary_json"])
                documents.append({
                    "doc_id": summary["doc_id"],
                    "source": summary["source"],
                    "document_type": summary.get("document_type", "other"),
                    "topics": summary.get("topics", []),
                    "extracted_at": summary.get("extracted_at", "")
                })

        logger.info(f"Found {len(documents)} documents in namespace '{namespace}'")
        return documents

    except Exception as e:
        logger.error(f"Failed to list documents in namespace '{namespace}': {e}")
        return []


def search_summaries(
    query: str,
    namespace: str = "default",
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Search document summaries semantically to find relevant facts.

    This is the FIRST TIER of retrieval - fast summary search that returns
    chunk references for detailed evidence retrieval.

    Args:
        query: Search query
        namespace: Pinecone namespace
        top_k: Number of summaries to return

    Returns:
        List of summaries with relevant facts and chunk references
    """
    try:
        # Get query embedding
        query_embedding = get_embedding(query, EMBEDDING_MODEL)

        # Search for summaries only
        response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
            filter={"type": "document_summary"}
        )

        results = []
        for match in response.get("matches", []):
            metadata = match.get("metadata", {})
            if "summary_json" in metadata:
                summary = json.loads(metadata["summary_json"])
                results.append({
                    "doc_id": summary["doc_id"],
                    "source": summary["source"],
                    "summary": summary,
                    "score": match.get("score", 0.0)
                })

        logger.info(f"Summary search for '{query}' found {len(results)} documents")
        return results

    except Exception as e:
        logger.error(f"Summary search failed: {e}")
        return []
