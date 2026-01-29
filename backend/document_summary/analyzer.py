# backend/document_summary/analyzer.py

"""Cross-document overlap detection and analysis."""

from typing import Dict, Any, Optional
from datetime import datetime

from logger import logger
from .storage import list_documents_in_namespace


def detect_cross_document_overlap(
    new_summary: Dict[str, Any],
    namespace: str = "default",
    overlap_threshold: float = 0.3
) -> Optional[Dict[str, Any]]:
    """
    Detect if the new document has significant topic overlap with existing documents.

    Args:
        new_summary: Summary of newly uploaded document
        namespace: Pinecone namespace
        overlap_threshold: Minimum topic overlap ratio to trigger cross-doc summary

    Returns:
        Cross-document summary if overlap detected, None otherwise
    """
    existing_docs = list_documents_in_namespace(namespace)

    # Filter out the current document
    existing_docs = [d for d in existing_docs if d["doc_id"] != new_summary["doc_id"]]

    if not existing_docs:
        logger.info("No existing documents to compare for cross-document summary")
        return None

    new_topics = set(t.lower() for t in new_summary.get("topics", []))

    if not new_topics:
        logger.info("New document has no topics, skipping cross-document analysis")
        return None

    # Find overlapping documents
    overlapping_docs = []
    for doc in existing_docs:
        doc_topics = set(t.lower() for t in doc.get("topics", []))
        if not doc_topics:
            continue

        overlap = new_topics & doc_topics
        overlap_ratio = len(overlap) / len(new_topics | doc_topics)

        if overlap_ratio >= overlap_threshold:
            overlapping_docs.append({
                "doc_id": doc["doc_id"],
                "source": doc["source"],
                "shared_topics": list(overlap),
                "overlap_ratio": overlap_ratio
            })

    if not overlapping_docs:
        logger.info("No significant topic overlap detected, skipping cross-document summary")
        return None

    logger.info(
        f"Found {len(overlapping_docs)} overlapping documents for {new_summary['doc_id']}: "
        f"{[d['doc_id'] for d in overlapping_docs]}"
    )

    # Generate cross-document summary
    cross_summary = {
        "type": "cross_document_summary",
        "namespace": namespace,
        "primary_doc": new_summary["doc_id"],
        "related_docs": [d["doc_id"] for d in overlapping_docs],
        "shared_topics": list(set(
            topic for doc in overlapping_docs for topic in doc["shared_topics"]
        )),
        "overlap_details": overlapping_docs,
        "created_at": datetime.utcnow().isoformat()
    }

    return cross_summary
