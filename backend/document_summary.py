# backend/document_summary.py

import json
from typing import List, Dict, Any, Set, Optional
from datetime import datetime

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from logger import logger
from pinecone_client import index
from config import EMBEDDING_MODEL
from utils import get_embedding

llm = ChatNVIDIA(model="meta/llama-4-maverick-17b-128e-instruct", temperature=0.0)


def extract_structured_summary(
    doc_id: str,
    chunks: List[str],
    source: str,
    namespace: str = "default"
) -> Dict[str, Any]:
    """
    Extract structured facts, entities, and topics from document chunks.

    Args:
        doc_id: Document identifier
        chunks: List of text chunks
        source: Original filename
        namespace: Pinecone namespace

    Returns:
        Structured summary with citations
    """
    # Combine chunks for extraction (limit to reasonable size)
    max_chars = 8000  # ~4000 tokens for LLM context
    combined_text = ""
    chunk_map = {}  # Maps chunk index to text

    for i, chunk in enumerate(chunks):
        if len(combined_text) + len(chunk) > max_chars:
            break
        combined_text += f"\n\n[CHUNK_{i}]\n{chunk}"
        chunk_map[i] = chunk

    # Structured extraction prompt (generic, works for any document type)
    extraction_prompt = f"""Extract key information from this document in strict JSON format.

Document: {source}

Text:
{combined_text}

Extract the following and return ONLY valid JSON:
{{
  "document_type": "resume|research_paper|technical_doc|article|business_doc|other",
  "primary_subject": "brief description of what this document is about",
  "key_concepts": [
    {{"concept": "important term/technology/name/idea", "context": "brief context", "chunk_refs": [0, 1]}}
  ],
  "key_facts": [
    {{"fact": "important statement or claim", "chunk_refs": [0]}}
  ],
  "topics": ["topic1", "topic2"]
}}

Important:
- chunk_refs should reference the [CHUNK_X] numbers where information was found
- Extract 5-15 most important facts that answer potential questions
- key_concepts: Extract ALL important concepts mentioned (skills, technologies, methods, people, companies, theories, tools, etc.)
- For each concept, provide brief context about how it's used or mentioned
- Identify 3-5 main topics/themes
- Be comprehensive and specific
- This should work for ANY document type (resumes, papers, articles, documentation, etc.)

Return ONLY the JSON, nothing else."""

    try:
        response = llm.invoke(extraction_prompt)
        extracted_text = response.content.strip()

        # Clean up markdown code blocks if present
        if extracted_text.startswith("```"):
            extracted_text = extracted_text.split("```")[1]
            if extracted_text.startswith("json"):
                extracted_text = extracted_text[4:]

        extracted = json.loads(extracted_text)

        # Build summary structure
        summary = {
            "type": "document_summary",
            "doc_id": doc_id,
            "source": source,
            "namespace": namespace,
            "document_type": extracted.get("document_type", "other"),
            "primary_subject": extracted.get("primary_subject", ""),
            "key_concepts": extracted.get("key_concepts", []),
            "key_facts": extracted.get("key_facts", []),
            "topics": extracted.get("topics", []),
            "chunk_count": len(chunks),
            "extracted_at": datetime.utcnow().isoformat(),
            "embedding_model": EMBEDDING_MODEL
        }

        logger.info(
            f"Extracted summary for {doc_id}: {len(summary['key_concepts'])} concepts, "
            f"{len(summary['key_facts'])} facts, {len(summary['topics'])} topics, type={summary['document_type']}"
        )

        return summary

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse extraction JSON for {doc_id}: {e}")
        logger.error(f"Raw response: {extracted_text[:500]}")
        # Return minimal summary
        return {
            "type": "document_summary",
            "doc_id": doc_id,
            "source": source,
            "namespace": namespace,
            "document_type": "other",
            "primary_subject": f"Document: {source}",
            "key_entities": [],
            "key_facts": [],
            "topics": [],
            "chunk_count": len(chunks),
            "extracted_at": datetime.utcnow().isoformat(),
            "embedding_model": EMBEDDING_MODEL,
            "extraction_error": str(e)
        }
    except Exception as e:
        logger.error(f"Summary extraction failed for {doc_id}: {e}")
        return {
            "type": "document_summary",
            "doc_id": doc_id,
            "source": source,
            "namespace": namespace,
            "document_type": "other",
            "primary_subject": f"Document: {source}",
            "key_entities": [],
            "key_facts": [],
            "topics": [],
            "chunk_count": len(chunks),
            "extracted_at": datetime.utcnow().isoformat(),
            "embedding_model": EMBEDDING_MODEL,
            "extraction_error": str(e)
        }


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
        if EMBEDDING_MODEL == "text-embedding-3-small":
            dimension = 1536
        else:
            dimension = 1024

        dummy_vector = [0.0] * dimension

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
