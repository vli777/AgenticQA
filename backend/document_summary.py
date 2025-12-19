# backend/document_summary.py

import json
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime

from pydantic import BaseModel, Field
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from logger import logger
from pinecone_client import index, EMBED_DIM
from config import EMBEDDING_MODEL, LLM_SUMMARY_MODEL
from utils import get_embedding, get_embeddings_batch
from exceptions import CircuitBreakerOpenError
from structured_output import get_document_summary_schema, get_batch_document_summary_schema, extract_with_nvidia_guided_json

llm = ChatNVIDIA(model=LLM_SUMMARY_MODEL, temperature=0.0)

# Circuit breaker for failed summary extractions
_summary_failure_count = 0
_summary_circuit_open = False
SUMMARY_FAILURE_THRESHOLD = 3


# ---------------------------
# Pydantic Models for Structured Output
# ---------------------------

class KeyConcept(BaseModel):
    """A key concept or important term from the document."""
    concept: str = Field(description="Important term, technology, name, idea, or skill")
    context: str = Field(description="Brief context about how it's used or mentioned")
    chunk_refs: List[int] = Field(description="Reference to [CHUNK_X] numbers where information was found")


class KeyFact(BaseModel):
    """An important statement or claim from the document."""
    fact: str = Field(description="Important statement or claim")
    chunk_refs: List[int] = Field(description="Reference to [CHUNK_X] numbers where information was found")


class DocumentExtraction(BaseModel):
    """Structured extraction of key information from a document."""
    document_type: Literal["resume", "research_paper", "technical_doc", "article", "business_doc", "other"] = Field(
        description="Type of document"
    )
    primary_subject: str = Field(
        description="Brief description of what this document is about"
    )
    key_concepts: List[KeyConcept] = Field(
        description="List of important concepts mentioned (skills, technologies, methods, people, companies, theories, tools, etc.)"
    )
    key_facts: List[KeyFact] = Field(
        description="List of 5-15 most important facts that answer potential questions"
    )
    topics: List[str] = Field(
        description="List of 3-5 main topics or themes"
    )


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
    global _summary_failure_count, _summary_circuit_open

    # Circuit breaker: if too many failures, abort processing
    if _summary_circuit_open:
        logger.error(
            f"Summary circuit breaker open ({_summary_failure_count} consecutive failures), "
            f"aborting processing"
        )
        raise CircuitBreakerOpenError("An error occurred during summary generation")

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
    extraction_prompt = f"""Extract key information from this document.

Document: {source}

Text:
{combined_text}

Important:
- chunk_refs should reference the [CHUNK_X] numbers where information was found
- Extract 5-15 most important facts that answer potential questions
- key_concepts: Extract ALL important concepts mentioned (skills, technologies, methods, people, companies, theories, tools, etc.)
- For each concept, provide brief context about how it's used or mentioned
- Identify 3-5 main topics/themes
- Be comprehensive and specific
- This should work for ANY document type (resumes, papers, articles, documentation, etc.)"""

    # Use NVIDIA's guided_json for structured output
    try:
        doc_schema = get_document_summary_schema()
        extracted_dict = extract_with_nvidia_guided_json(llm, extraction_prompt, doc_schema)

        # Build summary structure
        summary = {
            "type": "document_summary",
            "doc_id": doc_id,
            "source": source,
            "namespace": namespace,
            "document_type": extracted_dict.get("document_type", "other"),
            "primary_subject": extracted_dict.get("primary_subject", f"Document: {source}"),
            "key_concepts": extracted_dict.get("key_concepts", []),
            "key_facts": extracted_dict.get("key_facts", []),
            "topics": extracted_dict.get("topics", []),
            "chunk_count": len(chunks),
            "extracted_at": datetime.utcnow().isoformat(),
            "embedding_model": EMBEDDING_MODEL
        }

        # Success - reset failure counter
        _summary_failure_count = 0
        _summary_circuit_open = False
        logger.info(
            f"Extracted summary for {doc_id}: {len(summary['key_concepts'])} concepts, "
            f"{len(summary['key_facts'])} facts, {len(summary['topics'])} topics, type={summary['document_type']}"
        )

        return summary

    except Exception as e:
        logger.error(f"Summary extraction failed for {doc_id}: {e}")
        _summary_failure_count += 1
        if _summary_failure_count >= SUMMARY_FAILURE_THRESHOLD:
            _summary_circuit_open = True
            logger.error(f"Summary circuit breaker opened after {_summary_failure_count} consecutive failures")

        # Return minimal summary
        return {
            "type": "document_summary",
            "doc_id": doc_id,
            "source": source,
            "namespace": namespace,
            "document_type": "other",
            "primary_subject": f"Document: {source}",
            "key_concepts": [],
            "key_facts": [],
            "topics": [],
            "chunk_count": len(chunks),
            "extracted_at": datetime.utcnow().isoformat(),
            "embedding_model": EMBEDDING_MODEL,
            "extraction_error": str(e)
        }


def extract_structured_summaries_batch(
    documents: List[Dict[str, Any]],
    namespace: str = "default",
    batch_size: int = 3
) -> List[Dict[str, Any]]:
    """
    Extract structured summaries for multiple documents in batches (optimization).

    Args:
        documents: List of dicts with 'doc_id', 'chunks', 'source' keys
        namespace: Pinecone namespace
        batch_size: Number of documents to process per LLM call (default 3)

    Returns:
        List of summary dicts, one for each document
    """
    global _summary_failure_count, _summary_circuit_open

    if not documents:
        return []

    # Circuit breaker check
    if _summary_circuit_open:
        logger.error(
            f"Summary circuit breaker open ({_summary_failure_count} consecutive failures), "
            f"aborting batch processing"
        )
        raise CircuitBreakerOpenError("An error occurred during summary generation")

    all_summaries: List[Dict[str, Any]] = []
    max_chars_per_doc = 8000  # Same as single extraction

    # Process in batches
    for batch_start in range(0, len(documents), batch_size):
        batch_docs = documents[batch_start:batch_start + batch_size]

        try:
            # Build combined prompt with all documents
            doc_texts = []
            doc_metadata = []

            for doc_idx, doc in enumerate(batch_docs):
                doc_id = doc['doc_id']
                chunks = doc['chunks']
                source = doc['source']

                # Combine chunks for this document
                combined_text = ""
                chunk_map = {}
                for i, chunk in enumerate(chunks):
                    if len(combined_text) + len(chunk) > max_chars_per_doc:
                        break
                    combined_text += f"\n\n[CHUNK_{i}]\n{chunk}"
                    chunk_map[i] = chunk

                doc_texts.append(f"=== DOCUMENT {doc_idx + 1}: {source} ===\n{combined_text}")
                doc_metadata.append({
                    'doc_id': doc_id,
                    'source': source,
                    'chunks': chunks,
                    'chunk_count': len(chunks)
                })

            # Create batch prompt
            combined_prompt = "Extract key information from each of the following documents.\n\n"
            combined_prompt += "\n\n".join(doc_texts)
            combined_prompt += """

Important:
- chunk_refs should reference the [CHUNK_X] numbers where information was found
- Extract 5-15 most important facts that answer potential questions for EACH document
- key_concepts: Extract ALL important concepts mentioned (skills, technologies, methods, people, companies, theories, tools, etc.)
- For each concept, provide brief context about how it's used or mentioned
- Identify 3-5 main topics/themes for EACH document
- Be comprehensive and specific
- This should work for ANY document type (resumes, papers, articles, documentation, etc.)
- Return one summary object for each document in the same order"""

            # Extract with batch schema
            batch_schema = get_batch_document_summary_schema(len(batch_docs))
            batch_result = extract_with_nvidia_guided_json(llm, combined_prompt, batch_schema)

            if "documents" in batch_result and isinstance(batch_result["documents"], list):
                # Success - reset failure counter
                _summary_failure_count = 0
                _summary_circuit_open = False

                # Build summary objects
                for i, (extracted_dict, meta) in enumerate(zip(batch_result["documents"], doc_metadata)):
                    summary = {
                        "type": "document_summary",
                        "doc_id": meta['doc_id'],
                        "source": meta['source'],
                        "namespace": namespace,
                        "document_type": extracted_dict.get("document_type", "other"),
                        "primary_subject": extracted_dict.get("primary_subject", f"Document: {meta['source']}"),
                        "key_concepts": extracted_dict.get("key_concepts", []),
                        "key_facts": extracted_dict.get("key_facts", []),
                        "topics": extracted_dict.get("topics", []),
                        "chunk_count": meta['chunk_count'],
                        "extracted_at": datetime.utcnow().isoformat(),
                        "embedding_model": EMBEDDING_MODEL
                    }
                    all_summaries.append(summary)
                    logger.info(
                        f"Batch extracted summary for {meta['doc_id']}: "
                        f"{len(summary['key_concepts'])} concepts, "
                        f"{len(summary['key_facts'])} facts, "
                        f"{len(summary['topics'])} topics"
                    )
            else:
                logger.warning(f"Invalid batch summary format, falling back to individual extraction")
                # Fallback to individual extraction for this batch
                for doc in batch_docs:
                    summary = extract_structured_summary(
                        doc['doc_id'],
                        doc['chunks'],
                        doc['source'],
                        namespace
                    )
                    all_summaries.append(summary)

        except Exception as e:
            logger.error(f"Batch summary extraction failed for batch starting at {batch_start}: {e}")
            _summary_failure_count += 1
            if _summary_failure_count >= SUMMARY_FAILURE_THRESHOLD:
                _summary_circuit_open = True
                logger.error(f"Summary circuit breaker opened after {_summary_failure_count} consecutive failures")

            # Return minimal summaries for this batch
            for doc in batch_docs:
                all_summaries.append({
                    "type": "document_summary",
                    "doc_id": doc['doc_id'],
                    "source": doc['source'],
                    "namespace": namespace,
                    "document_type": "other",
                    "primary_subject": f"Document: {doc['source']}",
                    "key_concepts": [],
                    "key_facts": [],
                    "topics": [],
                    "chunk_count": len(doc['chunks']),
                    "extracted_at": datetime.utcnow().isoformat(),
                    "embedding_model": EMBEDDING_MODEL,
                    "extraction_error": str(e)
                })

    return all_summaries


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
