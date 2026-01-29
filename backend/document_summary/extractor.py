# backend/document_summary/extractor.py

"""LLM-based structured summary extraction from document chunks."""

from typing import List, Dict, Any
from datetime import datetime

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from logger import logger
from config import EMBEDDING_MODEL, LLM_SUMMARY_MODEL
from circuit_breaker import CircuitBreaker
from structured_output import get_document_summary_schema, get_batch_document_summary_schema, extract_with_nvidia_guided_json

llm = ChatNVIDIA(model=LLM_SUMMARY_MODEL, temperature=0.0, max_tokens=8192)

# Circuit breaker for failed summary extractions
summary_circuit = CircuitBreaker(
    name="summary",
    threshold=3,
    error_message="An error occurred during summary generation",
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
    summary_circuit.check()

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

        summary_circuit.record_success()
        logger.info(
            f"Extracted summary for {doc_id}: {len(summary['key_concepts'])} concepts, "
            f"{len(summary['key_facts'])} facts, {len(summary['topics'])} topics, type={summary['document_type']}"
        )

        return summary

    except Exception as e:
        logger.error(f"Summary extraction failed for {doc_id}: {e}")
        summary_circuit.record_failure()

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
    if not documents:
        return []

    summary_circuit.check()

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
                summary_circuit.record_success()

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
            summary_circuit.record_failure()

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
