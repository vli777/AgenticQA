# backend/agent/tools.py

"""
Agent tool factories for the QA pipeline.

Each tool is a self-contained factory function that returns a LangChain Tool.
This design enables future migration to MCP tool servers or agent skills:
each factory can be extracted into its own module/server with minimal changes.
"""

import re
from typing import List

from langchain_core.tools import Tool

from config import VECTOR_K
from utils.text_processing import clean_text
from document_summary import (
    list_documents_in_namespace,
    get_document_summary,
)

from .models import MAX_MATCHES_TO_RETURN


def _list_documents_tool(namespace: str = "default") -> Tool:
    """
    Tool to list all documents in the namespace.
    Useful for answering "what files do you have?" or "what documents are available?"
    """
    def list_fn(query: str = "") -> str:
        docs = list_documents_in_namespace(namespace, limit=50)

        if not docs:
            return "No documents found in this namespace."

        # Format as readable list
        lines = [f"Found {len(docs)} document(s):\n"]
        for i, doc in enumerate(docs, 1):
            topics_str = ", ".join(doc.get("topics", [])[:3])
            lines.append(
                f"{i}. {doc['source']} (ID: {doc['doc_id']}, Type: {doc.get('document_type', 'unknown')}, "
                f"Topics: {topics_str})"
            )

        return "\n".join(lines)

    return Tool(
        name="list_documents",
        func=list_fn,
        description="List all available documents in the knowledge base. Use this when the user asks 'what files do you have?' or 'what documents are available?' or 'show me what's uploaded'."
    )


def _get_document_summary_tool(namespace: str = "default") -> Tool:
    """
    Tool to get structured summary with citations for a specific document.
    This should be the PRIMARY tool for broad questions about a document.
    """
    def summary_fn(doc_id: str) -> str:
        summary = get_document_summary(doc_id, namespace)

        if not summary:
            return f"No summary found for document '{doc_id}'. Use list_documents to see available documents."

        # Format summary with citations
        lines = [
            f"Document: {summary['source']}",
            f"Type: {summary.get('document_type', 'unknown')}",
            f"Subject: {summary.get('primary_subject', 'N/A')}",
            f"\nTopics: {', '.join(summary.get('topics', []))}",
            "\nKey Concepts:"
        ]

        for concept in summary.get("key_concepts", [])[:10]:
            chunk_refs = concept.get("chunk_refs", [])
            context = concept.get("context", "N/A")
            refs_str = f"[chunks: {', '.join(map(str, chunk_refs))}]" if chunk_refs else ""
            lines.append(f"  - {concept.get('concept', 'N/A')}: {context} {refs_str}")

        lines.append("\nKey Facts:")
        for fact in summary.get("key_facts", [])[:15]:
            chunk_refs = fact.get("chunk_refs", [])
            refs_str = f"[chunks: {', '.join(map(str, chunk_refs))}]" if chunk_refs else ""
            lines.append(f"  - {fact.get('fact', 'N/A')} {refs_str}")

        lines.append(f"\nTotal chunks: {summary.get('chunk_count', 0)}")

        return "\n".join(lines)

    return Tool(
        name="get_document_summary",
        func=summary_fn,
        description="Get a structured summary of a specific document with key facts and entities. Use this for broad questions like 'what is this about?', 'summarize this document', or 'what are the main points?'. Input should be the doc_id from list_documents."
    )


def _pinecone_search_tool(namespace: str = "default") -> Tool:
    """
    Vector search tool using NVIDIA Llama 3.2 embeddings with reranking.
    Returns top-ranked evidence snippets with provenance tags.
    """

    def search_fn(query: str) -> str:
        # Use hybrid search with re-ranking
        # Use sync wrapper to avoid event loop conflicts
        from hybrid_search import hybrid_search_sync

        reranked_results = hybrid_search_sync(
            query=query,
            namespace=namespace,
            top_k=MAX_MATCHES_TO_RETURN,
            vector_k=VECTOR_K
        )

        if not reranked_results:
            return "No results."

        # Format provenance-tagged snippets
        lines: List[str] = []
        for result in reranked_results:
            md = result.get("metadata") or {}
            text = clean_text(md.get("text") or "")
            if not text or len(text) < 50:
                continue
            source = md.get("source") or md.get("file_name") or "unknown"
            section = md.get("section_index") or md.get("chunk_id")
            score = result.get("rerank_score", 0.0)
            prov = f"{source}::section-{section}" if section is not None else source

            # Trim to ~1000 chars on sentence boundaries
            if len(text) > 1000:
                pieces = re.split(r"([.!?])\s+", text)
                buf = ""
                for i in range(0, len(pieces), 2):
                    s = pieces[i] + (pieces[i + 1] if i + 1 < len(pieces) else "")
                    if len(buf) + len(s) > 1000:
                        break
                    buf += s + " "
                text = buf.strip()

            lines.append(f"[{prov}] (rerank_score: {score:.3f}) {text}")

        return "\n\n".join(lines) if lines else "No meaningful text chunks found."

    return Tool(
        name="semantic_search",
        func=search_fn,
        description="Search the indexed knowledge base using NVIDIA Llama 3.2 vector embeddings with reranking. Returns up to 3 evidence snippets with provenance like [source::section-X] ...",
    )
