# backend/routes/debug_summary.py

from fastapi import APIRouter
from document_summary import list_documents_in_namespace, get_document_summary

router = APIRouter()


@router.get("/summaries/{namespace}")
async def list_summaries(namespace: str = "default"):
    """Debug endpoint to see all document summaries."""
    docs = list_documents_in_namespace(namespace, limit=50)
    return {"documents": docs}


@router.get("/summary/{namespace}/{doc_id}")
async def get_summary_detail(namespace: str, doc_id: str):
    """Debug endpoint to see detailed summary for a document."""
    summary = get_document_summary(doc_id, namespace)
    if not summary:
        return {"error": f"No summary found for {doc_id}"}
    return summary
