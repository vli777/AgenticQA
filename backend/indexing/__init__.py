# backend/indexing/__init__.py

"""Document indexing: chunk embedding and Pinecone upsert."""

from .upsert import upsert_doc

__all__ = ["upsert_doc"]
