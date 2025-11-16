# backend/hybrid_search.py

from typing import List, Dict, Any, Optional, Set
import re

from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_core.documents import Document

from logger import logger
from pinecone_client import index, EMBED_DIM
from config import EMBEDDING_MODEL, SEMANTIC_TAG_BOOST, NVIDIA_API_KEY
from utils import get_embedding
from semantic_tags import infer_query_tags


class HybridSearchEngine:
    """
    NVIDIA-optimized vector search engine with embeddings and reranker.

    Uses llama-3.2-nv-embedqa-1b-v2 for embeddings and nvidia/llama-3.2-nv-rerankqa-1b-v2 for reranking.

    Benchmarks (MLQA dataset):
    - This pipeline: 86.83% recall@5
    - Embeddings only: 79.86% recall@5
    - BM25 only: 13.01% recall@5
    """

    def __init__(self, reranker_model: str = None):
        """
        Initialize the NVIDIA-optimized search engine.

        Args:
            reranker_model: NVIDIA reranker model (optional, defaults to nvidia/llama-3.2-nv-rerankqa-1b-v2)
        """
        # Use NVIDIA hosted reranker (no local dependencies!)
        self.reranker = None
        if NVIDIA_API_KEY:
            try:
                model_name = reranker_model or "nvidia/llama-3.2-nv-rerankqa-1b-v2"
                self.reranker = NVIDIARerank(
                    model=model_name,
                    api_key=NVIDIA_API_KEY
                )
                logger.info(f"Initialized HybridSearchEngine with NVIDIA reranker: {model_name} (hosted, 0MB image overhead)")
            except Exception as e:
                logger.warning(f"Failed to initialize NVIDIA reranker: {e}. Falling back to hybrid score sorting.")
                self.reranker = None
        else:
            logger.info("NVIDIA_API_KEY not set, reranker disabled. Using hybrid score sorting.")

    def _fetch_tagged_documents(
        self,
        namespace: str,
        tags: Optional[Set[str]],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Fetch additional documents that share inferred semantic tags."""
        if not tags:
            return []

        # Use EMBED_DIM from pinecone_client (controlled by VECTOR_DIMENSION env var)
        try:
            response = index.query(
                vector=[0.0] * EMBED_DIM,
                top_k=limit,
                include_metadata=True,
                namespace=namespace,
                filter={"semantic_tags": {"$in": list(tags)}},
            )
        except Exception as exc:
            logger.warning("Failed to fetch semantic-tagged documents: %s", exc)
            return []

        docs: List[Dict[str, Any]] = []
        for match in response.get("matches", []):
            docs.append({
                "id": match.get("id"),
                "score": match.get("score", 0.0),
                "metadata": match.get("metadata", {}),
            })
        if docs:
            logger.info("Fetched %s additional candidates via semantic tags %s", len(docs), tags)
        return docs

    async def vector_search(
        self,
        query: str,
        namespace: str = "default",
        top_k: int = 20,
        query_tags: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using Pinecone.

        Args:
            query: Search query
            namespace: Pinecone namespace
            top_k: Number of results to return

        Returns:
            List of matches with scores
        """
        # Get embedding for query
        q_embed = get_embedding(text=query, model=EMBEDDING_MODEL)

        # Query Pinecone with the embedding vector
        response = index.query(
            vector=q_embed,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )

        # Handle response (could be dict or object depending on SDK version)
        if hasattr(response, 'matches'):
            matches = response.matches or []
        else:
            matches = response.get("matches", [])
        results = []
        for match in matches:
            # Handle both dict and object matches
            if hasattr(match, 'id'):
                # Object-style match (newer SDK)
                results.append({
                    "id": match.id,
                    "score": getattr(match, 'score', 0.0),
                    "metadata": getattr(match, 'metadata', {}) or {}
                })
            else:
                # Dict-style match (older SDK)
                results.append({
                    "id": match.get("id"),
                    "score": match.get("score", 0.0),
                    "metadata": match.get("metadata", {})
                })

        # If the user query implies semantic tags, fetch extra candidates that
        # share those tags (even if they don't rank highly by pure similarity).
        tagged_additions = self._fetch_tagged_documents(namespace, query_tags)
        if tagged_additions:
            existing_ids = {r["id"] for r in results}
            for doc in tagged_additions:
                if doc["id"] not in existing_ids:
                    results.append(doc)

        return results

    async def hybrid_search(
        self,
        query: str,
        namespace: str = "default",
        vector_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Vector search with NVIDIA Llama 3.2 embeddings.

        PIPELINE (NVIDIA optimized):
        1. Vector top-k (semantic matching using llama-3.2-nv-embedqa-1b-v2)
        2. Return ALL candidates for reranking

        Benchmarks (MLQA dataset): NVIDIA embeddings + reranker achieves 86.83% recall@5
        vs 13.01% for BM25. Pure vector + reranker is the optimal approach.

        Args:
            query: Search query
            namespace: Pinecone namespace
            vector_k: Number of results from vector search

        Returns:
            Vector search results for reranking
        """
        logger.info(f"Performing vector search for query: '{query}' (Vector k={vector_k})")

        # Infer semantic tags for boosting
        query_tags = infer_query_tags(query)
        if query_tags:
            logger.info(f"Detected semantic tags for '{query}': {sorted(query_tags)}")

        # Vector search with semantic tag augmentations
        vector_results = await self.vector_search(query, namespace, vector_k, query_tags=query_tags)

        logger.info(f"Vector search found {len(vector_results)} results")

        return vector_results

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents using NVIDIA hosted reranker.

        Args:
            query: Search query
            documents: List of documents to re-rank
            top_k: Number of top results to return (None = return all)

        Returns:
            Re-ranked documents with updated scores
        """
        if not documents:
            return []

        # If reranker is not configured, fall back to hybrid score sorting
        if not self.reranker:
            logger.warning("NVIDIA reranker not configured, falling back to hybrid score sorting")
            sorted_docs = sorted(documents, key=lambda x: x.get("score", 0.0), reverse=True)
            if top_k:
                sorted_docs = sorted_docs[:top_k]
            return sorted_docs

        # Prepare documents for reranking - convert to LangChain Document objects
        langchain_docs = [
            Document(page_content=doc.get("metadata", {}).get("text", ""))
            for doc in documents
        ]

        # Use NVIDIA reranker
        try:
            reranked_results = self.reranker.compress_documents(
                query=query,
                documents=langchain_docs
            )

            # Map reranked results back to original documents
            # The reranker returns documents in relevance order with metadata
            reranked_docs = []
            for idx, reranked_doc in enumerate(reranked_results):
                # Find matching original document by content
                matching_doc = None
                for orig_doc in documents:
                    if orig_doc.get("metadata", {}).get("text", "") == reranked_doc.page_content:
                        matching_doc = orig_doc.copy()
                        break

                if matching_doc:
                    # Extract rerank score from Document metadata
                    # NVIDIA reranker puts relevance_score in metadata dict
                    if hasattr(reranked_doc, 'metadata') and isinstance(reranked_doc.metadata, dict):
                        rerank_score = reranked_doc.metadata.get('relevance_score', 1.0 - (idx * 0.01))
                    else:
                        # Fallback: use position-based scoring (higher position = higher score)
                        rerank_score = 1.0 - (idx * 0.01)

                    matching_doc["rerank_score"] = rerank_score
                    matching_doc["original_score"] = matching_doc.get("score", 0.0)
                    reranked_docs.append(matching_doc)

            if top_k:
                reranked_docs = reranked_docs[:top_k]

            logger.info(f"NVIDIA reranker processed {len(documents)} documents, returning top {len(reranked_docs)}")
            return reranked_docs

        except Exception as e:
            logger.error(f"NVIDIA reranker failed: {e}. Falling back to hybrid score sorting.")
            sorted_docs = sorted(documents, key=lambda x: x.get("score", 0.0), reverse=True)
            if top_k:
                sorted_docs = sorted_docs[:top_k]
            return sorted_docs

    async def hybrid_search_with_rerank(
        self,
        query: str,
        namespace: str = "default",
        top_k: int = 3,
        vector_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        NVIDIA OPTIMIZED RETRIEVAL PIPELINE (llama-3.2-nv-embedqa-1b-v2 + llama-3.2-nv-rerankqa-1b-v2):

        Step 1: Vector search using llama-3.2-nv-embedqa-1b-v2 (top 60 candidates)
        Step 2: Re-rank ALL candidates using nvidia/llama-3.2-nv-rerankqa-1b-v2
        Step 3: Return top 3-5 for context

        Benchmarks (MLQA dataset):
        - This pipeline: 86.83% recall@5
        - Embeddings only: 79.86% recall@5
        - BM25 only: 13.01% recall@5

        Args:
            query: Search query
            namespace: Pinecone namespace
            top_k: Final number of results to return (3-5 recommended)
            vector_k: Number of vector candidates (60 recommended)

        Returns:
            Re-ranked top-k results
        """
        # Step 1: Vector search with NVIDIA embeddings
        vector_results = await self.hybrid_search(query, namespace, vector_k)

        if not vector_results:
            logger.warning("No results from vector search")
            return []

        logger.info(f"Vector search returned {len(vector_results)} candidates for re-ranking")

        # Step 2: Re-rank ALL candidates with NVIDIA reranker
        reranked_results = await self.rerank(query, vector_results, top_k=None)

        # Step 3: Return top-k
        final_results = reranked_results[:top_k]

        logger.info(f"Returning top {len(final_results)} after re-ranking")

        return final_results


# Global instance
hybrid_search_engine = HybridSearchEngine()


def hybrid_search_sync(
    query: str,
    namespace: str = "default",
    top_k: int = 3,
    vector_k: int = 60
) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for hybrid_search_with_rerank.

    This function handles async/sync context switching to work correctly
    in both synchronous (LangChain tools) and asynchronous (FastAPI) contexts.

    Args:
        query: Search query
        namespace: Pinecone namespace
        top_k: Final number of results to return after re-ranking
        vector_k: Number of vector candidates

    Returns:
        Re-ranked top-k results
    """
    import concurrent.futures

    def run_async():
        """Run in a new event loop"""
        import asyncio
        return asyncio.run(
            hybrid_search_engine.hybrid_search_with_rerank(
                query=query,
                namespace=namespace,
                top_k=top_k,
                vector_k=vector_k
            )
        )

    # Run in a thread pool to avoid event loop conflicts
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_async)
        return future.result()
