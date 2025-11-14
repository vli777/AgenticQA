# backend/hybrid_search.py

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import re

from logger import logger
from pinecone_client import index
from config import EMBEDDING_MODEL, CROSS_ENCODER_MODEL
from utils import get_embedding


class HybridSearchEngine:
    """
    Hybrid search engine combining BM25 (sparse) and vector (dense) retrieval
    with cross-encoder re-ranking.
    """

    def __init__(self, cross_encoder_model: str = None):
        """
        Initialize the hybrid search engine.

        Args:
            cross_encoder_model: HuggingFace model name for cross-encoder re-ranking
                                 (defaults to value from config)
        """
        model = cross_encoder_model or CROSS_ENCODER_MODEL
        self.cross_encoder = CrossEncoder(model)
        self.bm25_cache: Dict[str, Tuple[BM25Okapi, List[Dict[str, Any]]]] = {}
        logger.info(f"Initialized HybridSearchEngine with cross-encoder: {model}")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Lowercase and split on non-alphanumeric characters
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _build_bm25_index(self, documents: List[Dict[str, Any]]) -> BM25Okapi:
        """
        Build BM25 index from documents.

        Args:
            documents: List of documents with 'text' field

        Returns:
            BM25Okapi index
        """
        corpus = [doc.get("text", "") for doc in documents]
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25

    def _fetch_all_documents(self, namespace: str = "default", limit: int = 10000) -> List[Dict[str, Any]]:
        """
        Fetch all documents from Pinecone namespace for BM25 indexing.

        Args:
            namespace: Pinecone namespace
            limit: Maximum number of documents to fetch

        Returns:
            List of documents with metadata
        """
        try:
            # Use a dummy query to fetch documents
            # We'll query with a zero vector or use list/scan if available
            logger.info(f"Fetching documents from namespace '{namespace}' for BM25 indexing")

            # Create a dummy vector for querying
            # Get dimension from config or use default
            if EMBEDDING_MODEL == "multilingual-e5-large":
                dimension = 1024
            else:  # text-embedding-3-small or default
                dimension = 1536

            dummy_vector = [0.0] * dimension

            # Fetch top results with dummy vector
            response = index.query(
                vector=dummy_vector,
                top_k=min(limit, 10000),  # Pinecone limit
                include_metadata=True,
                namespace=namespace
            )

            documents = []
            for match in response.get("matches", []):
                metadata = match.get("metadata", {})
                if "text" in metadata:
                    documents.append({
                        "id": match.get("id"),
                        "text": metadata.get("text"),
                        "metadata": metadata
                    })

            logger.info(f"Fetched {len(documents)} documents from Pinecone")
            return documents

        except Exception as e:
            logger.error(f"Error fetching documents for BM25: {str(e)}")
            return []

    def _get_or_build_bm25(self, namespace: str = "default") -> Tuple[BM25Okapi, List[Dict[str, Any]]]:
        """
        Get cached BM25 index or build a new one.

        Args:
            namespace: Pinecone namespace

        Returns:
            Tuple of (BM25 index, documents list)
        """
        if namespace not in self.bm25_cache:
            logger.info(f"Building BM25 index for namespace '{namespace}'")
            documents = self._fetch_all_documents(namespace)
            if documents:
                bm25 = self._build_bm25_index(documents)
                self.bm25_cache[namespace] = (bm25, documents)
            else:
                # Return empty index if no documents
                self.bm25_cache[namespace] = (None, [])

        return self.bm25_cache[namespace]

    def clear_cache(self, namespace: Optional[str] = None):
        """
        Clear BM25 cache for a namespace or all namespaces.

        Args:
            namespace: Specific namespace to clear, or None to clear all
        """
        if namespace:
            self.bm25_cache.pop(namespace, None)
            logger.info(f"Cleared BM25 cache for namespace '{namespace}'")
        else:
            self.bm25_cache.clear()
            logger.info("Cleared all BM25 caches")

    async def bm25_search(
        self,
        query: str,
        namespace: str = "default",
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 search.

        Args:
            query: Search query
            namespace: Pinecone namespace
            top_k: Number of results to return

        Returns:
            List of matches with scores
        """
        bm25, documents = self._get_or_build_bm25(namespace)

        if not bm25 or not documents:
            logger.warning(f"No BM25 index available for namespace '{namespace}'")
            return []

        # Tokenize query and get scores
        tokenized_query = self._tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                doc = documents[idx]
                results.append({
                    "id": doc["id"],
                    "score": float(scores[idx]),
                    "metadata": doc["metadata"]
                })

        return results

    async def vector_search(
        self,
        query: str,
        namespace: str = "default",
        top_k: int = 20
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
        if EMBEDDING_MODEL in {"multilingual-e5-large", "text-embedding-3-small"}:
            q_embed = get_embedding(text=query, model=EMBEDDING_MODEL)
            response = index.query(
                vector=q_embed,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace
            )
        else:
            query_body = {"top_k": top_k, "inputs": {"text": query}}
            if EMBEDDING_MODEL == "llama-text-embed-v2":
                query_body["model"] = "llama-text-embed-v2"

            response = index.search(
                namespace=namespace,
                query=query_body
            )

        matches = response.get("matches", [])
        results = []
        for match in matches:
            results.append({
                "id": match.get("id"),
                "score": match.get("score", 0.0),
                "metadata": match.get("metadata", {})
            })

        return results

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range using min-max normalization."""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score - min_score == 0:
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _combine_results(
        self,
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Combine BM25 and vector search results using weighted scoring.

        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search
            alpha: Weight for BM25 scores (1-alpha for vector scores)

        Returns:
            Combined and sorted results
        """
        # Create score dictionaries
        bm25_scores = {r["id"]: r["score"] for r in bm25_results}
        vector_scores = {r["id"]: r["score"] for r in vector_results}

        # Normalize scores separately
        bm25_norm = self._normalize_scores(list(bm25_scores.values())) if bm25_scores else []
        vector_norm = self._normalize_scores(list(vector_scores.values())) if vector_scores else []

        bm25_scores_norm = dict(zip(bm25_scores.keys(), bm25_norm)) if bm25_norm else {}
        vector_scores_norm = dict(zip(vector_scores.keys(), vector_norm)) if vector_norm else {}

        # Combine scores
        all_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
        combined = {}

        for doc_id in all_ids:
            bm25_score = bm25_scores_norm.get(doc_id, 0.0)
            vector_score = vector_scores_norm.get(doc_id, 0.0)
            combined[doc_id] = alpha * bm25_score + (1 - alpha) * vector_score

        # Get full document info
        id_to_doc = {}
        for r in bm25_results + vector_results:
            if r["id"] not in id_to_doc:
                id_to_doc[r["id"]] = r

        # Create sorted results
        results = []
        for doc_id in sorted(combined.keys(), key=lambda x: combined[x], reverse=True):
            doc = id_to_doc[doc_id]
            results.append({
                "id": doc_id,
                "score": combined[doc_id],
                "metadata": doc["metadata"],
                "bm25_score": bm25_scores.get(doc_id, 0.0),
                "vector_score": vector_scores.get(doc_id, 0.0)
            })

        return results

    async def hybrid_search(
        self,
        query: str,
        namespace: str = "default",
        top_k: int = 20,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and vector search.

        Args:
            query: Search query
            namespace: Pinecone namespace
            top_k: Number of results to return from each method before combining
            alpha: Weight for BM25 scores (0.0 = vector only, 1.0 = BM25 only, 0.5 = equal weight)

        Returns:
            Combined and sorted results
        """
        logger.info(f"Performing hybrid search for query: '{query}' (alpha={alpha})")

        # Run both searches in parallel conceptually (but in sequence for simplicity)
        bm25_results = await self.bm25_search(query, namespace, top_k)
        vector_results = await self.vector_search(query, namespace, top_k)

        logger.info(f"BM25 found {len(bm25_results)} results, Vector found {len(vector_results)} results")

        # Combine results
        combined_results = self._combine_results(bm25_results, vector_results, alpha)

        return combined_results[:top_k]

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents using cross-encoder.

        Args:
            query: Search query
            documents: List of documents to re-rank
            top_k: Number of top results to return (None = return all)

        Returns:
            Re-ranked documents with updated scores
        """
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            text = doc.get("metadata", {}).get("text", "")
            pairs.append([query, text])

        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)

        # Create re-ranked results
        reranked = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            doc_copy["original_score"] = doc.get("score", 0.0)
            reranked.append(doc_copy)

        # Sort by rerank score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        logger.info(f"Re-ranked {len(documents)} documents, returning top {len(reranked)}")

        return reranked

    async def hybrid_search_with_rerank(
        self,
        query: str,
        namespace: str = "default",
        top_k: int = 3,
        retrieval_k: int = 20,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Complete pipeline: hybrid search + re-ranking.

        Args:
            query: Search query
            namespace: Pinecone namespace
            top_k: Final number of results to return after re-ranking
            retrieval_k: Number of results to retrieve before re-ranking
            alpha: Weight for BM25 in hybrid search

        Returns:
            Re-ranked top-k results
        """
        # Step 1: Hybrid search
        hybrid_results = await self.hybrid_search(query, namespace, retrieval_k, alpha)

        # Step 2: Re-rank
        reranked_results = await self.rerank(query, hybrid_results, top_k)

        return reranked_results


# Global instance
hybrid_search_engine = HybridSearchEngine()


def hybrid_search_sync(
    query: str,
    namespace: str = "default",
    top_k: int = 3,
    retrieval_k: int = 20,
    alpha: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for hybrid_search_with_rerank.

    This function handles async/sync context switching to work correctly
    in both synchronous (LangChain tools) and asynchronous (FastAPI) contexts.

    Args:
        query: Search query
        namespace: Pinecone namespace
        top_k: Final number of results to return after re-ranking
        retrieval_k: Number of results to retrieve before re-ranking
        alpha: Weight for BM25 in hybrid search

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
                retrieval_k=retrieval_k,
                alpha=alpha
            )
        )

    # Run in a thread pool to avoid event loop conflicts
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_async)
        return future.result()
