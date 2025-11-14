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

    def _boost_score(self, base_score: float, text: str, query: str) -> float:
        """
        Apply section-aware boosting to BM25 scores.

        Boosts scores for:
        - Important sections (Skills, Languages, Education, etc.)
        - Exact term matches
        - Short, focused chunks

        Args:
            base_score: Original BM25 score
            text: Document text
            query: Search query

        Returns:
            Boosted score
        """
        boost = 1.0
        text_lower = text.lower()
        query_lower = query.lower()

        # Boost for important section headers
        important_sections = [
            'skills', 'languages', 'technologies', 'experience',
            'education', 'tools', 'frameworks', 'expertise'
        ]

        for section in important_sections:
            if section in text_lower[:100]:  # Check first 100 chars for section header
                boost *= 2.0
                logger.debug(f"Section boost applied: {section}")
                break

        # Boost for exact term matches (case-insensitive)
        query_terms = query_lower.split()
        for term in query_terms:
            if len(term) > 2:  # Skip very short terms
                # Count exact word matches (not substrings)
                pattern = r'\b' + re.escape(term) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    boost *= (1.0 + 0.5 * matches)  # Boost by 50% per match

        # Boost shorter, focused chunks (more signal, less noise)
        if len(text) < 500:  # Short chunks are usually more focused
            boost *= 1.5

        return base_score * boost

    async def bm25_search(
        self,
        query: str,
        namespace: str = "default",
        top_k: int = 20,
        apply_boosting: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 search with optional section-aware boosting.

        Args:
            query: Search query
            namespace: Pinecone namespace
            top_k: Number of results to return
            apply_boosting: Apply section-aware score boosting

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

        # Apply boosting if enabled
        if apply_boosting:
            boosted_scores = []
            for idx, base_score in enumerate(scores):
                if base_score > 0:
                    text = documents[idx].get("text", "")
                    boosted = self._boost_score(base_score, text, query)
                    boosted_scores.append((idx, boosted))
                else:
                    boosted_scores.append((idx, base_score))

            # Sort by boosted scores
            boosted_scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in boosted_scores[:top_k]]
            top_scores = [score for _, score in boosted_scores[:top_k]]
        else:
            # Original behavior
            top_indices = np.argsort(scores)[::-1][:top_k]
            top_scores = [scores[idx] for idx in top_indices]

        results = []
        for idx, score in zip(top_indices, top_scores):
            if score > 0:  # Only include results with positive scores
                doc = documents[idx]
                results.append({
                    "id": doc["id"],
                    "score": float(score),
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
        vector_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combine BM25 and vector search results with deduplication.
        Does NOT apply score fusion - that's only for fallback.
        Re-ranker will determine final order.

        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search

        Returns:
            Merged and deduplicated results with both scores attached
        """
        # Track BM25 and vector scores separately
        bm25_scores = {r["id"]: r["score"] for r in bm25_results}
        vector_scores = {r["id"]: r["score"] for r in vector_results}

        # Get all unique document IDs
        all_ids = set(bm25_scores.keys()) | set(vector_scores.keys())

        # Get full document info (prefer vector result if in both)
        id_to_doc = {}
        for r in bm25_results:
            if r["id"] not in id_to_doc:
                id_to_doc[r["id"]] = r
        for r in vector_results:  # Vector overwrites if duplicate
            id_to_doc[r["id"]] = r

        # Create merged results with both scores
        results = []
        for doc_id in all_ids:
            doc = id_to_doc[doc_id]
            results.append({
                "id": doc_id,
                "metadata": doc["metadata"],
                "bm25_score": bm25_scores.get(doc_id, 0.0),
                "vector_score": vector_scores.get(doc_id, 0.0),
                "in_bm25": doc_id in bm25_scores,
                "in_vector": doc_id in vector_scores
            })

        logger.info(f"Merged {len(bm25_results)} BM25 + {len(vector_results)} vector = {len(results)} unique documents")

        return results

    async def hybrid_search(
        self,
        query: str,
        namespace: str = "default",
        bm25_k: int = 30,
        vector_k: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and vector search.

        CORRECT PIPELINE:
        1. BM25 top-k (default 30)
        2. Vector top-k (default 30)
        3. Merge + dedupe (up to 60 unique)
        4. Return ALL merged (re-ranker will sort)

        Args:
            query: Search query
            namespace: Pinecone namespace
            bm25_k: Number of results from BM25 search
            vector_k: Number of results from vector search

        Returns:
            Merged results (NOT sorted, re-ranker does that)
        """
        logger.info(f"Performing hybrid search for query: '{query}' (BM25 k={bm25_k}, Vector k={vector_k})")

        # Step 1: BM25 top-k (with section boosting enabled by default)
        bm25_results = await self.bm25_search(query, namespace, bm25_k, apply_boosting=True)

        # Step 2: Vector top-k
        vector_results = await self.vector_search(query, namespace, vector_k)

        logger.info(f"BM25 found {len(bm25_results)} results, Vector found {len(vector_results)} results")

        # Step 3: Merge + dedupe
        combined_results = self._combine_results(bm25_results, vector_results)

        # Return ALL merged results (re-ranker will handle sorting)
        return combined_results

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
        bm25_k: int = 30,
        vector_k: int = 30
    ) -> List[Dict[str, Any]]:
        """
        CORRECT HYBRID RAG PIPELINE (Best Practice):

        Step 1: BM25 top-k (default 30) - lexical candidates
        Step 2: Vector top-k (default 30) - semantic candidates
        Step 3: Merge + dedupe (up to 60 unique documents)
        Step 4: Cross-encoder re-rank ALL merged candidates
        Step 5: Return top 3-5 for context

        This prevents lexical false positives from dominating
        and ensures semantic relevance is properly weighted.

        Args:
            query: Search query
            namespace: Pinecone namespace
            top_k: Final number of results to return (3-5 recommended)
            bm25_k: Number of BM25 candidates (30 recommended)
            vector_k: Number of vector candidates (30 recommended)

        Returns:
            Re-ranked top-k results
        """
        # Step 1-3: Hybrid search (BM25 + Vector + Merge)
        hybrid_results = await self.hybrid_search(query, namespace, bm25_k, vector_k)

        if not hybrid_results:
            logger.warning("No results from hybrid search")
            return []

        logger.info(f"Hybrid search returned {len(hybrid_results)} merged candidates for re-ranking")

        # Step 4: Re-rank ALL merged candidates with cross-encoder
        reranked_results = await self.rerank(query, hybrid_results, top_k=None)

        # Step 5: Return top-k
        final_results = reranked_results[:top_k]

        logger.info(f"Returning top {len(final_results)} after re-ranking")

        return final_results


# Global instance
hybrid_search_engine = HybridSearchEngine()


def hybrid_search_sync(
    query: str,
    namespace: str = "default",
    top_k: int = 3,
    bm25_k: int = 30,
    vector_k: int = 30
) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for hybrid_search_with_rerank.

    This function handles async/sync context switching to work correctly
    in both synchronous (LangChain tools) and asynchronous (FastAPI) contexts.

    Args:
        query: Search query
        namespace: Pinecone namespace
        top_k: Final number of results to return after re-ranking
        bm25_k: Number of BM25 candidates
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
                bm25_k=bm25_k,
                vector_k=vector_k
            )
        )

    # Run in a thread pool to avoid event loop conflicts
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_async)
        return future.result()
