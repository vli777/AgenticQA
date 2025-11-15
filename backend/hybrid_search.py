# backend/hybrid_search.py

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from rank_bm25 import BM25Okapi
import re

from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_core.documents import Document

from logger import logger
from pinecone_client import index
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

    Note: BM25 code is kept for backward compatibility but NOT used in the retrieval pipeline.
    The NVIDIA embedding + reranker stack vastly outperforms BM25.
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

        self.bm25_cache: Dict[str, Tuple[BM25Okapi, List[Dict[str, Any]]]] = {}

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
            if EMBEDDING_MODEL == "text-embedding-3-small":
                dimension = 1536
            else:  # NVIDIA embeddings default to 1024
                dimension = 1024

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

    def _fetch_tagged_documents(
        self,
        namespace: str,
        tags: Optional[Set[str]],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Fetch additional documents that share inferred semantic tags."""
        if not tags:
            return []

        # Match dimension logic from _fetch_all_documents
        if EMBEDDING_MODEL == "text-embedding-3-small":
            dimension = 1536
        else:  # NVIDIA/Llama embeddings default to 1024
            dimension = 1024

        try:
            response = index.query(
                vector=[0.0] * dimension,
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
        Clear cache for a namespace or all namespaces.
        (Legacy BM25 cache, kept for backward compatibility)

        Args:
            namespace: Specific namespace to clear, or None to clear all
        """
        if namespace:
            self.bm25_cache.pop(namespace, None)
            logger.info(f"Cleared search cache for namespace '{namespace}'")
        else:
            self.bm25_cache.clear()
            logger.info("Cleared all search caches")

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

        # If the user query implies semantic tags, fetch extra candidates that
        # share those tags (even if they don't rank highly by pure similarity).
        tagged_additions = self._fetch_tagged_documents(namespace, query_tags)
        if tagged_additions:
            existing_ids = {r["id"] for r in results}
            for doc in tagged_additions:
                if doc["id"] not in existing_ids:
                    results.append(doc)

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
        query_tags: Optional[Set[str]] = None
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
            metadata = doc.get("metadata", {})
            metadata_tags = {
                str(tag).lower()
                for tag in metadata.get("semantic_tags", []) if tag
            }
            tag_match = bool(query_tags and metadata_tags and (metadata_tags & query_tags))
            boosted_vector_score = vector_scores.get(doc_id, 0.0) + (SEMANTIC_TAG_BOOST if tag_match else 0.0)

            metadata["semantic_tag_match"] = tag_match
            results.append({
                "id": doc_id,
                "metadata": metadata,
                "bm25_score": bm25_scores.get(doc_id, 0.0),
                "vector_score": boosted_vector_score,
                "in_bm25": doc_id in bm25_scores,
                "in_vector": doc_id in vector_scores
            })

        logger.info(f"Merged {len(bm25_results)} BM25 + {len(vector_results)} vector = {len(results)} unique documents")

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
