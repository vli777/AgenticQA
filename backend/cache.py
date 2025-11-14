# backend/cache.py

import hashlib
import json
from functools import lru_cache
from typing import List, Dict, Any, Optional, Callable
import asyncio
from collections import OrderedDict
import time

from logger import logger


class LRUCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[int] = 3600):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of items to cache
            ttl_seconds: Time-to-live in seconds (None = no expiration)
        """
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    def _is_expired(self, timestamp: float) -> bool:
        """Check if a cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return (time.time() - timestamp) > self.ttl_seconds

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if not self._is_expired(timestamp):
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    self.hits += 1
                    logger.debug(f"Cache hit for key: {key[:50]}...")
                    return value
                else:
                    # Remove expired entry
                    del self.cache[key]

            self.misses += 1
            logger.debug(f"Cache miss for key: {key[:50]}...")
            return None

    async def set(self, key: str, value: Any):
        """Set value in cache."""
        async with self.lock:
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self.cache.popitem(last=False)

            self.cache[key] = (value, time.time())
            self.cache.move_to_end(key)

    async def clear(self):
        """Clear all cache entries."""
        async with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "ttl_seconds": self.ttl_seconds
        }


class EmbeddingCache:
    """Cache for text embeddings with batching support."""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 7200):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache
            ttl_seconds: Time-to-live for cached embeddings (default: 2 hours)
        """
        self.cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)
        self.pending_batch: List[tuple] = []
        self.batch_size = 32
        self.batch_timeout = 0.1  # 100ms
        self.last_batch_time = time.time()

    def _make_key(self, text: str, model: str) -> str:
        """Create cache key from text and model."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"{model}:{text_hash}:{len(text)}"

    async def get_embedding(
        self,
        text: str,
        model: str,
        embed_fn: Callable[[str, str], List[float]]
    ) -> List[float]:
        """
        Get embedding with caching.

        Args:
            text: Text to embed
            model: Model name
            embed_fn: Function to compute embedding if not cached

        Returns:
            Embedding vector
        """
        key = self._make_key(text, model)

        # Try cache first
        cached = await self.cache.get(key)
        if cached is not None:
            return cached

        # Compute embedding
        embedding = embed_fn(text, model)

        # Cache result
        await self.cache.set(key, embedding)

        return embedding

    async def get_embeddings_batch(
        self,
        texts: List[str],
        model: str,
        embed_fn: Callable[[List[str], str], List[List[float]]]
    ) -> List[List[float]]:
        """
        Get embeddings for multiple texts with caching and batching.

        Args:
            texts: List of texts to embed
            model: Model name
            embed_fn: Function to compute embeddings in batch

        Returns:
            List of embedding vectors
        """
        results = []
        uncached_indices = []
        uncached_texts = []

        # Check cache for each text
        for i, text in enumerate(texts):
            key = self._make_key(text, model)
            cached = await self.cache.get(key)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Compute uncached embeddings in batch
        if uncached_texts:
            logger.info(f"Computing {len(uncached_texts)} uncached embeddings in batch")
            embeddings = embed_fn(uncached_texts, model)

            # Cache and insert results
            for idx, text, embedding in zip(uncached_indices, uncached_texts, embeddings):
                key = self._make_key(text, model)
                await self.cache.set(key, embedding)
                results[idx] = embedding

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


class SearchCache:
    """Cache for search results."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 1800):
        """
        Initialize search cache.

        Args:
            max_size: Maximum number of search results to cache
            ttl_seconds: Time-to-live for cached results (default: 30 minutes)
        """
        self.cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)

    def _make_key(self, query: str, namespace: str, top_k: int, **kwargs) -> str:
        """Create cache key from search parameters."""
        params = {
            "query": query,
            "namespace": namespace,
            "top_k": top_k,
            **kwargs
        }
        # Sort kwargs for consistent key generation
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(params_str.encode()).hexdigest()[:32]

    async def get(
        self,
        query: str,
        namespace: str,
        top_k: int,
        **kwargs
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results."""
        key = self._make_key(query, namespace, top_k, **kwargs)
        return await self.cache.get(key)

    async def set(
        self,
        query: str,
        namespace: str,
        top_k: int,
        results: List[Dict[str, Any]],
        **kwargs
    ):
        """Cache search results."""
        key = self._make_key(query, namespace, top_k, **kwargs)
        await self.cache.set(key, results)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


class LLMResponseCache:
    """Cache for LLM responses."""

    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        """
        Initialize LLM response cache.

        Args:
            max_size: Maximum number of responses to cache
            ttl_seconds: Time-to-live for cached responses (default: 1 hour)
        """
        self.cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)

    def _make_key(self, prompt: str, model: str, temperature: float) -> str:
        """Create cache key from LLM parameters."""
        # Only cache for temperature = 0 (deterministic responses)
        if temperature > 0:
            return None

        params_str = f"{model}:{prompt}"
        return hashlib.sha256(params_str.encode()).hexdigest()[:32]

    async def get(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0
    ) -> Optional[str]:
        """Get cached LLM response."""
        key = self._make_key(prompt, model, temperature)
        if key is None:
            return None
        return await self.cache.get(key)

    async def set(
        self,
        prompt: str,
        model: str,
        response: str,
        temperature: float = 0.0
    ):
        """Cache LLM response."""
        key = self._make_key(prompt, model, temperature)
        if key is not None:
            await self.cache.set(key, response)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


# Global cache instances
embedding_cache = EmbeddingCache(max_size=10000, ttl_seconds=7200)
search_cache = SearchCache(max_size=1000, ttl_seconds=1800)
llm_cache = LLMResponseCache(max_size=500, ttl_seconds=3600)


def get_all_cache_stats() -> Dict[str, Any]:
    """Get statistics for all caches."""
    return {
        "embedding_cache": embedding_cache.get_stats(),
        "search_cache": search_cache.get_stats(),
        "llm_cache": llm_cache.get_stats()
    }


async def clear_all_caches():
    """Clear all caches."""
    await embedding_cache.cache.clear()
    await search_cache.cache.clear()
    await llm_cache.cache.clear()
    logger.info("All caches cleared")
