#!/usr/bin/env python3
"""Debug script to test search and see what's indexed."""

import asyncio
from hybrid_search import hybrid_search_engine

async def debug_search():
    """Debug search to see what's being retrieved."""

    query = "Python"
    namespace = "default"

    print(f"\n{'='*60}")
    print(f"DEBUG: Searching for '{query}' in namespace '{namespace}'")
    print(f"{'='*60}\n")

    # Clear BM25 cache first
    print("1. Clearing BM25 cache...")
    hybrid_search_engine.clear_cache(namespace)
    print("   ✓ Cache cleared\n")

    # Try BM25 search
    print("2. BM25 Search (top 10):")
    print("-" * 60)
    bm25_results = await hybrid_search_engine.bm25_search(query, namespace, top_k=10)
    if bm25_results:
        for i, result in enumerate(bm25_results, 1):
            text = result.get("metadata", {}).get("text", "")[:200]
            score = result.get("score", 0)
            print(f"   {i}. Score: {score:.3f}")
            print(f"      Text: {text}...")
            print()
    else:
        print("   ⚠ No BM25 results found!\n")

    # Try vector search
    print("3. Vector Search (top 10):")
    print("-" * 60)
    vector_results = await hybrid_search_engine.vector_search(query, namespace, top_k=10)
    if vector_results:
        for i, result in enumerate(vector_results, 1):
            text = result.get("metadata", {}).get("text", "")[:200]
            score = result.get("score", 0)
            print(f"   {i}. Score: {score:.3f}")
            print(f"      Text: {text}...")
            print()
    else:
        print("   ⚠ No vector results found!\n")

    # Try hybrid search
    print("4. Hybrid Search (top 10):")
    print("-" * 60)
    hybrid_results = await hybrid_search_engine.hybrid_search(query, namespace, bm25_k=30, vector_k=30)
    if hybrid_results:
        print(f"   Found {len(hybrid_results)} merged results\n")
        for i, result in enumerate(hybrid_results[:10], 1):
            text = result.get("metadata", {}).get("text", "")[:200]
            bm25_score = result.get("bm25_score", 0)
            vector_score = result.get("vector_score", 0)
            print(f"   {i}. BM25: {bm25_score:.3f}, Vector: {vector_score:.3f}")
            print(f"      In BM25: {result.get('in_bm25')}, In Vector: {result.get('in_vector')}")
            print(f"      Text: {text}...")
            print()
    else:
        print("   ⚠ No hybrid results found!\n")

    # Try with re-ranking
    print("5. Full Pipeline (hybrid + rerank, top 5):")
    print("-" * 60)
    final_results = await hybrid_search_engine.hybrid_search_with_rerank(
        query, namespace, top_k=5, bm25_k=30, vector_k=30
    )
    if final_results:
        for i, result in enumerate(final_results, 1):
            text = result.get("metadata", {}).get("text", "")[:200]
            rerank_score = result.get("rerank_score", 0)
            bm25_score = result.get("bm25_score", 0)
            vector_score = result.get("vector_score", 0)
            print(f"   {i}. Rerank: {rerank_score:.3f}, BM25: {bm25_score:.3f}, Vector: {vector_score:.3f}")
            print(f"      Text: {text}...")
            print()
    else:
        print("   ⚠ No final results!\n")

if __name__ == "__main__":
    asyncio.run(debug_search())
