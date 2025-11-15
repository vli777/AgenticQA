#!/usr/bin/env python3
"""Quick test to verify NVIDIA reranker initialization"""

import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIARerank

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
model_name = "nvidia/llama-3.2-nv-rerankqa-1b-v2"

print(f"Testing reranker model: {model_name}")
print(f"NVIDIA_API_KEY set: {'Yes' if NVIDIA_API_KEY else 'No'}")

try:
    reranker = NVIDIARerank(
        model=model_name,
        api_key=NVIDIA_API_KEY
    )
    print(f"SUCCESS! Reranker initialized with model: {model_name}")

    # Test with sample documents
    from langchain_core.documents import Document
    test_docs = [
        Document(page_content="Python is a high-level programming language"),
        Document(page_content="JavaScript is used for web development"),
        Document(page_content="Java is an object-oriented language")
    ]

    result = reranker.compress_documents(
        query="programming languages",
        documents=test_docs
    )
    print(f"Test rerank succeeded! Returned {len(result)} documents")
    for i, doc in enumerate(result):
        print(f"  {i+1}. {doc.page_content[:50]}...")

except Exception as e:
    print(f"FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
