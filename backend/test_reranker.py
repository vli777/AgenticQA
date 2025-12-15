# Test script to check NVIDIA reranker initialization

import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIARerank

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

print(f"NVIDIA_API_KEY set: {'Yes' if NVIDIA_API_KEY else 'No'}")
print(f"NVIDIA_API_KEY (first 10 chars): {NVIDIA_API_KEY[:10] if NVIDIA_API_KEY else 'N/A'}")

# First, list available models
print("\n=== Available NVIDIA Reranker Models ===")
try:
    available = NVIDIARerank.get_available_models(api_key=NVIDIA_API_KEY)
    for model in available:
        print(f"  - {model.id}")
except Exception as e:
    print(f"Could not fetch available models: {e}")

if not NVIDIA_API_KEY:
    print("\n❌ ERROR: NVIDIA_API_KEY is not set!")
    print("Set it in your .env file or environment")
    exit(1)

# Try different model names
model_names_to_try = [
    "llama-3.2-nv-rerankqa-1b-v2",
    "nvidia/llama-3.2-nv-rerankqa-1b-v2",
    "nv-rerankqa-mistral-4b-v3",
    "nvidia/nv-rerankqa-mistral-4b-v3",
    "nv-rerank-qa-mistral-4b:1",
]

print("\nTrying different reranker model names:\n")

for model_name in model_names_to_try:
    try:
        print(f"Trying: {model_name}...", end=" ")
        reranker = NVIDIARerank(
            model=model_name,
            api_key=NVIDIA_API_KEY
        )
        print("✅ SUCCESS!")
        print(f"  Model initialized: {model_name}")

        # Try a test query
        from langchain_core.documents import Document
        test_docs = [
            Document(page_content="Python is a programming language"),
            Document(page_content="JavaScript is used for web development")
        ]

        result = reranker.compress_documents(
            query="programming languages",
            documents=test_docs
        )
        print(f"  Test rerank succeeded! Returned {len(result)} documents")
        break

    except Exception as e:
        print("❌ FAILED")
        print(f"  Error: {str(e)[:100]}")

print("\nDone!")
