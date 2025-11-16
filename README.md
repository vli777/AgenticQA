 # AgenticQA
<img width="1245" height="1252" alt="Screenshot 2025-11-14 204431" src="https://github.com/user-attachments/assets/16487805-56f6-4c74-876d-1d73aa0ed36c" />

  AgenticQA is a document-centric Q&A system that feels like a careful research assistant. Upload PDF or text files, ask
  natural language questions, and receive answers that cite supporting passages. The backend runs a conversation-aware
  retrieval agent that plans searches, rewrites follow-up questions, and verifies every response against the evidence
  before replying.

  ---

  ## Overview

  Traditional RAG pipelines run a single search and hope the results are relevant. AgenticQA extends that idea with:

  - Multi-step planning: the agent rephrases the question, runs several searches, and merges the findings.
  - Topic-aware memory: the conversation is grouped into topics so follow-up questions automatically reference the right
  context without polluting the vector store.
  - Evidence-first answers: each response is drafted from the retrieved snippets, then verified to ensure it is actually
  supported. If the evidence is unclear, the agent says so.
  - Deterministic chunking: documents are split once inside the backend, guaranteeing consistent windows and overlap no
  matter how they were uploaded.

  The result is a more cautious and transparent assistant that refuses to guess when the documents do not clearly answer
  the question.

  ---

  ## Key Features

  - Upload PDF or TXT documents directly from the chat interface.
  - NVIDIA-optimized retrieval (nvidia/llama-3.2-nv-embedqa-1b-v2 + nvidia/llama-3.2-nv-rerankqa-1b-v2) with 86.83% recall, Pinecone storage, and sentence-level chunking.
  - NVIDIA hosted embeddings and reranking for minimal Docker image size (~500MB vs 8GB+ with local models).
  - Per-topic conversational memory that automatically rewrites follow-up questions before retrieval.
  - Strict answer synthesis: answers cite their sources, fall back to "The documents do not clearly specify this." when
  evidence is thin, and include a verification verdict in the reasoning trail.
  - React chat frontend with agentic and plain RAG modes, plus namespace management and easy clearing of stored vectors.

  ---

  ## Architecture in Brief

  1. The frontend uploads files to the FastAPI backend. Documents are cleaned, chunked (1,000 characters with 20%
  overlap, optimized for NVIDIA's 512-token limit), embedded, and upserted to Pinecone.
  2. Each chat turn carries a stable `conversation_id`. The backend memory manager assigns messages to topics,
  summarizes older turns, and supplies the relevant topic context.
  3. Before retrieval, the agent rewrites the user's latest question into a standalone query using the current topic
  history.
  4. Vector search (nvidia/llama-3.2-nv-embedqa-1b-v2) returns 60 candidate snippets, which are re-ranked by NVIDIA's reranker
  (llama-3.2-nv-rerankqa-1b-v2) for 86.83% recall. The agent drafts an answer using only those snippets, verifies it,
  and cites the provenance tags.
  5. The answer, reasoning steps, verification verdict, and sources are returned to the client.

  ---

  ## Requirements

  - Python 3.10+
  - Node.js 18+ (for the frontend)
  - NVIDIA API key (REQUIRED - for hosted embeddings, reranking, and LLM inference. Free tier available!)
  - Pinecone API key and index
  - OpenAI API key (OPTIONAL - only if using text-embedding-3-small instead of NVIDIA embeddings)
  - Railway or Render account (backend deployment) and Vercel (frontend) if you plan to host it

  ---

  ## Getting Started

  ### Backend

  ```bash
  git clone https://github.com/vli777/agenticqa.git
  cd agenticqa/backend
  python -m venv venv
  source venv/bin/activate  # or venv\Scripts\activate on Windows
  pip install -r requirements.txt
  uvicorn main:app --reload

  Environment variables (set in .env or via your hosting platform):

  NVIDIA_API_KEY=...  # REQUIRED - Get free tier at build.nvidia.com
  PINECONE_API_KEY=...
  PINECONE_INDEX_NAME=agenticqa

  # Embedding Model (auto-defaults to nvidia/llama-3.2-nv-embedqa-1b-v2 if not specified)
  # Options:
  #   - nvidia/llama-3.2-nv-embedqa-1b-v2 (1024-dim, DEFAULT, Q&A optimized, recommended)
  #   - nvidia/nv-embedqa-e5-v5 (1024-dim, legacy, works with existing e5-large indexes)
  #   - nvidia/nv-embed-v1 (4096-dim, requires new Pinecone index with 4096 dimensions)
  #   - text-embedding-3-small (1536-dim, requires OpenAI key and 1536-dim index)
  # EMBEDDING_MODEL=nvidia/llama-3.2-nv-embedqa-1b-v2

  OPENAI_API_KEY=...  # Optional - only needed if using text-embedding-3-small

  ### Frontend

  cd ../frontend
  npm install
  npm run dev

  Create a .env with VITE_API_BASE_URL=http://localhost:8000 (or your deployed backend) and visit the dev server.
