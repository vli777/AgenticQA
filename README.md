 # AgenticQA

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
  - Hybrid search (BM25 + vector) with cross-encoder re-ranking, Pinecone storage, and sentence-level chunking.
  - Per-topic conversational memory that automatically rewrites follow-up questions before retrieval.
  - Strict answer synthesis: answers cite their sources, fall back to “The documents do not clearly specify this.” when
  evidence is thin, and include a verification verdict in the reasoning trail.
  - React chat frontend with agentic and plain RAG modes, plus namespace management and easy clearing of stored vectors.

  ---

  ## Architecture in Brief

  1. The frontend uploads files to the FastAPI backend. Documents are cleaned, chunked (2,000 characters with 20%
  overlap), embedded, and upserted to Pinecone.
  2. Each chat turn carries a stable `conversation_id`. The backend memory manager assigns messages to topics,
  summarizes older turns, and supplies the relevant topic context.
  3. Before retrieval, the agent rewrites the user’s latest question into a standalone query using the current topic
  history.
  4. Hybrid search returns candidate snippets. The agent drafts an answer using only those snippets, verifies it, and
  cites the provenance tags.
  5. The answer, reasoning steps, verification verdict, and sources are returned to the client.

  ---

  ## Requirements

  - Python 3.10+
  - Node.js 18+ (for the frontend)
  - OpenAI API key (for embeddings or LLMs)
  - NVIDIA API key (for Llama/Maverick inference)
  - Pinecone API key and index
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

  OPENAI_API_KEY=...
  NVIDIA_API_KEY=...
  PINECONE_API_KEY=...
  PINECONE_INDEX_NAME=agenticqa
  EMBEDDING_MODEL=text-embedding-3-small  # or multilingual-e5-large, llama-text-embed-v2

  ### Frontend

  cd ../frontend
  npm install
  npm run dev

  Create a .env with VITE_API_BASE_URL=http://localhost:8000 (or your deployed backend) and visit the dev server.