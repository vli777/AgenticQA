# AgenticQA

*A modern, agent-powered document Q&A system. Upload your docs, ask questions, and get smart, context-rich answers—powered by an LLM agent that searches, reasons, and cites its sources.*

---

## 🚀 Overview

**AgenticQA** uses advanced agentic retrieval (beyond traditional RAG) to enable natural-language Q&A over your own files. Instead of just searching once, our agent plans, searches, and reasons in multiple steps—like a real researcher.

- **Upload**: PDF or text files  
- **Ask**: Any question in natural language  
- **Get**: Detailed, cited answers powered by OpenAI GPT and Pinecone, via LangChain agentic workflows

---

## ✨ Features

- Agentic retrieval (LLM “agent” can plan, retrieve, and reason step-by-step)
- Q&A over uploaded docs (PDF, TXT)
- Fast, secure embeddings + search (OpenAI + Pinecone)
- Modern chat interface (React)
- Source citations and agent reasoning trace
- Easy deployment (Vercel & Railway/Render)
- Runs on free-tier cloud infrastructure

---

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, LangChain, OpenAI API, Pinecone vector DB  
- **Frontend**: React, Vercel  
- **Cloud**: Railway/Render (backend), Vercel (frontend)

---

## 🏗️ Architecture
![image](https://github.com/user-attachments/assets/5ddb6149-a73a-4607-9f60-985b52d44e1c)

**Flow:**
1. **User uploads a doc** → backend splits and embeds → stored in Pinecone
2. **User asks a question** → agent plans retrieval steps, pulls context, generates answer, cites sources
3. **Answer & sources** returned to frontend chat UI

---

## 📦 Requirements

- Python 3.10+
- Node.js 18+ (for frontend)
- OpenAI API key ([get one here](https://platform.openai.com/signup))
- Pinecone API key ([free signup](https://www.pinecone.io/start/))
- Railway or Render account (backend deploy)
- Vercel account (frontend deploy)

---

## ⚡ Usage

### 1. Clone and set up backend

```
git clone https://github.com/yourusername/agenticdocqa.git
cd agenticdocqa/backend
pip install -r requirements.txt
# Set environment variables: OPENAI_API_KEY, PINECONE_API_KEY, etc.
uvicorn main:app --reload
```

### 2. Set up frontend
```
cd ../frontend
npm install
npm run dev
# Edit .env to point to your backend API
```

### 3. Deploy
Backend: Deploy to Railway or Render
Frontend: Deploy to Vercel
