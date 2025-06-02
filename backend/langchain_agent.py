# backend/langchain_agent.py

import os
from typing import List, Any, Dict

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone as LC_Pinecone
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from config import PINECONE_INDEX_NAME, EMBEDDING_MODEL, OPENAI_API_KEY

def _get_vectorstore(namespace: str = "default") -> LC_Pinecone:
    """
    Returns a LangChain-Pinecone wrapper for an existing index, exactly as shown in:
    https://python.langchain.com/docs/integrations/vectorstores/pinecone/
    We pick an Embeddings object based on EMBEDDING_MODEL; if unset or 'llama-text-embed-v2',
    we default to OpenAIEmbeddings (because Pinecone will embed server-side if that index was created with LLaMA).
    """
    # 2) Choose embedding based on EMBEDDING_MODEL
    if EMBEDDING_MODEL == "multilingual-e5-large":
        # Use HuggingFace E5 for local embeddings
        embedder = HuggingFaceEmbeddings(model_name="intfloat/e5-large")
    elif EMBEDDING_MODEL == "text-embedding-3-small":
        # Use OpenAI’s text-embedding-3-small via langchain_openai
        embedder = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        )
    else:
        # EMBEDDING_MODEL is None or "llama-text-embed-v2"
        # We still need to pass an Embeddings object, so fall back to OpenAIEmbeddings.
        embedder = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    # 3) Wrap the existing Pinecone index (created elsewhere) into a PineconeVectorStore
    vectorstore = LC_Pinecone.from_existing_index(
        embedding=embedder,
        index_name=PINECONE_INDEX_NAME,
        namespace=namespace,
        text_key="text",
    )
    return vectorstore


def _pinecone_search_tool(namespace: str = "default") -> Tool:
    """
    Returns a LangChain Tool that performs Pinecone similarity_search under the hood,
    then formats the top-5 hits as “[source::chunk_id] snippet…”, following the Pinecone docs.
    """
    vectorstore = _get_vectorstore(namespace)

    def search_fn(query: str) -> str:
        # Retrieve up to 5 matching chunks
        docs = vectorstore.similarity_search(query, k=5)
        lines: List[str] = []
        for doc in docs:
            meta = doc.metadata or {}
            src = meta.get("source", "unknown")
            chunk_id = meta.get("chunk_id", meta.get("id", ""))
            snippet = doc.page_content or ""
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
            lines.append(f"[{src}::{chunk_id}] {snippet}")
        return "\n".join(lines)

    return Tool(
        name="semantic_search",
        func=search_fn,
        description=(
            "Use this tool to perform a semantic search over uploaded documents. "
            "Input: a query string. Output: up to five matching chunks "
            "formatted as “[source::chunk_id] snippet…”."
        )
    )


def get_agent(namespace: str = "default"):
    """
    Returns a LangChain Agent that can call our Pinecone “semantic_search” tool.
    We use ChatOpenAI (gpt-3.5-turbo) as the LLM and ZERO_SHOT_REACT_DESCRIPTION.
    """
    # 1) Create the Chat model
    llm = ChatNVIDIA(model="meta/llama-4-maverick-17b-128e-instruct", temperature=0.0)

    # llm = OpenAI(
    #     model_name="gpt-4.0",
    #     openai_api_key=os.getenv("OPENAI_API_KEY"),
    #     temperature=0.0
    # )

    # 2) Build the Pinecone search tool
    pinecone_tool = _pinecone_search_tool(namespace)

    # 3) Initialize a zero-shot agent with that single tool
    system_prompt = """
        You are an agent designed to answer questions step by step.
        On each turn, do only one of:
        - Output an Action (with Action Input) if a tool call is needed.
        - Output a Final Answer if done.
        Never output both Action and Final Answer in a single response.
    """

    agent = initialize_agent(
        tools=[pinecone_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,      # shows “Thought:”, “Action:”, “Observation:”, etc.
        max_iterations=3,  # limit how many times the agent can loop
        early_stopping_method="generate",
        handle_parsing_errors=True, 
        system_message=system_prompt
    )
    return agent