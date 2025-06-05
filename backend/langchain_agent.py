# backend/langchain_agent.py

import os
import json
import re
from typing import List, Any, Dict, TypedDict, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LC_Pinecone
from langchain.tools import Tool

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
        # Use OpenAI's text-embedding-3-small via langchain_openai
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
    then formats the top-5 hits as "[source::chunk_id] snippet…", following the Pinecone docs.
    """
    vectorstore = _get_vectorstore(namespace)

    def clean_text(text: str) -> str:
        """Clean and format text for better readability."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Fix spacing around numbers
        text = re.sub(r'(\d+)\s+', r'\1 ', text)
        # Fix spacing after punctuation
        text = re.sub(r'([.!?])\s+', r'\1 ', text)
        # Remove page numbers at the end
        text = re.sub(r'\s+\d+$', '', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        # Ensure proper sentence structure
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        if text and not text[-1] in '.!?':
            text += '.'
        return text

    def search_fn(query: str) -> str:
        # Retrieve up to 10 matching chunks with higher threshold
        docs = vectorstore.similarity_search(
            query,
            k=10,
            score_threshold=0.8  # Higher threshold for better quality
        )
        
        # Group results by document
        doc_results = {}
        for doc in docs:
            meta = doc.metadata or {}
            doc_id = meta.get("doc_id", "unknown")
            if doc_id not in doc_results:
                doc_results[doc_id] = []
            doc_results[doc_id].append(doc)
        
        # Format results
        lines: List[str] = []
        for doc_id, chunks in doc_results.items():
            # Sort chunks by their position in the document
            chunks.sort(key=lambda x: x.metadata.get("chunk_id", 0))
            
            # Get the source from the first chunk
            source = chunks[0].metadata.get("source", "unknown")
            
            # Combine consecutive chunks if they're from the same document
            combined_text = " ".join(chunk.page_content for chunk in chunks)
            
            # Only include if the combined text is meaningful
            if len(combined_text.strip()) > 100 and re.search(r'[.!?]', combined_text):
                if len(combined_text) > 1000:
                    # Try to find a good breaking point
                    sentences = re.split(r'([.!?])\s+', combined_text)
                    combined_text = ""
                    for i in range(0, len(sentences), 2):
                        if i + 1 < len(sentences):
                            sentence = sentences[i] + sentences[i + 1]
                        else:
                            sentence = sentences[i]
                        if len(combined_text) + len(sentence) <= 1000:
                            combined_text += sentence + " "
                        else:
                            break
                    combined_text = combined_text.strip() + "..."
                
                # Clean and format the text
                combined_text = clean_text(combined_text)
                lines.append(f"[{source}::{doc_id}] {combined_text}")
        
        if not lines:
            return "No relevant information found."
            
        # Return top 5 most relevant results with extra spacing
        return "\n\n".join(lines[:5])

    return Tool(
        name="semantic_search",
        func=search_fn,
        description=(
            "Use this tool to perform a semantic search over uploaded documents. "
            "Input: a query string. Output: up to five matching chunks "
            "formatted as '[source::doc_id] snippet...'."
        )
    )


class AgentOutput(TypedDict):
    answer: str
    reasoning: str
    sources: Optional[List[str]]

def extract_json_from_text(text: str) -> Dict:
    """Extract JSON from text, handling cases where there might be text before or after the JSON."""
    try:
        # First try to parse the entire text as JSON
        if isinstance(text, dict):
            return text
        return json.loads(text)
    except json.JSONDecodeError:
        # If that fails, try to find JSON in the text
        if isinstance(text, str):
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
    return None

def get_agent(namespace: str = "default", tools: list = None):
    """
    Returns a LangChain chain that can use our Pinecone "semantic_search" tool.
    Uses NVIDIA's Llama model with structured output.
    """
    if tools is None:
        tools = [_pinecone_search_tool(namespace=namespace)]

    llm = ChatNVIDIA(model="meta/llama-4-maverick-17b-128e-instruct", temperature=0.0)

    system = """You are a helpful AI assistant that answers questions based on the provided context.
    Follow these steps:
    1. Think about how to answer the question
    2. If you need more information, use the search tool
    3. Provide your answer with reasoning
    4. Include relevant sources if you used the search tool

    CRITICAL: You MUST output ONLY a JSON object, with no text before or after it. DO NOT include your thought process or reasoning in the output.
    The JSON must look exactly like this:
    {{
        "answer": "The final answer to the question",
        "reasoning": [
            "First, I considered...",
            "Then, I looked at...",
            "Finally, I concluded..."
        ],
        "sources": [
            "[source1::chunk_id] relevant snippet...",
            "[source2::chunk_id] relevant snippet..."
        ]
    }}

    Rules:
    - Output ONLY the JSON object, nothing else
    - DO NOT include any text before or after the JSON
    - DO NOT include your thought process in the output
    - The answer field must be a string with your final answer
    - The reasoning field must be an array of strings, where each string is one step in your reasoning
    - The sources field must be an array of strings, where each string is a source from the search tool
    - Make sure the JSON is properly formatted with double quotes
    - Each step in reasoning should be a separate string in the array
    - If you used the search tool, include the exact source strings in the sources array
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{input}")
    ])

    # Create a chain that ensures JSON output
    json_parser = JsonOutputParser()
    
    # Create a chain that includes post-processing
    chain = (
        prompt 
        | llm 
        | json_parser  # Parse directly to JSON
    )

    return chain