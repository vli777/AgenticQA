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
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

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


def _pinecone_search_tool(namespace: str = "default", similarity_threshold: float = 0.6) -> Tool:
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
        # Retrieve matching chunks with scores so we can respect the similarity threshold
        hits = vectorstore.similarity_search_with_score(
            query,
            k=20,  # Match regular RAG endpoint's top_k
        )

        # LangChain returns (Document, score) tuples; translate scores to cosine similarity
        qualified = []
        for doc, raw_score in hits:
            meta_score = doc.metadata.get("score")
            if meta_score is not None:
                score = meta_score
            else:
                # For Pinecone cosine searches the raw score is the distance (0 identical → 2 opposite).
                if 0.0 <= raw_score <= 2.0:
                    score = 1.0 - raw_score
                else:
                    score = raw_score

            if score >= similarity_threshold:
                qualified.append((doc, score))

        if not qualified:
            return f"No results found above the {similarity_threshold} similarity threshold. Try searching with different terms."

        # Group results by document
        doc_results = {}
        for doc, score in qualified:
            meta = doc.metadata or {}
            doc_id = meta.get("doc_id", "unknown")
            if doc_id not in doc_results:
                doc_results[doc_id] = []
            doc_results[doc_id].append((doc, score))

        # Format results
        lines: List[str] = []
        for doc_id, chunks in doc_results.items():
            # Sort chunks by their position in the document
            chunks.sort(key=lambda x: x[0].metadata.get("chunk_id", 0))

            # Get the source from the first chunk
            source = chunks[0][0].metadata.get("source", "unknown")

            # Process each chunk individually to avoid combining unrelated information
            for chunk, score in chunks:
                text = chunk.page_content

                # Only include if the text is meaningful
                if len(text.strip()) > 50 and re.search(r'[.!?]', text):
                    if len(text) > 1000:
                        # Try to find a good breaking point
                        sentences = re.split(r'([.!?])\s+', text)
                        text = ""
                        for i in range(0, len(sentences), 2):
                            if i + 1 < len(sentences):
                                sentence = sentences[i] + sentences[i + 1]
                            else:
                                sentence = sentences[i]
                            if len(text) + len(sentence) <= 1000:
                                text += sentence + " "
                            else:
                                break
                        text = text.strip() + "..."
                    # Clean and format the text, include score for transparency
                    text = clean_text(text)
                    lines.append(
                        f"[{source}::{doc_id}] (similarity: {score:.2f}) {text}"
                    )

        if not lines:
            return "No meaningful text chunks found in the results. Try searching with different terms."

        # Return more results that passed the threshold
        return "\n\n".join(lines[:5])  # Return top 5 results

    return Tool(
        name="semantic_search",
        func=search_fn,
        description=(
            "Use this tool to perform a semantic search over uploaded documents. "
            "Input: a query string. Output: up to three matching chunks "
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
        # Use a lower similarity threshold (0.6) for the agentic endpoint to get more context
        tools = [_pinecone_search_tool(namespace=namespace, similarity_threshold=0.6)]

    llm = ChatNVIDIA(model="meta/llama-4-maverick-17b-128e-instruct", temperature=0.0)

    system = """You are a helpful AI assistant that answers questions based on the provided context.
    Follow these steps:
    1. Break down complex questions into simpler search terms
    2. Use the search tool multiple times with different terms if needed
    3. If the first search doesn't yield good results, try alternative phrasings
    4. Only provide your answer after thorough searching
    5. Include all relevant sources in your response

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
    - If you can't find information after multiple searches, say so in your answer
    - Don't make up information - if you can't find it, say you don't know
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # Create the agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # Create a chain that ensures JSON output
    json_parser = JsonOutputParser()
    
    # Create a chain that includes post-processing
    chain = (
        agent_executor
        | (lambda x: {
            "search_results": x["output"],
            "question": x["input"]
        })
        | (lambda x: f"""You are a helpful AI assistant that answers questions based on the provided context.
Your task is to answer this question: {x['question']}

Here are the search results:
{x['search_results']}

IMPORTANT:
1. If the search results contain relevant information, use it to answer the question
2. If the search results don't contain enough information, try searching again with different terms
3. If you still can't find the information after multiple searches, say so explicitly
4. Include ALL relevant sources in your response
5. If you find partial information, explain what you found and what's missing

Please provide your answer in this JSON format:
{{
    "answer": "Your detailed answer here",
    "reasoning": [
        "First, I found...",
        "Then, I discovered...",
        "Finally, I concluded..."
    ],
    "sources": [
        "Include exact source strings here"
    ]
}}""")
        | llm  # Use LLM to reason about the search results
        | json_parser  # Parse to JSON
    )

    return chain
