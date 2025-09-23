# backend/langchain_agent.py

import os
import json
import re
from typing import List, Any, Dict, TypedDict, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
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

        # LangChain returns (Document, score) tuples; normalise the numeric score to a similarity.
        qualified: List[tuple] = []
        ranked: List[tuple] = []
        for doc, raw_score in hits:
            score = doc.metadata.get("score", raw_score)

            try:
                score = float(score)
            except (TypeError, ValueError):
                continue

            if -1.0 <= score <= 1.0:
                similarity = score
            elif 0.0 <= score <= 2.0:
                similarity = 1.0 - score
            else:
                similarity = score

            similarity = max(min(similarity, 1.0), -1.0)

            ranked.append((doc, similarity))

            if similarity >= similarity_threshold:
                qualified.append((doc, similarity))

        low_confidence = False
        if not qualified and ranked:
            low_confidence = True
            ranked.sort(key=lambda x: x[1], reverse=True)
            qualified = ranked[:5]

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
            meta0 = chunks[0][0].metadata or {}
            source = meta0.get("source") or meta0.get("file_name") or "unknown"

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
                    section = chunk.metadata.get("section_index")
                    if section is not None:
                        provenance = f"{source}::section-{section}"
                    else:
                        provenance = f"{source}::{doc_id}"

                    lines.append(
                        f"[{provenance}] (similarity: {score:.2f}) {text}"
                    )

        if low_confidence and lines:
            lines.insert(0, "Results below similarity threshold; showing best available matches.")

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
        # Use a fairly permissive similarity threshold so the agent can reason over weaker matches
        tools = [_pinecone_search_tool(namespace=namespace, similarity_threshold=0.3)]

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
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    def _invoke(inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = agent_executor.invoke(inputs)
        output = result.get("output", result)

        parsed = extract_json_from_text(output)
        if parsed is None:
            parsed = {
                "answer": str(output),
                "reasoning": result.get("intermediate_steps", []),
                "sources": [],
            }

        return parsed

    return RunnableLambda(_invoke)
