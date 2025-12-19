# backend/structured_output/providers.py

"""Provider-specific implementations for structured LLM outputs."""

import json
import re
from typing import Any, Dict, Set
from logger import logger


def extract_with_nvidia_guided_json(client: Any, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured output using NVIDIA's guided_json approach.

    Args:
        client: LangChain ChatNVIDIA client
        prompt: Prompt text
        schema: JSON schema for output

    Returns:
        Parsed JSON dict

    Raises:
        json.JSONDecodeError: If response is not valid JSON
    """
    json_prompt = f"""{prompt}

Return ONLY valid JSON matching this schema (no markdown, no explanations):
{json.dumps(schema, indent=2)}"""

    # Try with model_kwargs for guided_json
    try:
        response = client.invoke(
            json_prompt,
            model_kwargs={"extra_body": {"guided_json": schema}}
        )
    except (TypeError, Exception) as e:
        # Fallback: invoke without guided_json
        logger.debug(f"guided_json not supported, falling back to regular invoke: {e}")
        response = client.invoke(json_prompt)

    response_text = response.content if hasattr(response, 'content') else str(response)

    # Try direct parsing first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        # Fallback: try to extract JSON from response (might be wrapped in markdown)
        logger.debug(f"Direct JSON parse failed: {e}. Attempting to extract JSON from response.")
        logger.debug(f"Response text: {response_text[:500]}")  # Log first 500 chars for debugging

        # Try to find JSON object in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        # Try to find JSON array in the response
        array_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if array_match:
            result = json.loads(array_match.group())
            # If schema expects object but we got array, wrap it
            if isinstance(result, list) and schema.get("type") == "object":
                # Check if the array items match a property name
                for prop_name, prop_schema in schema.get("properties", {}).items():
                    if prop_schema.get("type") == "array":
                        return {prop_name: result}
            return result

        # Re-raise original error if we couldn't extract JSON
        raise e


def extract_with_openai_structured_output(client: Any, prompt: str, schema: Dict[str, Any], model: str = "gpt-4") -> Dict[str, Any]:
    """
    Extract structured output using OpenAI's native structured output API.

    Args:
        client: OpenAI client
        prompt: Prompt text
        schema: JSON schema for output
        model: OpenAI model name

    Returns:
        Parsed JSON dict

    Raises:
        Exception: If structured output API not available or fails
    """
    try:
        # Try OpenAI's native structured output (beta.chat.completions.parse)
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_schema", "json_schema": {"name": "output", "schema": schema}},
        )
        parsed = response.choices[0].message.parsed
        if parsed:
            # Convert Pydantic model to dict if needed
            if hasattr(parsed, 'model_dump'):
                return parsed.model_dump()
            return parsed
    except (AttributeError, Exception) as e:
        # Fallback to JSON mode if structured output not available
        logger.info(f"OpenAI structured output not available, using JSON mode: {e}")
        return extract_with_openai_json_mode(client, prompt, schema, model)


def extract_with_openai_json_mode(client: Any, prompt: str, schema: Dict[str, Any], model: str = "gpt-4") -> Dict[str, Any]:
    """
    Extract structured output using OpenAI's JSON mode fallback.

    Args:
        client: OpenAI client
        prompt: Prompt text
        schema: JSON schema for output
        model: OpenAI model name

    Returns:
        Parsed JSON dict

    Raises:
        json.JSONDecodeError: If response is not valid JSON
    """
    json_prompt = f"""{prompt}

Return ONLY a valid JSON object matching this schema: {schema}"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": json_prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content

    # Try to extract JSON from response (might be wrapped in markdown)
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return json.loads(content)


def extract_with_langchain_structured_output(client: Any, prompt: str, pydantic_model: Any) -> Dict[str, Any]:
    """
    Extract structured output using LangChain's with_structured_output (fallback for compatible models).

    Args:
        client: LangChain client with with_structured_output support
        prompt: Prompt text
        pydantic_model: Pydantic model class

    Returns:
        Parsed dict from Pydantic model

    Raises:
        Exception: If structured output fails or returns None
    """
    if not hasattr(client, "with_structured_output"):
        raise AttributeError("Client does not support with_structured_output")

    structured_llm = client.with_structured_output(pydantic_model)
    response = structured_llm.invoke(prompt)

    if response is None or not hasattr(response, 'model_dump'):
        raise ValueError("LLM returned None or invalid response")

    return response.model_dump()
