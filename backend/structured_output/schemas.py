# backend/structured_output/schemas.py

"""JSON schemas for structured LLM outputs."""

from typing import Dict, Any


def get_tag_schema(tag_count: int) -> Dict[str, Any]:
    """
    Get JSON schema for semantic tag extraction.

    Args:
        tag_count: Number of tags to extract

    Returns:
        JSON schema dict
    """
    return {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": f"Array of {tag_count} lowercase keyword strings"
            }
        },
        "required": ["tags"]
    }


def get_document_summary_schema() -> Dict[str, Any]:
    """
    Get JSON schema for document summary extraction.

    Returns:
        JSON schema dict
    """
    return {
        "type": "object",
        "properties": {
            "document_type": {
                "type": "string",
                "enum": ["resume", "research_paper", "technical_doc", "article", "business_doc", "other"]
            },
            "primary_subject": {"type": "string"},
            "key_concepts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "concept": {"type": "string"},
                        "context": {"type": "string"},
                        "chunk_refs": {
                            "type": "array",
                            "items": {"type": "integer"}
                        }
                    },
                    "required": ["concept", "context", "chunk_refs"]
                }
            },
            "key_facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fact": {"type": "string"},
                        "chunk_refs": {
                            "type": "array",
                            "items": {"type": "integer"}
                        }
                    },
                    "required": ["fact", "chunk_refs"]
                }
            },
            "topics": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["document_type", "primary_subject", "key_concepts", "key_facts", "topics"]
    }


def get_batch_document_summary_schema(num_documents: int) -> Dict[str, Any]:
    """
    Get JSON schema for batch document summary extraction.

    Args:
        num_documents: Number of documents in the batch

    Returns:
        JSON schema dict
    """
    single_schema = get_document_summary_schema()

    return {
        "type": "object",
        "properties": {
            "documents": {
                "type": "array",
                "items": single_schema,
                "minItems": num_documents,
                "maxItems": num_documents
            }
        },
        "required": ["documents"]
    }
