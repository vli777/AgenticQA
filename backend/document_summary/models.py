# backend/document_summary/models.py

"""Pydantic models for document summary extraction."""

from typing import List, Literal

from pydantic import BaseModel, Field


class KeyConcept(BaseModel):
    """A key concept or important term from the document."""
    concept: str = Field(description="Important term, technology, name, idea, or skill")
    context: str = Field(description="Brief context about how it's used or mentioned")
    chunk_refs: List[int] = Field(description="Reference to [CHUNK_X] numbers where information was found")


class KeyFact(BaseModel):
    """An important statement or claim from the document."""
    fact: str = Field(description="Important statement or claim")
    chunk_refs: List[int] = Field(description="Reference to [CHUNK_X] numbers where information was found")


class DocumentExtraction(BaseModel):
    """Structured extraction of key information from a document."""
    document_type: Literal["resume", "research_paper", "technical_doc", "article", "business_doc", "other"] = Field(
        description="Type of document"
    )
    primary_subject: str = Field(
        description="Brief description of what this document is about"
    )
    key_concepts: List[KeyConcept] = Field(
        description="List of important concepts mentioned (skills, technologies, methods, people, companies, theories, tools, etc.)"
    )
    key_facts: List[KeyFact] = Field(
        description="List of 5-15 most important facts that answer potential questions"
    )
    topics: List[str] = Field(
        description="List of 3-5 main topics or themes"
    )
