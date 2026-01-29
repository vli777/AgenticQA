# backend/utils/file_extraction.py

import io

from pypdf import PdfReader
from docx import Document


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Read all pages from a PDF (given as raw bytes) and return their concatenated text.
    Relies on PyPDF2 (PdfReader) to extract page-by-page.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    """
    Extract text from a DOCX file (given as raw bytes) and return the concatenated text.
    Uses python-docx to read all paragraphs.
    """
    doc = Document(io.BytesIO(docx_bytes))
    text_parts = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text)
    return "\n".join(text_parts)


def chunk_document_text(full_text: str, max_chars: int = 1000) -> list[str]:
    """
    Split a long string into chunks of roughly `max_chars` characters each.
    This version splits on double-newlines when possible; if a paragraph
    would exceed max_chars, it starts a new chunk.
    """
    paragraphs = full_text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph would exceed max_chars, flush current chunk
        if len(current) + len(para) + 2 > max_chars:
            chunks.append(current.strip())
            current = para
        else:
            if current:
                current += "\n\n" + para
            else:
                current = para

    if current:
        chunks.append(current.strip())

    return chunks
