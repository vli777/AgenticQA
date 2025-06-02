# backend/utils.py

from sentence_transformers import SentenceTransformer
import openai
import io
from pypdf import PdfReader

from config import EMBEDDING_MODEL, OPENAI_API_KEY

_e5 = SentenceTransformer('intfloat/e5-large')

def get_embedding(text: str, model: str = None):
    """
    Returns a list-of-floats embedding for `text` using:
      • local E5-Large if model == "multilingual-e5-large"
      • OpenAI text-embedding-3-small if model == "text-embedding-3-small"
    """
    model = model or EMBEDDING_MODEL
    if model == "multilingual-e5-large":
        vec = _e5.encode([text])[0]  # Returns numpy array
        return vec.astype("float32").tolist()
    elif model == "text-embedding-3-small":
        resp = openai.embeddings.create(input=[text], model="text-embedding-3-small", api_key=OPENAI_API_KEY)
        return resp.data[0].embedding
    raise ValueError(f"get_embedding(): unsupported model_name={model!r}")

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