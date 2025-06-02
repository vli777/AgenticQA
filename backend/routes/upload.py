# backend/upload.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document

from utils import extract_text_from_pdf_bytes, chunk_document_text
from services import upsert_doc

router = APIRouter()

def load_and_chunk_documents(
    file_bytes: bytes,
    filename: str,
    max_chars: int = 1000
) -> List[Document]:
    """
    Use LangChain's document loaders to read and chunk a PDF or TXT.
    Returns a list of Document(page_content, metadata).
    """
    lower = filename.lower()
    if lower.endswith(".pdf"):
        # PyPDFLoader expects a file path, so we need to write bytes to a temp file
        import tempfile, os

        suffix = ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load_and_split()  
        os.unlink(tmp_path)  # clean up temp file

        # By default, load_and_split() will chunk by page or by whitespace; 
        # if you want a fixed max_chars, you can re‐chunk here:
        final_docs: List[Document] = []
        for doc in docs:
            text = doc.page_content
            # manual sub‐chunking to max_chars (if desired)
            if len(text) > max_chars:
                # simple split by max_chars chunks (no overlap)
                for i in range(0, len(text), max_chars):
                    chunk_text = text[i : i + max_chars]
                    metadata = doc.metadata.copy() if doc.metadata else {}
                    # store original page number if present
                    metadata["source"] = filename
                    metadata["chunk_id"] = f"{filename}__{i}"
                    final_docs.append(Document(page_content=chunk_text, metadata=metadata))
            else:
                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata["source"] = filename
                metadata["chunk_id"] = f"{filename}__0"
                final_docs.append(Document(page_content=text, metadata=metadata))
        return final_docs

    elif lower.endswith(".txt"):
        # TextLoader also expects a file path
        import tempfile, os

        suffix = ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="wb") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loader = TextLoader(tmp_path, encoding="utf-8")
        docs = loader.load_and_split()
        os.unlink(tmp_path)

        # Similar re‐chunking by max_chars if needed
        final_docs: List[Document] = []
        for doc in docs:
            text = doc.page_content
            if len(text) > max_chars:
                for i in range(0, len(text), max_chars):
                    chunk_text = text[i : i + max_chars]
                    metadata = doc.metadata.copy() if doc.metadata else {}
                    metadata["source"] = filename
                    metadata["chunk_id"] = f"{filename}__{i}"
                    final_docs.append(Document(page_content=chunk_text, metadata=metadata))
            else:
                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata["source"] = filename
                metadata["chunk_id"] = f"{filename}__0"
                final_docs.append(Document(page_content=text, metadata=metadata))
        return final_docs

    else:
        raise ValueError("Unsupported file type in loader")


@router.post("/")
async def upload_documents(
    files: List[UploadFile] = File(...),
    namespace: str = "default"
):
    """
    Upload endpoint. By default, uses LangChain’s PyPDFLoader/TextLoader to extract + chunk.
    If you want to revert to manual chunking, comment out the LangChain code and uncomment below.
    """
    total_chunks = 0

    for uploaded in files:
        name = uploaded.filename
        data = await uploaded.read()

        try:
            docs = load_and_chunk_documents(data, name, max_chars=1000)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Loader error: {e}")

        for doc in docs:
            # Each doc is a langchain.schema.Document with page_content + metadata
            chunk_id = doc.metadata.get("chunk_id", f"{name}__0")
            upsert_doc(doc.page_content, doc_id=chunk_id, namespace=namespace)
            total_chunks += 1

        # if name.lower().endswith(".pdf"):
        #     try:
        #         text = extract_text_from_pdf_bytes(data)
        #     except Exception as e:
        #         raise HTTPException(status_code=400, detail=f"PDF parse error: {e}")
        # elif name.lower().endswith(".txt"):
        #     text = data.decode("utf-8", errors="ignore")
        # else:
        #     raise HTTPException(status_code=415, detail="Only PDF or TXT allowed")
        #
        # chunks = chunk_document_text(text)
        # for idx, chunk in enumerate(chunks):
        #     chunk_id = f"{name}__chunk{idx}"
        #     upsert_doc(chunk, doc_id=chunk_id, namespace=namespace)
        #     total_chunks += 1

    return {"indexed_chunks": total_chunks}