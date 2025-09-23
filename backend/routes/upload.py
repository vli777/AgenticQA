# backend/upload.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
import re

from utils import extract_text_from_pdf_bytes, chunk_document_text
from services import upsert_doc, chunk_text
from logger import logger

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

        # Combine all pages into one text
        full_text = "\n\n".join(doc.page_content for doc in docs)
        
        # First split by paragraphs to maintain document structure
        paragraphs = re.split(r'\n\s*\n', full_text)
        
        # Then process each paragraph into chunks
        chunks = []
        for para in paragraphs:
            # Split paragraph into sentences
            sentences = re.split(r'([.!?])\s+', para)
            current_chunk = ""
            
            # Process sentences in pairs (sentence + punctuation)
            for i in range(0, len(sentences), 2):
                if i + 1 < len(sentences):
                    sentence = sentences[i] + sentences[i + 1]
                else:
                    sentence = sentences[i]
                
                # If adding this sentence would exceed chunk size, save current chunk
                if len(current_chunk) + len(sentence) > 800:  # Smaller chunks for better context
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence
            
            # Add the last chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        # Convert chunks to Documents with metadata
        final_docs = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": filename,
                "chunk_id": i,
                "total_chunks": len(chunks)
            }
            final_docs.append(Document(page_content=chunk, metadata=metadata))
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

        # Combine all text into one
        full_text = "\n\n".join(doc.page_content for doc in docs)
        
        # First split by paragraphs to maintain document structure
        paragraphs = re.split(r'\n\s*\n', full_text)
        
        # Then process each paragraph into chunks
        chunks = []
        for para in paragraphs:
            # Split paragraph into sentences
            sentences = re.split(r'([.!?])\s+', para)
            current_chunk = ""
            
            # Process sentences in pairs (sentence + punctuation)
            for i in range(0, len(sentences), 2):
                if i + 1 < len(sentences):
                    sentence = sentences[i] + sentences[i + 1]
                else:
                    sentence = sentences[i]
                
                # If adding this sentence would exceed chunk size, save current chunk
                if len(current_chunk) + len(sentence) > 800:  # Smaller chunks for better context
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence
            
            # Add the last chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        # Convert chunks to Documents with metadata
        final_docs = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": filename,
                "chunk_id": i,
                "total_chunks": len(chunks)
            }
            final_docs.append(Document(page_content=chunk, metadata=metadata))
        return final_docs

    else:
        raise ValueError("Unsupported file type in loader")


@router.post("/")
async def upload_documents(
    files: List[UploadFile] = File(...),
    namespace: str = "default"
):
    """
    Upload endpoint. By default, uses LangChain's PyPDFLoader/TextLoader to extract + chunk.
    If you want to revert to manual chunking, comment out the LangChain code and uncomment below.
    """
    total_chunks = 0
    total_upserted = 0

    logger.info(
        "Upload request received (files=%s, namespace=%s)",
        [uploaded.filename for uploaded in files],
        namespace,
    )

    for uploaded in files:
        name = uploaded.filename
        data = await uploaded.read()

        logger.info(
            "Processing file '%s' (bytes=%s, namespace=%s)",
            name,
            len(data),
            namespace,
        )

        try:
            docs = load_and_chunk_documents(data, name, max_chars=1000)
        except Exception as e:
            logger.exception("Failed to load file '%s' for indexing", name)
            raise HTTPException(status_code=400, detail=f"Loader error: {e}")

        if not docs:
            logger.warning(
                "Loader produced no chunks for file '%s' (namespace=%s)",
                name,
                namespace,
            )
            continue

        file_chunks_indexed = 0
        file_vectors_upserted = 0

        for doc in docs:
            # Each doc is a langchain.schema.Document with page_content + metadata
            chunk_id = doc.metadata.get("chunk_id", f"{name}__0")
            result = upsert_doc(doc.page_content, doc_id=chunk_id, namespace=namespace)

            file_chunks_indexed += result.get("chunks", 0)
            file_vectors_upserted += result.get("upserted", 0)

            logger.info(
                "Indexed chunk %s from file '%s' (namespace=%s)",
                chunk_id,
                name,
                namespace,
            )

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

        total_chunks += file_chunks_indexed
        total_upserted += file_vectors_upserted

        logger.info(
            "Finished indexing file '%s' (namespace=%s, chunks_indexed=%s, pinecone_vectors=%s)",
            name,
            namespace,
            file_chunks_indexed,
            file_vectors_upserted,
        )

    logger.info(
        "Upload request complete (namespace=%s, total_chunks=%s, total_vectors=%s)",
        namespace,
        total_chunks,
        total_upserted,
    )

    return {"indexed_chunks": total_chunks}
