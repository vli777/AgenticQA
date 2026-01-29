# backend/agent/composer.py

"""Answer composition, verification, and source extraction."""

from typing import List, Dict, Optional

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from logger import logger
from .models import VerificationVerdict


def _extract_sources(obs: str, limit: int = 5) -> List[str]:
    """Extract provenance-tagged source lines from observation text."""
    if not isinstance(obs, str):
        return []
    out: List[str] = []
    seen = set()
    for ln in obs.splitlines():
        ln = ln.strip()
        if ln.startswith("["):
            if ln not in seen:
                out.append(ln)
                seen.add(ln)
            if len(out) >= limit:
                break
    return out


def _compose_answer(
    llm: ChatNVIDIA,
    question: str,
    evidence: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Compose a natural, conversational answer grounded in evidence."""
    history_section = ""
    if chat_history:
        history_lines = "\n".join(f"{item['role']}: {item['content']}" for item in chat_history)
        history_section = f"\n\nRecent conversation:\n{history_lines}\n"
    if not evidence or evidence.strip().lower() in {"no results.", "no meaningful text chunks found."}:
        return "I don't see information about that in the documents."

    prompt = (
        "You are a helpful assistant that answers questions using the provided evidence.\n"
        "You DO have access to all of the evidence text shown below; treat it as your only source of truth.\n"
        "Rules:\n"
        "1. Directly answer the user's question using ONLY the evidence.\n"
        "2. Do NOT mention tools, PDFs, file formats, or that you \"don't have access\" to something.\n"
        "3. Do NOT describe how search or retrieval was done.\n"
        "4. If evidence clearly answers the question: give a direct, confident answer.\n"
        "5. If evidence is partial or uncertain: say so explicitly, but still answer as best you can.\n"
        "6. If there is no relevant evidence at all, say exactly: \"I don't see information about that in the documents.\"\n"
        "7. Keep answers concise (1-3 sentences) and natural.\n\n"
        f"Question: {question}\n"
        f"{history_section}\n"
        f"Evidence:\n{evidence}\n\n"
        "Answer:"
    )
    response = llm.invoke(prompt)
    text = getattr(response, "content", "").strip()
    if not text:
        return "I don't see information about that in the documents."
    return text


def _verify_answer(llm: ChatNVIDIA, question: str, evidence: str, answer: str) -> str:
    """
    Verify whether the answer is supported by the evidence.
    Uses Pydantic structured output for reliable verdict parsing.
    """
    if not answer:
        return "UNSUPPORTED"

    lower = answer.strip().lower()

    # Treat meta/limitation answers as unsupported
    if any(phrase in lower for phrase in [
        "i don't have access",
        "i do not have access",
        "i can't access",
        "i cannot access",
        "i don't have the content",
        "i do not have the content",
        "i would need access",
        "i need access to",
        "the pdf",
        "the document is not",
    ]):
        logger.warning(f"Answer contains meta-talk about access/tools, marking UNSUPPORTED: '{answer[:100]}'")
        return "UNSUPPORTED"

    if lower == "the documents do not clearly specify this.":
        return "UNSUPPORTED"

    prompt = (
        "You are a strict fact checker. "
        "Given a question, an answer, and evidence snippets, determine the verification verdict.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Evidence:\n{evidence}\n"
    )

    try:
        # Use structured output with Pydantic model
        structured_llm = llm.with_structured_output(VerificationVerdict)
        response = structured_llm.invoke(prompt)
        return response.verdict
    except Exception as e:
        # Fallback to PARTIAL if structured output fails
        logger.warning(f"Verification failed with structured output: {e}, defaulting to PARTIAL")
        return "PARTIAL"
