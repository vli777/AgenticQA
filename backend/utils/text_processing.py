# backend/utils/text_processing.py

from typing import List
from collections import deque

from config import CHUNK_SIZE, CHUNK_OVERLAP
from logger import logger
from patterns import (
    WHITESPACE_PATTERN,
    SPECIAL_CHARS_PATTERN,
    PARAGRAPH_SPLIT_PATTERN,
    SENTENCE_SPLIT_PATTERN,
    NUMBERS_ONLY_PATTERN,
    NAME_PATTERN,
    URL_PATTERN,
    ARXIV_PATTERN,
    DIGITS_ONLY_PATTERN,
    PUNCTUATION_PATTERN,
    WORD_PATTERN,
)


def clean_text(text: str) -> str:
    """Clean and normalize text while keeping symbols like +/# for skills."""
    text = WHITESPACE_PATTERN.sub(' ', text)
    text = SPECIAL_CHARS_PATTERN.sub('', text)
    return text.strip()


def is_meaningful_chunk(text: str) -> bool:
    """Check if a chunk contains meaningful content."""
    text = text.strip()
    if not text:
        return False

    if NUMBERS_ONLY_PATTERN.match(text):
        return False
    if NAME_PATTERN.match(text):
        return False
    if URL_PATTERN.match(text):
        return False
    if ARXIV_PATTERN.match(text):
        return False
    if DIGITS_ONLY_PATTERN.match(text):
        return False

    # Allow short bullet lists / skill inventories without punctuation
    if not PUNCTUATION_PATTERN.search(text):
        tokens = WORD_PATTERN.findall(text)
        unique_terms = len(set(t.lower() for t in tokens))
        capitalized_terms = len([t for t in tokens if t and t[0].isupper()])
        if unique_terms == 0 and capitalized_terms == 0:
            return False

    return True


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """
    Split text into sentence-aligned chunks with overlap.

    Args:
        chunk_size: Characters per chunk (default from config: 800)
        chunk_overlap: Overlap between chunks (default from config: 300, ~37%)
    """
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP

    logger.info(
        "Chunking with size=%s, overlap=%s (%.1f%%)",
        chunk_size,
        chunk_overlap,
        chunk_overlap / chunk_size * 100,
    )

    # Clean text once before splitting
    text = clean_text(text)

    raw_sections = PARAGRAPH_SPLIT_PATTERN.split(text)
    sentences: List[str] = []
    for section in raw_sections:
        if not section:
            continue
        # Split into sentences but keep bullet-like fragments
        parts = SENTENCE_SPLIT_PATTERN.split(section)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1  # account for space
        if current_len + sentence_len > chunk_size and current_sentences:
            chunk_text_value = " ".join(current_sentences).strip()
            if is_meaningful_chunk(chunk_text_value):
                chunks.append(chunk_text_value)

            # Build overlap from the tail sentences (using deque for O(1) insertions)
            overlap_sentences = deque()
            overlap_len = 0
            for prev_sentence in reversed(current_sentences):
                prev_len = len(prev_sentence) + 1
                if overlap_len + prev_len > chunk_overlap:
                    break
                overlap_sentences.appendleft(prev_sentence)
                overlap_len += prev_len

            current_sentences = list(overlap_sentences)
            current_len = overlap_len

        current_sentences.append(sentence)
        current_len += sentence_len

    if current_sentences:
        chunk_text_value = " ".join(current_sentences).strip()
        if is_meaningful_chunk(chunk_text_value):
            chunks.append(chunk_text_value)

    return chunks
