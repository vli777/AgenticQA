# backend/patterns.py

"""
Pre-compiled regex patterns for performance optimization.
Compiled once at module load time and reused throughout the application.
"""

import re

# Text cleaning patterns
WHITESPACE_PATTERN = re.compile(r'\s+')
SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s.,!?+#/-]')

# Text splitting patterns
PARAGRAPH_SPLIT_PATTERN = re.compile(r'\n\s*\n')
SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+|•\s+')

# Chunk validation patterns
NUMBERS_ONLY_PATTERN = re.compile(r'^[\d\s.,]+$')
NAME_PATTERN = re.compile(r'^[A-Z][a-z]+\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)*$')
URL_PATTERN = re.compile(r'^.*\d{4}\.?\s+URL\s+https?://.*$')
ARXIV_PATTERN = re.compile(r'^.*arXiv:?\d{4}\.\d{4,5}.*$')
DIGITS_ONLY_PATTERN = re.compile(r'^\d+$')
PUNCTUATION_PATTERN = re.compile(r'[.!?]')
WORD_PATTERN = re.compile(r'\b[\w+#]+\b')
