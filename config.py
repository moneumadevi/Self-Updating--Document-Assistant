"""
config.py
=========
Project settings (all the "knobs" of the app live here, in one place).

Beginner note (Telugu-English):
    Ee file lo manam project settings pedutunnam - folders, chunk size,
    supported file types, answer threshold etc. Code lo hard-code cheyyakunda
    ikkada eenadu marchukovachu.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
# BASE_DIR = the folder where this file lives. Using absolute paths means the
# app works no matter from which folder you start it on Windows.
BASE_DIR = Path(__file__).resolve().parent

# The "watched" folder: any file you drop here (or upload from the web page)
# is picked up automatically.
DOCUMENTS_DIR = Path(os.environ.get("DOCS_ASSISTANT_DOCUMENTS", BASE_DIR / "documents"))

# Ready-made sample documentation used for testing / demo.
SAMPLE_DOCUMENTS_DIR = BASE_DIR / "sample_documents"

# Where the saved (persisted) search index lives.
INDEX_DIR = Path(os.environ.get("DOCS_ASSISTANT_INDEX", BASE_DIR / "index_store"))

# Names of the two files inside INDEX_DIR.
INDEX_BINARY_NAME = "index.pkl"       # vectorizer + TF-IDF matrix + chunk text
INDEX_MANIFEST_NAME = "manifest.json"  # human-readable summary (safe to open)

# ---------------------------------------------------------------------------
# Supported file types
# ---------------------------------------------------------------------------
# Only these extensions are accepted. Everything else is rejected with a clear
# error message (requirement: error handling for unsupported formats).
SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}

# ---------------------------------------------------------------------------
# Chunking (how we split a document into small searchable passages)
# ---------------------------------------------------------------------------
# A chunk of 120 words with 20 words overlapping the next chunk is small enough
# for a precise answer, but big enough to keep the meaning together.
CHUNK_SIZE_WORDS = 120
CHUNK_OVERLAP_WORDS = 20
MIN_CHUNK_WORDS = 8   # ignore tiny leftovers like a single heading

# ---------------------------------------------------------------------------
# TF-IDF settings
# ---------------------------------------------------------------------------
# Single words only. (Two-word phrases lowered every similarity score a lot
# without improving the ranking in our tests, so unigrams are the better default
# for question answering on small documentation sets.)
TFIDF_NGRAM_RANGE = (1, 1)
TFIDF_MIN_DF = 1
TFIDF_MAX_FEATURES = 50_000
TFIDF_SUBLINEAR_TF = True

# Small reranking bonus added when the exact question phrase appears in a
# passage. Base ranking is still cosine similarity (see qa_engine.py).
PHRASE_MATCH_BONUS = 0.10

# ---------------------------------------------------------------------------
# Retrieval / answer settings
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 3            # how many passages we show as evidence
MAX_TOP_K = 8
DEFAULT_MIN_SCORE = 0.10     # below this cosine similarity = "no relevant answer"
# At least this share of the question's meaningful words must appear in the
# winning passage. Stops a single accidental common word from "answering".
MIN_QUERY_COVERAGE = 0.5
MAX_ANSWER_SENTENCES = 2     # extractive answer length (quote + one context line)
MIN_SENTENCE_CHARS = 35

# High / medium confidence cut-offs (only used for the coloured badge in the UI)
HIGH_CONFIDENCE = 0.35
MEDIUM_CONFIDENCE = 0.15

# Bump this number if you change the chunking/vectorizer settings above.
# The saved index stores this value; when it changes the index is rebuilt
# automatically (this avoids silently using an outdated index).
INDEX_VERSION = 5

# Auto-watch refresh interval for the Streamlit fragment (seconds).
AUTO_REFRESH_SECONDS = 3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    """Create the documents/ and index_store/ folders if they do not exist."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def human_supported_formats() -> str:
    """Return a friendly list like 'TXT, MD, PDF' for messages in the UI."""
    return ", ".join(sorted(ext.replace(".", "").upper() for ext in SUPPORTED_EXTENSIONS))


__all__ = [
    "BASE_DIR",
    "DOCUMENTS_DIR",
    "SAMPLE_DOCUMENTS_DIR",
    "INDEX_DIR",
    "INDEX_BINARY_NAME",
    "INDEX_MANIFEST_NAME",
    "SUPPORTED_EXTENSIONS",
    "CHUNK_SIZE_WORDS",
    "CHUNK_OVERLAP_WORDS",
    "MIN_CHUNK_WORDS",
    "TFIDF_NGRAM_RANGE",
    "TFIDF_MIN_DF",
    "TFIDF_MAX_FEATURES",
    "TFIDF_SUBLINEAR_TF",
    "PHRASE_MATCH_BONUS",
    "DEFAULT_TOP_K",
    "MAX_TOP_K",
    "DEFAULT_MIN_SCORE",
    "MIN_QUERY_COVERAGE",
    "MAX_ANSWER_SENTENCES",
    "MIN_SENTENCE_CHARS",
    "HIGH_CONFIDENCE",
    "MEDIUM_CONFIDENCE",
    "INDEX_VERSION",
    "AUTO_REFRESH_SECONDS",
    "ensure_directories",
    "human_supported_formats",
]
