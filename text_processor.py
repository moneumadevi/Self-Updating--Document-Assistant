"""
text_processor.py
=================
Cleans extracted text and splits it into small searchable "chunks".

Beginner note (Telugu-English):
    Oka pedda document ni direct ga search cheyalem, enduku ante answer chala
    peddaga vastundi. So manam document ni 100-150 words unna chinna
    "chunks" (passages) ga vibhajistam. Prathi chunk ki file name + page
    number save chestam - answer chupinchettappudu source cheppadaniki.
    Chunks madhya konchem overlap (20 words) unchutunnam, so oka sentence
    rendu chunks madhya cut avvadu.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import config
from document_loader import ExtractedDocument

# ---------------------------------------------------------------------------
# Regexes (compiled once, reused - faster)
# ---------------------------------------------------------------------------
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t\u00a0]+")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_HTML_TAG = re.compile(r"<[^>]+>")
_MD_BOLD = re.compile(r"(\*\*|__)(.*?)\1")
_MD_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_HR = re.compile(r"^\s*([-*_])\1{2,}\s*$", re.MULTILINE)
_CODE_FENCE = re.compile(r"^\s*```.*$", re.MULTILINE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'_-]+")
_SUFFIXES = ("ings", "ingly", "ing", "edly", "ies", "es", "ed", "ly", "s")


# ---------------------------------------------------------------------------
# Chunk data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Chunk:
    """One searchable passage of a document."""

    chunk_id: str          # unique id, e.g. "python_basics.md::p0::c3"
    filename: str          # source file name (shown to the user)
    page: int | None       # PDF page number, None for TXT/MD
    index: int             # position of this chunk inside the file (0, 1, 2, ...)
    text: str              # the passage itself
    word_count: int

    @property
    def location(self) -> str:
        """Human readable location, e.g. 'page 4, chunk 2'."""
        if self.page:
            return f"page {self.page}, chunk {self.index + 1}"
        return f"chunk {self.index + 1}"

    def to_dict(self) -> dict:
        """Plain dict so the chunk can be saved as JSON/pickle."""
        return {
            "chunk_id": self.chunk_id,
            "filename": self.filename,
            "page": self.page,
            "index": self.index,
            "text": self.text,
            "word_count": self.word_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            chunk_id=str(data["chunk_id"]),
            filename=str(data["filename"]),
            page=data.get("page"),
            index=int(data["index"]),
            text=str(data["text"]),
            word_count=int(data.get("word_count", len(str(data["text"]).split()))),
        )


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str, is_markdown: bool = False) -> str:
    """
    Normalise messy text so that searching works on real words.

    * Markdown syntax (#, **, links, images, ``` fences) is removed but the
      readable words are kept.
    * Windows line endings (\\r\\n), tabs and extra blank lines are fixed.
    * Lines are stripped so the chunks look neat.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    text = text.replace("\u00ad", "")  # soft hyphen from PDFs

    if is_markdown:
        text = _CODE_FENCE.sub("", text)
        text = _MD_IMAGE.sub(r"\1", text)
        text = _MD_LINK.sub(r"\1", text)
        text = _MD_HTML_TAG.sub(" ", text)
        text = _MD_BOLD.sub(r"\2", text)
        text = _MD_INLINE_CODE.sub(r"\1", text)
        text = _MD_HEADING.sub("", text)
        text = _MD_HR.sub("", text)

    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]

    # Short standalone lines are usually headings ("Limitations of TF-IDF").
    # Give them a full stop so the answer builder does not glue a heading onto
    # the next sentence.
    for position, line in enumerate(lines):
        if not line or line[-1] in ".!?:;)\"'":
            continue
        next_line_is_blank = position + 1 >= len(lines) or not lines[position + 1]
        if next_line_is_blank and len(line.split()) <= 12:
            lines[position] = line + "."

    text = "\n".join(lines)

    # PDF text often has words joined across lines; join single line breaks
    # that are NOT followed by a blank line or a list bullet.
    text = re.sub(r"(?<![.!?:;])\n(?![ \t]*[-*+\d])", " ", text)
    text = _MULTI_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def word_tokens(text: str) -> list[str]:
    """Split text into words (used for word counts and chunking)."""
    return text.split()


def normalise_term(term: str) -> str:
    """
    Very small suffix stripper so that "install", "installing" and "installed"
    become the same search term.

    This is NOT a full linguistic stemmer like Porter - it is a simple rule that
    is easy to explain in a viva: chop a common English ending when at least
    three letters remain. Both documents and questions use the same rule, so the
    matching stays consistent.
    """
    term = term.lower()
    if len(term) < 5:
        return term
    for suffix in _SUFFIXES:
        if term.endswith(suffix) and len(term) - len(suffix) >= 3:
            base = term[: -len(suffix)]
            if suffix == "ies":
                return base + "y"
            return base
    return term


def tokenize(text: str) -> list[str]:
    """Turn text into normalised search tokens (used by the TF-IDF vectorizer)."""
    return [normalise_term(match) for match in _TOKEN.findall(text or "")]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _chunk_page_text(
    text: str,
    size: int,
    overlap: int,
    min_words: int,
) -> list[str]:
    """
    Sliding-window chunking over the words of one page.

    Example with size=5, overlap=2:
        words:  1 2 3 4 5 6 7 8 9
        chunk1: 1 2 3 4 5
        chunk2:     4 5 6 7 8     <- 2 words overlap
        chunk3:         7 8 9
    """
    words = word_tokens(text)
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)] if len(words) >= min_words else []

    chunks: list[str] = []
    step = max(1, size - overlap)
    start = 0
    while start < len(words):
        window = words[start : start + size]
        if len(window) >= min_words:
            chunks.append(" ".join(window))
        if start + size >= len(words):
            break
        start += step
    return chunks


def chunk_document(
    document: ExtractedDocument,
    size: int = config.CHUNK_SIZE_WORDS,
    overlap: int = config.CHUNK_OVERLAP_WORDS,
    min_words: int = config.MIN_CHUNK_WORDS,
    markdown: bool | None = None,
) -> list[Chunk]:
    """
    Turn a loaded document into a list of Chunk objects.

    Markdown is auto-detected from the file extension, so links/headings do not
    pollute the search index.
    """
    if size <= 0:
        raise ValueError("Chunk size must be greater than 0 words.")
    if overlap < 0 or overlap >= size:
        raise ValueError("Chunk overlap must be >= 0 and smaller than the chunk size.")

    if markdown is None:
        markdown = document.path.suffix.lower() in {".md", ".markdown"}

    chunks: list[Chunk] = []
    counter = 0
    for page in document.pages:
        cleaned = clean_text(page.text, is_markdown=markdown)
        for piece in _chunk_page_text(cleaned, size, overlap, min_words):
            chunk = Chunk(
                chunk_id=f"{document.filename}::p{page.number or 0}::c{counter}",
                filename=document.filename,
                page=page.number,
                index=counter,
                text=piece,
                word_count=len(piece.split()),
            )
            chunks.append(chunk)
            counter += 1
    return chunks


# ---------------------------------------------------------------------------
# Sentence helpers (used by the answer builder)
# ---------------------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    """Split a passage into sentences (simple, dependency-free rule based)."""
    text = _WHITESPACE.sub(" ", text.replace("\n", " ")).strip()
    if not text:
        return []

    parts = _SENTENCE_SPLIT.split(text)
    sentences: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Merge very short fragments (e.g. "Q.3") with the next sentence.
        if sentences and len(sentences[-1]) < 12:
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part)
    return sentences


def short_text(text: str, max_chars: int = 220) -> str:
    """Shorten a passage for compact display ('...' is added when cut)."""
    text = _WHITESPACE.sub(" ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def text_preview(text: str, max_words: int = 18) -> str:
    """First few words of a chunk - used in the Documents tab."""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " …"


__all__ = [
    "Chunk",
    "clean_text",
    "word_tokens",
    "normalise_term",
    "tokenize",
    "chunk_document",
    "split_sentences",
    "short_text",
    "text_preview",
]
