"""
Tests for text_processor.py - cleaning, chunking and sentence splitting.

Purpose: the search quality depends on good chunks. These tests prove that
chunks are within the configured size, overlap correctly, and keep the source
filename/page information.
"""

from __future__ import annotations

from document_loader import ExtractedDocument, Page
from text_processor import (
    Chunk,
    chunk_document,
    clean_text,
    short_text,
    split_sentences,
    text_preview,
)


def _document(text: str, name: str = "notes.txt", pages=None) -> ExtractedDocument:
    from pathlib import Path

    return ExtractedDocument(
        filename=name,
        path=Path(name),
        pages=pages or [Page(number=None, text=text)],
    )


def test_clean_text_removes_markdown_noise_but_keeps_words():
    raw = "# Title\n\nSome **bold** text with a [link](https://x.com).\n\n```python\nprint('hi')\n```"

    cleaned = clean_text(raw, is_markdown=True)

    assert "#" not in cleaned
    assert "**" not in cleaned
    assert "https://" not in cleaned
    assert "bold" in cleaned
    assert "link" in cleaned
    assert "print" in cleaned


def test_clean_text_normalises_windows_line_endings():
    cleaned = clean_text("first line\r\nsecond line\r\n\r\n\r\n\r\nthird")

    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned


def test_chunking_splits_long_text_with_overlap():
    words = [f"word{i}" for i in range(300)]
    document = _document(" ".join(words))

    chunks = chunk_document(document, size=100, overlap=20, min_words=10)

    assert len(chunks) > 2
    assert all(chunk.word_count <= 100 for chunk in chunks)
    # 20-word overlap: the last 20 words of chunk 1 start chunk 2.
    first_tail = chunks[0].text.split()[-20:]
    assert chunks[1].text.split()[:20] == first_tail
    assert chunks[0].filename == "notes.txt"


def test_short_document_becomes_one_chunk():
    document = _document("This is a short but perfectly valid note about caching.")

    chunks = chunk_document(document, size=120, overlap=20)

    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].chunk_id == "notes.txt::p0::c0"


def test_empty_text_produces_no_chunks():
    assert chunk_document(_document("   \n\n  ")) == []


def test_invalid_chunk_settings_raise_value_error():
    document = _document("some text here")

    try:
        chunk_document(document, size=50, overlap=50)
    except ValueError as error:
        assert "overlap" in str(error).lower()
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for overlap >= size")


def test_pdf_pages_keep_page_numbers():
    pages = [Page(number=1, text="Page one about indexing " * 10), Page(number=2, text="Page two about queries " * 10)]

    chunks = chunk_document(_document("", name="faq.pdf", pages=pages), size=20, overlap=5)

    assert {chunk.page for chunk in chunks} == {1, 2}
    assert chunks[0].location.startswith("page 1")


def test_chunk_dict_roundtrip():
    chunk = Chunk(chunk_id="a.txt::p0::c0", filename="a.txt", page=None, index=0, text="hello world", word_count=2)

    restored = Chunk.from_dict(chunk.to_dict())

    assert restored == chunk


def test_split_sentences_handles_questions_and_abbreviations():
    sentences = split_sentences("What is TF-IDF? It is a weighting method. It uses IDF weights.")

    assert len(sentences) == 3
    assert sentences[0].endswith("?")


def test_short_text_and_preview():
    assert short_text("a" * 300, max_chars=50).endswith("…")
    assert text_preview("one two three four", max_words=2) == "one two …"


def test_normalise_term_matches_word_forms():
    from text_processor import normalise_term, tokenize

    assert normalise_term("installing") == normalise_term("install")
    assert normalise_term("documents") == normalise_term("document")
    assert normalise_term("libraries") == normalise_term("library")
    assert normalise_term("tf-idf") == "tf-idf"
    assert tokenize("Installing Documents, TF-IDF!") == ["install", "document", "tf-idf"]


def test_clean_text_gives_headings_a_full_stop():
    cleaned = clean_text("Limitations of TF-IDF\n\nThe score can be zero.", is_markdown=False)

    assert "Limitations of TF-IDF." in cleaned
