"""
Tests for document_loader.py - reading TXT, MD, PDF and the error handling.

Purpose: make sure a supported file gives text, and an unsupported/empty/broken
file gives a clear error instead of crashing the app.
"""

from __future__ import annotations

import pytest

from document_loader import (
    DocumentReadError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
    extension_of,
    is_supported_file,
    list_document_files,
    load_document,
    safe_upload_filename,
    unique_destination,
)


def test_load_txt_returns_text(docs_dir):
    path = docs_dir / "notes.txt"
    path.write_text("TF-IDF converts text into numbers.\nCosine similarity compares them.", encoding="utf-8")

    document = load_document(path)

    assert document.filename == "notes.txt"
    assert "TF-IDF" in document.text
    assert document.word_count > 5
    assert document.encoding in {"utf-8", "utf-8-sig"}


def test_load_markdown_keeps_readable_words(docs_dir):
    path = docs_dir / "guide.md"
    path.write_text("# Heading\n\nUse **bold** and [links](https://example.com).", encoding="utf-8")

    document = load_document(path)

    assert "Heading" in document.text
    assert "links" in document.text


def test_load_pdf_reads_every_page(sample_dir):
    document = load_document(sample_dir / "project_faq.pdf")

    assert len(document.pages) == 2
    assert document.pages[0].number == 1
    assert "Student Project FAQ" in document.pages[0].text
    assert "reset the search index" in document.text.lower()


def test_unsupported_extension_raises(docs_dir):
    path = docs_dir / "table.csv"
    path.write_text("a,b\n1,2", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError) as error:
        load_document(path)
    assert "unsupported format" in str(error.value).lower()


def test_empty_file_raises(docs_dir):
    path = docs_dir / "empty.txt"
    path.write_text("", encoding="utf-8")

    with pytest.raises(EmptyDocumentError) as error:
        load_document(path)
    assert "empty" in str(error.value).lower()


def test_whitespace_only_file_raises(docs_dir):
    path = docs_dir / "spaces.md"
    path.write_text("   \n\n\t  \n", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        load_document(path)


def test_missing_file_raises(docs_dir):
    with pytest.raises(DocumentReadError):
        load_document(docs_dir / "does_not_exist.txt")


def test_windows_encoded_file_is_decoded(docs_dir):
    path = docs_dir / "windows.txt"
    path.write_bytes("Machine learning is fun – café".encode("cp1252"))

    document = load_document(path)

    assert "Machine learning" in document.text
    assert document.encoding == "cp1252"


def test_broken_pdf_raises_read_error(docs_dir):
    path = docs_dir / "broken.pdf"
    path.write_bytes(b"%PDF-1.4 this is not really a pdf")

    with pytest.raises(DocumentReadError):
        load_document(path)


def test_safe_upload_filename_removes_folders_and_fixes_extension():
    assert safe_upload_filename("../../My Notes (final).MD") == "My Notes (final).md"
    assert safe_upload_filename("a<b>c.txt") == "a_b_c.txt"


def test_unique_destination_avoids_overwrite(docs_dir):
    (docs_dir / "report.txt").write_text("old", encoding="utf-8")

    destination = unique_destination(docs_dir, "report.txt")

    assert destination.name == "report (1).txt"
    assert not destination.exists()


def test_list_document_files_ignores_unsupported(docs_dir):
    (docs_dir / "a.txt").write_text("alpha", encoding="utf-8")
    (docs_dir / "b.md").write_text("beta", encoding="utf-8")
    (docs_dir / "c.docx").write_text("gamma", encoding="utf-8")

    names = [path.name for path in list_document_files(docs_dir)]

    assert names == ["a.txt", "b.md"]


def test_extension_helpers():
    assert extension_of("NOTES.TXT") == ".txt"
    assert is_supported_file("guide.markdown")
    assert not is_supported_file("image.png")
