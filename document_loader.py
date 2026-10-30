"""
document_loader.py
==================
Reads TXT, Markdown and PDF files and returns clean text.

Beginner note (Telugu-English):
    User upload chesina file ni first manam "text" ga marchali.
    - .txt / .md  -> Python built-in open() chalu.
    - .pdf        -> pypdf library text ni teestundi (PDF is a binary format).
    Ee file lo errors kuda chala clear ga cheptunnam:
    empty file, wrong format, corrupted PDF, password-protected PDF etc.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import config


# ---------------------------------------------------------------------------
# Custom errors
# ---------------------------------------------------------------------------
class DocumentError(Exception):
    """Base class for every problem caused by an uploaded/loaded document."""


class UnsupportedFileTypeError(DocumentError):
    """The file extension is not in config.SUPPORTED_EXTENSIONS."""


class EmptyDocumentError(DocumentError):
    """The file exists but there is no readable text inside it."""


class DocumentReadError(DocumentError):
    """The file could not be opened/decoded/parsed."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Page:
    """One page of a document. TXT/MD files have a single page (number=None)."""

    number: int | None
    text: str

    def label(self) -> str:
        return f"page {self.number}" if self.number else "text"


@dataclass
class ExtractedDocument:
    """Result of loading one file."""

    filename: str
    path: Path
    pages: list[Page] = field(default_factory=list)
    encoding: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip())

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def extension_of(name: str | Path) -> str:
    """Lower-case file extension including the dot ('' when there is none)."""
    return Path(name).suffix.lower()


def is_supported_file(name: str | Path) -> bool:
    """True if this file type can be indexed."""
    return extension_of(name) in config.SUPPORTED_EXTENSIONS


def list_document_files(folder: str | Path) -> list[Path]:
    """All supported files directly inside `folder`, sorted by name (case-insensitive)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and is_supported_file(p.name)]
    return sorted(files, key=lambda p: p.name.lower())


def safe_upload_filename(filename: str) -> str:
    """
    Make an uploaded filename safe for Windows + Linux.

    Example: "..\\..\\My Notes (final).MD" -> "My Notes (final).md"
    """
    name = Path(filename.replace("\\", "/")).name  # drop any folder part
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    if not name:
        name = "uploaded_document.txt"
    suffix = Path(name).suffix.lower()
    stem = Path(name).stem[:120] or "document"
    return f"{stem}{suffix}"


def unique_destination(folder: str | Path, filename: str) -> Path:
    """
    Return a path that does not exist yet: report.txt -> report (1).txt -> ...
    """
    folder = Path(folder)
    safe_name = safe_upload_filename(filename)
    candidate = folder / safe_name
    if not candidate.exists():
        return candidate
    stem, suffix = Path(safe_name).stem, Path(safe_name).suffix
    counter = 1
    while True:
        candidate = folder / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _decode_bytes(raw: bytes) -> tuple[str, str]:
    """
    Try a few common encodings. Returns (text, encoding_used).
    Windows Notepad users often save files as cp1252, so we support that too.
    """
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise DocumentReadError(
        "Could not decode this text file. Please save it as UTF-8 and try again."
    )


# ---------------------------------------------------------------------------
# Format-specific readers
# ---------------------------------------------------------------------------
def _read_text_file(path: Path) -> ExtractedDocument:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DocumentReadError(f"Could not read '{path.name}': {exc}") from exc

    text, encoding = _decode_bytes(raw)
    if not text.strip():
        raise EmptyDocumentError(
            f"'{path.name}' is empty (0 readable words). Add some text and save it again."
        )
    return ExtractedDocument(
        filename=path.name,
        path=path,
        pages=[Page(number=None, text=text)],
        encoding=encoding,
    )


def _read_pdf_file(path: Path) -> ExtractedDocument:
    try:
        from pypdf import PdfReader  # lazy import: keeps startup fast
    except ImportError as exc:  # pragma: no cover - only when pypdf is missing
        raise DocumentReadError(
            "PDF support needs the 'pypdf' package. Run: pip install -r requirements.txt"
        ) from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pypdf raises many different error types
        raise DocumentReadError(
            f"'{path.name}' is not a readable PDF ({type(exc).__name__}). "
            "It may be corrupted or password-protected."
        ) from exc

    if getattr(reader, "is_encrypted", False):
        try:
            # Many PDFs are "encrypted" with an empty password; try that once.
            reader.decrypt("")
        except Exception as exc:
            raise DocumentReadError(
                f"'{path.name}' is password-protected. Remove the password and upload again."
            ) from exc

    pages: list[Page] = []
    warnings: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
            warnings.append(f"Page {index} could not be read and was skipped.")
        if page_text.strip():
            pages.append(Page(number=index, text=page_text))

    if not pages:
        raise EmptyDocumentError(
            f"No text found in '{path.name}'. Scanned/image-only PDFs need OCR first "
            "(try opening it and saving as a text PDF), or upload a TXT/MD version."
        )

    document = ExtractedDocument(filename=path.name, path=path, pages=pages, warnings=warnings)
    if document.word_count < 3:
        raise EmptyDocumentError(
            f"'{path.name}' has almost no readable text ({document.word_count} words)."
        )
    return document


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_document(path: str | Path) -> ExtractedDocument:
    """
    Load one supported document and return its text.

    Raises:
        UnsupportedFileTypeError: extension not supported
        EmptyDocumentError:       file has no readable text
        DocumentReadError:        file missing/corrupted/cannot be decoded
    """
    path = Path(path)

    if not path.exists():
        raise DocumentReadError(f"File not found: {path.name}")
    if not path.is_file():
        raise DocumentReadError(f"'{path.name}' is not a file.")

    suffix = extension_of(path)

    if suffix not in config.SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"'{path.name}' has an unsupported format '{suffix or 'no extension'}'. "
            f"Allowed formats: {config.human_supported_formats()}."
        )

    if suffix == ".pdf":
        return _read_pdf_file(path)
    return _read_text_file(path)


def describe_errors() -> str:
    """One-line reminder of the error types, used in the README/UI."""
    return "UnsupportedFileTypeError, EmptyDocumentError, DocumentReadError"


__all__ = [
    "Page",
    "ExtractedDocument",
    "DocumentError",
    "UnsupportedFileTypeError",
    "EmptyDocumentError",
    "DocumentReadError",
    "extension_of",
    "is_supported_file",
    "list_document_files",
    "safe_upload_filename",
    "unique_destination",
    "load_document",
    "describe_errors",
]
