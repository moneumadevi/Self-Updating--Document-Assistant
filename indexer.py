"""
indexer.py
==========
Builds, saves and re-uses the TF-IDF search index.

What actually happens here (no fake "training" claims):
    * TfidfVectorizer is FIT on the text of your documents. Fitting = it learns
      the vocabulary and the IDF weights (rarer words get bigger weights).
    * transform() converts every chunk into a sparse number vector.
    * The vectors are stored in a matrix; searching later only needs cosine
      similarity between the question vector and these rows.
    * There is NO neural network and NO pre-trained model downloaded. TF-IDF is
      classic NLP maths, and it runs fully offline.

Caching (requirement 11):
    For every file we store a SHA-256 hash + size + modified time. On the next
    build, if a file's hash is unchanged we RE-USE its old chunks instead of
    reading/parsing it again. So adding one file does not re-process everything.

Beginner note (Telugu-English):
    Index = book kee last page lo unna "index" laantidi. Manam mundu
    documents ni chunks ga chesi, TF-IDF tho numbers ga marchi,
    index_store/index.pkl lo save chestam. Next time app open chesinappudu
    malli files chadavakunda ee saved index ne load chestam - chala fast.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import config
from document_loader import DocumentError, list_document_files, load_document
from text_processor import Chunk, chunk_document, normalise_term, tokenize


def _utc_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    """Hash of the file content - if it does not change, the file is unchanged."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 256), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class FileRecord:
    """What the index remembers about one file."""

    name: str
    sha256: str
    size: int
    mtime_ns: int
    chunk_count: int = 0
    page_count: int = 0
    status: str = "indexed"          # "indexed" or "error"
    error: str | None = None
    encoding: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "chunk_count": self.chunk_count,
            "page_count": self.page_count,
            "status": self.status,
            "error": self.error,
            "encoding": self.encoding,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileRecord":
        return cls(
            name=str(data["name"]),
            sha256=str(data["sha256"]),
            size=int(data["size"]),
            mtime_ns=int(data["mtime_ns"]),
            chunk_count=int(data.get("chunk_count", 0)),
            page_count=int(data.get("page_count", 0)),
            status=str(data.get("status", "indexed")),
            error=data.get("error"),
            encoding=data.get("encoding"),
        )


@dataclass
class DocumentIndex:
    """The in-memory search index: vectorizer + matrix + chunks + file records."""

    vectorizer: Any | None = None                 # sklearn TfidfVectorizer
    matrix: Any | None = None                     # scipy sparse matrix
    chunks: list[Chunk] = field(default_factory=list)
    files: dict[str, FileRecord] = field(default_factory=dict)
    built_at: str = ""
    version: int = config.INDEX_VERSION
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # -- convenience properties -------------------------------------------
    @property
    def is_empty(self) -> bool:
        return self.vectorizer is None or self.matrix is None or not self.chunks

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def vocabulary_size(self) -> int:
        if self.vectorizer is None:
            return 0
        return len(getattr(self.vectorizer, "vocabulary_", {}) or {})

    @property
    def total_words(self) -> int:
        return sum(chunk.word_count for chunk in self.chunks)

    def file_names(self) -> list[str]:
        return sorted(self.files.keys(), key=str.lower)

    def chunks_for(self, filename: str) -> list[Chunk]:
        return [chunk for chunk in self.chunks if chunk.filename == filename]

    def to_pickle_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "built_at": self.built_at,
            "vectorizer": self.vectorizer,
            "matrix": self.matrix,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "files": {name: record.to_dict() for name, record in self.files.items()},
            "stats": self.stats,
            "warnings": self.warnings,
        }

    @classmethod
    def from_pickle_dict(cls, data: dict[str, Any]) -> "DocumentIndex":
        return cls(
            vectorizer=data.get("vectorizer"),
            matrix=data.get("matrix"),
            chunks=[Chunk.from_dict(item) for item in data.get("chunks", [])],
            files={
                name: FileRecord.from_dict(record)
                for name, record in (data.get("files") or {}).items()
            },
            built_at=str(data.get("built_at", "")),
            version=int(data.get("version", 0)),
            stats=dict(data.get("stats") or {}),
            warnings=list(data.get("warnings") or []),
        )


@dataclass
class BuildReport:
    """Everything the UI needs to tell the user what just happened."""

    rebuilt: bool = False
    reason: str = ""
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    re_extracted: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    chunk_count: int = 0
    file_count: int = 0
    vocabulary_size: int = 0
    duration_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.removed)

    def summary(self) -> str:
        if not self.rebuilt:
            return self.reason or "Index already up to date."
        bits = [f"{self.file_count} file(s), {self.chunk_count} chunk(s)"]
        if self.added:
            bits.append(f"added: {', '.join(self.added)}")
        if self.modified:
            bits.append(f"edited: {', '.join(self.modified)}")
        if self.removed:
            bits.append(f"deleted: {', '.join(self.removed)}")
        if self.unchanged:
            bits.append(f"re-used without re-reading: {len(self.unchanged)}")
        if self.errors:
            bits.append(f"failed: {', '.join(self.errors)}")
        return " | ".join(bits)


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------
def _stop_words_normalised() -> set[str]:
    """
    sklearn's English stop-word list, normalised with the same suffix rule as
    the documents. This keeps "why" and "whys" from becoming two vocabulary
    entries, and avoids sklearn's "stop_words inconsistent with tokenizer" warning.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return {normalise_term(word) for word in ENGLISH_STOP_WORDS}


# Module-level (not a lambda) so the fitted vectorizer can be pickled and loaded
# again in a later run without breaking.
STOP_WORDS_NORMALISED = frozenset(_stop_words_normalised())


def tfidf_tokenizer(text: str) -> list[str]:
    """Tokenizer used by the vectorizer: normalise suffixes, drop stop words."""
    return [term for term in tokenize(text) if term not in STOP_WORDS_NORMALISED]


def _build_vectorizer():
    """Create the TF-IDF vectorizer with our project settings."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(
        lowercase=False,          # tokenize() already lowercases
        stop_words=None,          # tfidf_tokenizer() already removed stop words
        tokenizer=tfidf_tokenizer,
        preprocessor=None,
        token_pattern=None,       # tokenizer above replaces the default pattern
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_features=config.TFIDF_MAX_FEATURES,
        sublinear_tf=config.TFIDF_SUBLINEAR_TF,
        strip_accents="unicode",
        dtype=np.float64,
    )


def fit_tfidf(chunks: list[Chunk]):
    """
    Fit the vectorizer on all chunk texts and return (vectorizer, matrix).

    Two safety nets, because an empty vocabulary is a real possibility:
      1. only stop-words / numbers  -> retry without the stop-word filter
      2. still empty                -> return (None, None) and the UI explains it
    """
    if not chunks:
        return None, None

    texts = [chunk.text for chunk in chunks]

    vectorizer = _build_vectorizer()
    try:
        matrix = vectorizer.fit_transform(texts)
        if matrix.shape[1] > 0:
            return vectorizer, matrix
    except ValueError:
        pass  # "empty vocabulary" - fall through to the fallback

    from sklearn.feature_extraction.text import TfidfVectorizer

    fallback = TfidfVectorizer(
        lowercase=False,
        stop_words=None,
        tokenizer=tokenize,
        preprocessor=None,
        token_pattern=None,
        ngram_range=(1, 1),
        min_df=1,
        strip_accents="unicode",
        dtype=np.float64,
    )
    try:
        matrix = fallback.fit_transform(texts)
        if matrix.shape[1] > 0:
            return fallback, matrix
    except ValueError:
        pass
    return None, None


def build_index(
    documents_dir: str | Path = config.DOCUMENTS_DIR,
    previous: DocumentIndex | None = None,
) -> tuple[DocumentIndex, BuildReport]:
    """
    Scan `documents_dir`, re-use cached chunks where possible, fit TF-IDF.

    Never raises for a single bad file: the error is recorded in the report and
    the rest of the documents are still indexed.
    """
    started = time.perf_counter()
    documents_dir = Path(documents_dir)
    report = BuildReport(rebuilt=True, reason="Index rebuilt from the documents folder.")

    index = DocumentIndex(built_at=_utc_now_iso(), version=config.INDEX_VERSION)
    old_files = previous.files if previous else {}
    old_chunks: dict[str, list[Chunk]] = {}
    if previous:
        for chunk in previous.chunks:
            old_chunks.setdefault(chunk.filename, []).append(chunk)

    files = list_document_files(documents_dir)
    current_names = {path.name for path in files}

    # Deleted files = they were in the old index but are not in the folder now.
    report.removed = sorted(
        (name for name in old_files if name not in current_names), key=str.lower
    )

    new_chunks: list[Chunk] = []

    for path in files:
        name = path.name
        try:
            stat = path.stat()
            size = stat.st_size
            mtime_ns = stat.st_mtime_ns
            digest = file_sha256(path)
        except OSError as exc:
            report.errors[name] = f"Could not read file: {exc}"
            index.files[name] = FileRecord(
                name=name, sha256="", size=0, mtime_ns=0, status="error", error=str(exc)
            )
            continue

        old_record = old_files.get(name)
        old_file_chunks = old_chunks.get(name, [])

        if (
            old_record is not None
            and old_record.sha256 == digest
            and old_record.status == "indexed"
            and old_file_chunks
            and old_record.chunk_count == len(old_file_chunks)
        ):
            # UNCHANGED -> reuse the chunks we already extracted (no re-parsing).
            new_chunks.extend(old_file_chunks)
            index.files[name] = FileRecord(
                name=name,
                sha256=digest,
                size=size,
                mtime_ns=mtime_ns,
                chunk_count=len(old_file_chunks),
                page_count=old_record.page_count,
                status="indexed",
                encoding=old_record.encoding,
            )
            report.unchanged.append(name)
            continue

        # NEW or CHANGED -> read + clean + chunk the file now.
        report.re_extracted.append(name)
        try:
            document = load_document(path)
            chunks = chunk_document(document)
        except DocumentError as exc:
            message = str(exc)
            report.errors[name] = message
            index.files[name] = FileRecord(
                name=name,
                sha256=digest,
                size=size,
                mtime_ns=mtime_ns,
                status="error",
                error=message,
            )
            # Keep the old chunks searchable? No: a broken edited file must not
            # serve stale content. We drop it and show the error in the UI.
            continue
        except Exception as exc:  # unexpected bug in a parser
            message = f"Unexpected error while reading: {exc}"
            report.errors[name] = message
            index.files[name] = FileRecord(
                name=name,
                sha256=digest,
                size=size,
                mtime_ns=mtime_ns,
                status="error",
                error=message,
            )
            continue

        if not chunks:
            message = "No usable text found after cleaning (too short?)."
            report.errors[name] = message
            index.files[name] = FileRecord(
                name=name,
                sha256=digest,
                size=size,
                mtime_ns=mtime_ns,
                page_count=len(document.pages),
                status="error",
                error=message,
                encoding=document.encoding,
            )
            continue

        new_chunks.extend(chunks)
        index.files[name] = FileRecord(
            name=name,
            sha256=digest,
            size=size,
            mtime_ns=mtime_ns,
            chunk_count=len(chunks),
            page_count=len(document.pages),
            status="indexed",
            encoding=document.encoding,
        )
        report.warnings.extend(f"{name}: {w}" for w in document.warnings)

        if old_record is None:
            report.added.append(name)
        else:
            report.modified.append(name)

    # Fit TF-IDF once on the final chunk list.
    index.chunks = new_chunks
    index.vectorizer, index.matrix = fit_tfidf(new_chunks)

    if new_chunks and index.vectorizer is None:
        report.warnings.append(
            "Documents contained no searchable words (only numbers/stop-words), "
            "so the search index is empty."
        )

    # Sort chunk lists per file so chunk numbers read top-to-bottom.
    index.chunks.sort(key=lambda c: (c.filename.lower(), c.index))

    duration = time.perf_counter() - started
    stats = {
        "chunk_size_words": config.CHUNK_SIZE_WORDS,
        "chunk_overlap_words": config.CHUNK_OVERLAP_WORDS,
        "tfidf_ngram_range": list(config.TFIDF_NGRAM_RANGE),
        "files": index.file_count,
        "chunks": index.chunk_count,
        "vocabulary": index.vocabulary_size,
        "duration_seconds": round(duration, 3),
    }
    index.stats = stats
    index.warnings = report.warnings

    report.rebuilt = True
    report.chunk_count = index.chunk_count
    report.file_count = index.file_count
    report.vocabulary_size = index.vocabulary_size
    report.duration_seconds = round(duration, 3)
    return index, report


# ---------------------------------------------------------------------------
# Saving + loading the index on disk
# ---------------------------------------------------------------------------
class IndexManager:
    """Handles the whole lifecycle: load from disk, rebuild if stale, save."""

    def __init__(
        self,
        documents_dir: str | Path = config.DOCUMENTS_DIR,
        index_dir: str | Path = config.INDEX_DIR,
    ) -> None:
        self.documents_dir = Path(documents_dir)
        self.index_dir = Path(index_dir)

    # -- paths -------------------------------------------------------------
    @property
    def binary_path(self) -> Path:
        return self.index_dir / config.INDEX_BINARY_NAME

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / config.INDEX_MANIFEST_NAME

    # -- save --------------------------------------------------------------
    def save(self, index: DocumentIndex) -> None:
        """Write index.pkl (vectorizer/matrix/chunks) + manifest.json (summary)."""
        self.index_dir.mkdir(parents=True, exist_ok=True)

        payload = index.to_pickle_dict()
        tmp_binary = self.binary_path.with_suffix(".pkl.tmp")
        with tmp_binary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_binary, self.binary_path)  # atomic: no half-written index

        manifest = {
            "app": "Self Updating Documentation Assistant",
            "index_version": index.version,
            "built_at": index.built_at,
            "settings": {
                "chunk_size_words": config.CHUNK_SIZE_WORDS,
                "chunk_overlap_words": config.CHUNK_OVERLAP_WORDS,
                "tfidf": "TF-IDF (scikit-learn TfidfVectorizer, ngram 1-2)",
                "similarity": "cosine similarity",
            },
            "totals": {
                "files": index.file_count,
                "chunks": index.chunk_count,
                "vocabulary_terms": index.vocabulary_size,
                "words": index.total_words,
                "build_seconds": index.stats.get("duration_seconds"),
            },
            "files": [record.to_dict() for record in index.files.values()],
            "warnings": index.warnings,
        }
        tmp_manifest = self.manifest_path.with_suffix(".json.tmp")
        tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp_manifest, self.manifest_path)

    # -- load --------------------------------------------------------------
    def load(self) -> DocumentIndex | None:
        """
        Load the saved index. Returns None when there is nothing usable:
        missing file, old version, or corrupted pickle (never crashes the app).
        """
        if not self.binary_path.exists():
            return None
        try:
            with self.binary_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None
        if int(payload.get("version", 0)) != config.INDEX_VERSION:
            return None  # settings changed -> rebuild

        try:
            return DocumentIndex.from_pickle_dict(payload)
        except Exception:
            return None

    # -- change detection (cheap, no file content reading) -----------------
    def folder_snapshot(self) -> dict[str, tuple[int, int]]:
        """{filename: (size, mtime_ns)} for every supported file in the folder."""
        snapshot: dict[str, tuple[int, int]] = {}
        for path in list_document_files(self.documents_dir):
            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[path.name] = (stat.st_size, stat.st_mtime_ns)
        return snapshot

    def needs_rebuild(self, index: DocumentIndex | None) -> tuple[bool, str]:
        """
        Decide if the saved index is stale compared to the folder right now.

        We compare size + modified-time (fast). The full SHA-256 check happens
        inside build_index, so unchanged files are never parsed twice.
        """
        if index is None:
            return True, "No saved index found - building it for the first time."

        snapshot = self.folder_snapshot()
        indexed = index.files

        for name, (size, mtime_ns) in snapshot.items():
            record = indexed.get(name)
            if record is None:
                return True, f"New document detected: {name}"
            if record.size != size or record.mtime_ns != mtime_ns:
                return True, f"Document changed: {name}"

        for name in indexed:
            if name not in snapshot:
                return True, f"Document deleted: {name}"

        return False, "Index is up to date."

    # -- main entry point --------------------------------------------------
    def get(
        self, force_rebuild: bool = False
    ) -> tuple[DocumentIndex | None, BuildReport | None]:
        """
        Return the ready-to-use index.

        * force_rebuild=True -> always rebuild
        * otherwise: load the saved index and rebuild only when the folder changed
        Returns (index, report). report is None when nothing had to be rebuilt.
        """
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        previous = self.load()

        stale, reason = self.needs_rebuild(previous)
        if not force_rebuild and not stale:
            return previous, None

        index, report = build_index(self.documents_dir, previous=previous)
        report.reason = reason if not force_rebuild else "Manual rebuild requested."
        try:
            self.save(index)
        except OSError as exc:
            report.warnings.append(f"Index could not be saved to disk: {exc}")
        return index, report

    def build_fresh(self) -> tuple[DocumentIndex, BuildReport]:
        """Ignore caches completely and rebuild from zero (used by tests/Reset)."""
        index, report = build_index(self.documents_dir, previous=None)
        report.reason = "Fresh rebuild (cache ignored)."
        self.save(index)
        return index, report

    def delete_saved_index(self) -> None:
        for path in (self.binary_path, self.manifest_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def read_manifest(self) -> dict[str, Any] | None:
        """Readable JSON summary (handy for the UI 'Index status' panel)."""
        if not self.manifest_path.exists():
            return None
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None


def index_summary_lines(index: DocumentIndex) -> Iterable[str]:
    """Small helper for debug printing in tests."""
    yield f"files={index.file_count}"
    yield f"chunks={index.chunk_count}"
    yield f"vocabulary={index.vocabulary_size}"


__all__ = [
    "STOP_WORDS_NORMALISED",
    "tfidf_tokenizer",
    "FileRecord",
    "DocumentIndex",
    "BuildReport",
    "IndexManager",
    "build_index",
    "fit_tfidf",
    "file_sha256",
    "index_summary_lines",
]
