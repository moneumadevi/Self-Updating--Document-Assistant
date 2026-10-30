"""
Tests for indexer.py - TF-IDF building, caching, persistence and change tracking.

Purpose: prove that (a) the vectorizer is really fit on our documents,
(b) unchanged files are not processed again, (c) the saved index can be loaded
back, and (d) a broken file never stops the rest from being indexed.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pytest

import config
from indexer import BuildReport, IndexManager, build_index, file_sha256, fit_tfidf
from text_processor import Chunk


def test_build_index_fits_tfidf_on_sample_documents(sample_docs_dir):
    index, report = build_index(sample_docs_dir)

    assert report.rebuilt is True
    assert index.file_count >= 4
    assert index.chunk_count > 0
    # The vectorizer was really FIT: it learned a vocabulary and IDF weights.
    assert index.vectorizer is not None
    assert len(index.vectorizer.vocabulary_) > 50
    assert len(index.vectorizer.idf_) == len(index.vectorizer.vocabulary_)
    # matrix has one row per chunk and one column per vocabulary term
    assert index.matrix.shape == (index.chunk_count, index.vocabulary_size)
    assert report.vocabulary_size == index.vocabulary_size
    # sample data includes a PDF, a TXT and Markdown files
    assert any(name.endswith(".pdf") for name in index.file_names())
    assert any(name.endswith(".txt") for name in index.file_names())
    assert any(name.endswith(".md") for name in index.file_names())


def test_every_chunk_has_a_source_file(sample_docs_dir):
    index, _ = build_index(sample_docs_dir)

    for chunk in index.chunks:
        assert chunk.filename in index.files
        assert chunk.text.strip()
        assert chunk.word_count <= config.CHUNK_SIZE_WORDS


def test_second_build_reuses_cache_without_reading_files(sample_docs_dir):
    first, _ = build_index(sample_docs_dir)
    second, report = build_index(sample_docs_dir, previous=first)

    assert report.unchanged, "unchanged files should be reused from the cache"
    assert report.re_extracted == [], "unchanged files must not be parsed again"
    assert second.chunk_count == first.chunk_count
    assert (second.matrix != first.matrix).nnz == 0


def test_edited_file_is_reprocessed_and_old_chunks_dropped(sample_docs_dir):
    first, _ = build_index(sample_docs_dir)
    target = sample_docs_dir / "python_basics.md"
    target.write_text(
        "# Python\n\nPython was created by Guido van Rossum in the Netherlands.",
        encoding="utf-8",
    )

    second, report = build_index(sample_docs_dir, previous=first)

    assert report.modified == ["python_basics.md"]
    assert report.re_extracted == ["python_basics.md"]
    texts = " ".join(chunk.text for chunk in second.chunks_for("python_basics.md"))
    assert "Netherlands" in texts
    assert "virtual environment" not in texts.lower()


def test_deleted_file_is_removed_from_index(sample_docs_dir):
    first, _ = build_index(sample_docs_dir)
    (sample_docs_dir / "git_cheatsheet.md").unlink()

    second, report = build_index(sample_docs_dir, previous=first)

    assert report.removed == ["git_cheatsheet.md"]
    assert "git_cheatsheet.md" not in second.files
    assert second.chunks_for("git_cheatsheet.md") == []


def test_broken_file_is_reported_but_others_still_indexed(sample_docs_dir):
    (sample_docs_dir / "empty_upload.txt").write_text("", encoding="utf-8")

    index, report = build_index(sample_docs_dir)

    assert "empty_upload.txt" in report.errors
    assert index.files["empty_upload.txt"].status == "error"
    # The healthy files are still searchable.
    assert index.chunk_count > 0
    assert "python_basics.md" in index.files


def test_unsupported_files_are_ignored(docs_dir):
    (docs_dir / "readme.docx").write_bytes(b"not supported")
    (docs_dir / "good.txt").write_text(
        "Indexing works with plain text documents.", encoding="utf-8"
    )

    index, report = build_index(docs_dir)

    assert "readme.docx" not in index.files
    assert "good.txt" in index.files
    assert report.file_count == 1


def test_empty_folder_gives_empty_index(docs_dir):
    index, report = build_index(docs_dir)

    assert index.is_empty
    assert index.chunk_count == 0
    assert report.file_count == 0


def test_fit_tfidf_handles_only_stopwords():
    chunks = [Chunk("x::c0", "x.txt", None, 0, "the and is of to", 5)]

    vectorizer, matrix = fit_tfidf(chunks)

    # Fallback removes the stop-word filter, so real words get a vocabulary.
    assert vectorizer is not None
    assert matrix is not None
    assert matrix.shape[1] > 0


def test_index_is_saved_and_loaded(sample_index_manager):
    index, report = sample_index_manager.get()
    assert report is not None and report.rebuilt

    assert sample_index_manager.binary_path.exists()
    assert sample_index_manager.manifest_path.exists()

    loaded = sample_index_manager.load()
    assert loaded is not None
    assert loaded.chunk_count == index.chunk_count
    assert loaded.vocabulary_size == index.vocabulary_size
    assert loaded.file_names() == index.file_names()

    # A loaded index is immediately searchable.
    from qa_engine import answer_question

    answer = answer_question(loaded, "What is cosine similarity?")
    assert answer.status == "answered"


def test_unchanged_folder_does_not_rebuild(sample_index_manager):
    index, _ = sample_index_manager.get()

    again, report = sample_index_manager.get()

    assert report is None, "no rebuild should happen when nothing changed"
    assert again is not None
    assert again.chunk_count == index.chunk_count


def test_touching_a_file_triggers_rebuild_but_reuses_chunks(
    sample_index_manager, sample_docs_dir
):
    index, _ = sample_index_manager.get()
    target = sample_docs_dir / "streamlit_guide.md"
    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))

    stale, reason = sample_index_manager.needs_rebuild(index)
    assert stale is True
    assert "streamlit_guide.md" in reason

    _, report = sample_index_manager.get()
    assert report is not None and report.rebuilt
    # mtime changed but content is identical -> no parsing needed
    assert report.re_extracted == []
    assert "streamlit_guide.md" in report.unchanged


def test_new_file_triggers_rebuild(sample_index_manager, sample_docs_dir):
    sample_index_manager.get()
    (sample_docs_dir / "new_topic.md").write_text(
        "# Docker\n\nDocker packages an application with its dependencies into a container.",
        encoding="utf-8",
    )

    index, report = sample_index_manager.get()

    assert report is not None
    assert "new_topic.md" in report.added
    assert "new_topic.md" in index.files


def test_deleted_file_triggers_rebuild(sample_index_manager, sample_docs_dir):
    sample_index_manager.get()
    (sample_docs_dir / "git_cheatsheet.md").unlink()

    _, report = sample_index_manager.get()

    assert report is not None
    assert "git_cheatsheet.md" in report.removed


def test_force_rebuild(sample_index_manager):
    sample_index_manager.get()

    _, report = sample_index_manager.get(force_rebuild=True)

    assert report is not None
    assert report.rebuilt
    assert report.reason == "Manual rebuild requested."


def test_fresh_rebuild_ignores_cache(sample_index_manager):
    sample_index_manager.get()

    _, report = sample_index_manager.build_fresh()

    assert report.re_extracted, "fresh rebuild must parse every file again"


def test_corrupted_index_file_returns_none(sample_index_manager):
    sample_index_manager.get()
    sample_index_manager.binary_path.write_bytes(b"this is not a pickle")

    assert sample_index_manager.load() is None


def test_old_index_version_is_rejected(sample_index_manager):
    sample_index_manager.get()
    payload = pickle.loads(sample_index_manager.binary_path.read_bytes())
    payload["version"] = config.INDEX_VERSION - 1
    sample_index_manager.binary_path.write_bytes(pickle.dumps(payload))

    assert sample_index_manager.load() is None


def test_manifest_is_human_readable(sample_index_manager):
    sample_index_manager.get()

    manifest = sample_index_manager.read_manifest()

    assert manifest is not None
    assert manifest["totals"]["chunks"] > 0
    assert manifest["settings"]["tfidf"].startswith("TF-IDF")
    assert isinstance(manifest["files"], list)


def test_delete_saved_index(sample_index_manager):
    sample_index_manager.get()

    sample_index_manager.delete_saved_index()

    assert not sample_index_manager.binary_path.exists()
    assert not sample_index_manager.manifest_path.exists()


def test_file_hash_changes_with_content(docs_dir):
    path = docs_dir / "hash.txt"
    path.write_text("first", encoding="utf-8")
    first_hash = file_sha256(path)
    path.write_text("second", encoding="utf-8")

    assert file_sha256(path) != first_hash


def test_build_report_summary_text():
    report = BuildReport(rebuilt=True, added=["a.txt"], chunk_count=4, file_count=1)

    assert "a.txt" in report.summary()
    assert report.has_changes is True


@pytest.mark.parametrize("manager_attr", ["documents_dir", "index_dir"])
def test_manager_paths_are_paths(sample_index_manager, manager_attr):
    assert isinstance(getattr(sample_index_manager, manager_attr), Path)


def test_index_manager_creates_missing_documents_folder(tmp_path):
    manager = IndexManager(
        documents_dir=tmp_path / "missing" / "documents", index_dir=tmp_path / "index"
    )

    index, report = manager.get()

    assert manager.documents_dir.is_dir()
    assert index is not None
    assert report is not None


def test_vectorizer_stems_words_so_install_matches_installing(sample_docs_dir):
    index, _ = build_index(sample_docs_dir)
    vocabulary = index.vectorizer.vocabulary_

    assert "install" in vocabulary
    assert "installing" not in vocabulary
