"""
Tests for watcher.py - detecting added / edited / deleted documents.

Purpose: requirement 9 says the app must notice folder changes. These tests
simulate each change on disk and check the ChangeSet, plus the terminal
`python watcher.py --once` entry point.
"""

from __future__ import annotations

import os

from indexer import IndexManager
from watcher import ChangeSet, FileSnapshot, FolderWatcher, diff_snapshots, main, scan_folder


def _snapshot(name: str, size: int = 10, mtime: int = 1) -> FileSnapshot:
    return FileSnapshot(name=name, size=size, mtime_ns=mtime)


def test_diff_detects_added_modified_removed():
    before = {
        "a.txt": _snapshot("a.txt"),
        "b.md": _snapshot("b.md"),
        "gone.pdf": _snapshot("gone.pdf"),
    }
    after = {
        "a.txt": _snapshot("a.txt"),          # unchanged
        "b.md": _snapshot("b.md", size=99),   # edited
        "c.txt": _snapshot("c.txt"),          # added
    }

    changes = diff_snapshots(before, after)

    assert changes.added == ["c.txt"]
    assert changes.modified == ["b.md"]
    assert changes.removed == ["gone.pdf"]
    assert changes.has_changes is True


def test_diff_with_no_changes():
    before = {"a.txt": _snapshot("a.txt")}

    changes = diff_snapshots(before, dict(before))

    assert changes.has_changes is False
    assert changes.total == 0
    assert changes.summary() == "No changes detected."


def test_change_summary_lists_names():
    changes = ChangeSet(added=["new.md"], modified=["old.txt"], removed=["gone.pdf"])

    summary = changes.summary()

    assert "added: new.md" in summary
    assert "edited: old.txt" in summary
    assert "deleted: gone.pdf" in summary


def test_scan_folder_ignores_unsupported_files(docs_dir):
    (docs_dir / "a.txt").write_text("hello", encoding="utf-8")
    (docs_dir / "b.pdf").write_bytes(b"%PDF-1.4")
    (docs_dir / "c.jpg").write_bytes(b"image")

    snapshot = scan_folder(docs_dir)

    assert set(snapshot) == {"a.txt", "b.pdf"}


def test_folder_watcher_construction_is_the_baseline(docs_dir):
    (docs_dir / "a.txt").write_text("first version", encoding="utf-8")
    watcher = FolderWatcher(docs_dir)

    assert watcher.poll().has_changes is False


def test_folder_watcher_detects_add_edit_delete(docs_dir):
    watcher = FolderWatcher(docs_dir)

    (docs_dir / "a.txt").write_text("first version", encoding="utf-8")
    assert watcher.poll().added == ["a.txt"]

    (docs_dir / "a.txt").write_text("second version with more words", encoding="utf-8")
    assert watcher.poll().modified == ["a.txt"]

    (docs_dir / "a.txt").unlink()
    assert watcher.poll().removed == ["a.txt"]


def test_folder_watcher_ignores_untouched_file(docs_dir):
    path = docs_dir / "stable.md"
    path.write_text("# Stable\n\nContent stays the same.", encoding="utf-8")
    watcher = FolderWatcher(docs_dir)

    assert watcher.poll().has_changes is False
    assert watcher.poll().has_changes is False


def test_compare_with_index_finds_every_change(sample_index_manager, sample_docs_dir):
    index, _ = sample_index_manager.get()
    watcher = FolderWatcher(sample_docs_dir)

    assert watcher.compare_with_index(index).has_changes is False

    (sample_docs_dir / "python_basics.md").write_text("# Python\n\nChanged text.", encoding="utf-8")
    (sample_docs_dir / "brand_new.txt").write_text("A new document about testing.", encoding="utf-8")
    (sample_docs_dir / "git_cheatsheet.md").unlink()

    changes = watcher.compare_with_index(index)

    assert "brand_new.txt" in changes.added
    assert "python_basics.md" in changes.modified
    assert "git_cheatsheet.md" in changes.removed


def test_signature_changes_when_file_changes(docs_dir):
    path = docs_dir / "a.txt"
    path.write_text("one", encoding="utf-8")
    watcher = FolderWatcher(docs_dir)
    first = watcher.signature()

    path.write_text("two", encoding="utf-8")
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    assert watcher.signature() != first


def test_cli_once_returns_zero(tmp_path, capsys):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "note.txt").write_text("The watcher rebuilds the index automatically.", encoding="utf-8")
    index_dir = tmp_path / "index_store"

    exit_code = main(["--once", "--documents", str(docs), "--index", str(index_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "file" in captured.out.lower()
    assert (index_dir / "index.pkl").exists()


def test_cli_once_when_up_to_date(tmp_path, capsys):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "note.txt").write_text("Watcher content about polling.", encoding="utf-8")
    index_dir = tmp_path / "index_store"
    IndexManager(documents_dir=docs, index_dir=index_dir).get()
    capsys.readouterr()

    assert main(["--once", "--documents", str(docs), "--index", str(index_dir)]) == 0
    assert "up to date" in capsys.readouterr().out.lower()
