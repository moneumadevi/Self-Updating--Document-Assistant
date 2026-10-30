"""
watcher.py
==========
Detects documents that were ADDED, EDITED or DELETED in the documents folder.

Two ways to use it:
    1. Inside the app: the Streamlit page polls this module every few seconds
       (auto-watch toggle) and rebuilds the index automatically.
    2. From the terminal, without the web page:
           python watcher.py            # watch forever, rebuild on change
           python watcher.py --once     # check once and rebuild if needed

We use polling (compare size + modified time) instead of the 'watchdog'
library: it needs zero extra dependencies and behaves the same on Windows.

Beginner note (Telugu-English):
    File system "events" kosam extra library vadakunda, prathi 2-3 seconds ki
    folder lo unna files size + last modified time chustam. Ee rendu
    marithe file marcharu ani artham. Deleted file = mundu unna list lo
    undi, ippudu ledu. Added file = ippudu undi, mundu ledu. Chala simple!
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import config
from document_loader import list_document_files
from indexer import DocumentIndex, IndexManager


@dataclass(frozen=True)
class FileSnapshot:
    name: str
    size: int
    mtime_ns: int


def scan_folder(folder: str | Path = config.DOCUMENTS_DIR) -> dict[str, FileSnapshot]:
    """Current snapshot of the watched folder: {name: FileSnapshot}."""
    snapshot: dict[str, FileSnapshot] = {}
    for path in list_document_files(folder):
        try:
            stat = path.stat()
        except OSError:
            continue
        snapshot[path.name] = FileSnapshot(
            name=path.name, size=stat.st_size, mtime_ns=stat.st_mtime_ns
        )
    return snapshot


@dataclass
class ChangeSet:
    """What changed between two snapshots."""

    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.removed)

    @property
    def total(self) -> int:
        return len(self.added) + len(self.modified) + len(self.removed)

    def summary(self) -> str:
        if not self.has_changes:
            return "No changes detected."
        parts: list[str] = []
        if self.added:
            parts.append(f"added: {', '.join(sorted(self.added, key=str.lower))}")
        if self.modified:
            parts.append(f"edited: {', '.join(sorted(self.modified, key=str.lower))}")
        if self.removed:
            parts.append(f"deleted: {', '.join(sorted(self.removed, key=str.lower))}")
        return " | ".join(parts)

    def lower_case_names(self) -> list[str]:
        return [name.lower() for name in self.added + self.modified + self.removed]


def diff_snapshots(
    before: dict[str, FileSnapshot], after: dict[str, FileSnapshot]
) -> ChangeSet:
    """Compare two snapshots and return added/modified/removed file names."""
    changes = ChangeSet()
    for name, new in after.items():
        old = before.get(name)
        if old is None:
            changes.added.append(name)
        elif (old.size, old.mtime_ns) != (new.size, new.mtime_ns):
            changes.modified.append(name)
    for name in before:
        if name not in after:
            changes.removed.append(name)
    changes.added.sort(key=str.lower)
    changes.modified.sort(key=str.lower)
    changes.removed.sort(key=str.lower)
    return changes


class FolderWatcher:
    """
    Keeps the last known snapshot in memory and reports changes on each poll.

    Usage:
        watcher = FolderWatcher("documents")
        changes = watcher.poll()          # first call -> everything is "added"
        if changes.has_changes:
            manager.get(force_rebuild=True)
    """

    def __init__(
        self,
        documents_dir: str | Path = config.DOCUMENTS_DIR,
        initial: dict[str, FileSnapshot] | None = None,
    ) -> None:
        self.documents_dir = Path(documents_dir)
        self._snapshot = initial if initial is not None else scan_folder(self.documents_dir)

    @property
    def snapshot(self) -> dict[str, FileSnapshot]:
        return dict(self._snapshot)

    def poll(self) -> ChangeSet:
        """Scan the folder now and return the difference from the last scan."""
        current = scan_folder(self.documents_dir)
        changes = diff_snapshots(self._snapshot, current)
        self._snapshot = current
        return changes

    def reset(self) -> None:
        self._snapshot = scan_folder(self.documents_dir)

    def compare_with_index(self, index: DocumentIndex | None) -> ChangeSet:
        """
        Compare the folder with what the SAVED index knows (works after restart).
        """
        if index is None:
            return ChangeSet()

        changes = ChangeSet()
        current = scan_folder(self.documents_dir)
        for name, snap in current.items():
            record = index.files.get(name)
            if record is None:
                changes.added.append(name)
            elif (record.size, record.mtime_ns) != (snap.size, snap.mtime_ns):
                changes.modified.append(name)
        for name in index.files:
            if name not in current:
                changes.removed.append(name)
        changes.added.sort(key=str.lower)
        changes.modified.sort(key=str.lower)
        changes.removed.sort(key=str.lower)
        return changes

    def signature(self) -> tuple:
        """Hashable folder signature - useful for Streamlit caching."""
        return tuple(sorted((name, snap.size, snap.mtime_ns) for name, snap in scan_folder(self.documents_dir).items()))


# ---------------------------------------------------------------------------
# Command line mode:  python watcher.py
# ---------------------------------------------------------------------------
def watch_forever(
    documents_dir: str | Path = config.DOCUMENTS_DIR,
    index_dir: str | Path = config.INDEX_DIR,
    interval: float = 2.0,
) -> None:
    """Rebuild the index whenever the folder changes. Ctrl+C to stop."""
    manager = IndexManager(documents_dir=documents_dir, index_dir=index_dir)
    index, report = manager.get()
    if report:
        print(f"[start] {report.summary()}")
    else:
        print(f"[start] Loaded saved index ({index.chunk_count} chunks).")
    print(f"Watching '{Path(documents_dir).resolve()}' every {interval}s. Press Ctrl+C to stop.")

    watcher = FolderWatcher(documents_dir)
    watcher.reset()
    try:
        while True:
            time.sleep(interval)
            changes = watcher.poll()
            if changes.has_changes:
                print(f"[change] {changes.summary()}")
                index, report = manager.get(force_rebuild=True)
                if report:
                    print(f"[rebuild] {report.summary()}")
    except KeyboardInterrupt:
        print("\n[stop] Watcher stopped.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Watch the documents folder and rebuild the TF-IDF index on changes."
    )
    parser.add_argument(
        "--documents",
        default=str(config.DOCUMENTS_DIR),
        help="folder to watch (default: ./documents)",
    )
    parser.add_argument(
        "--index",
        default=str(config.INDEX_DIR),
        help="folder where the index is stored (default: ./index_store)",
    )
    parser.add_argument(
        "--interval", type=float, default=2.0, help="polling interval in seconds"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="check once, rebuild if needed, then exit",
    )
    args = parser.parse_args(argv)

    manager = IndexManager(documents_dir=args.documents, index_dir=args.index)
    if args.once:
        index, report = manager.get()
        if report:
            print(report.summary())
        else:
            print(f"Index is up to date: {index.file_count} files, {index.chunk_count} chunks.")
        return 0

    watch_forever(args.documents, args.index, args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
