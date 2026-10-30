"""
conftest.py
===========
Makes `import config`, `import indexer`, ... work when pytest is started from
any folder, and shares a few fixtures with all test files.

Beginner note (Telugu-English):
    Pytest ki project root path teliyali, appude manam rase modules (config.py,
    indexer.py) import avutayi. Ee file ee pani chestundi. Ikkada unna fixtures
    (sample_documents_folder, project_docs, ...) tests/ folder lo unna anni test
    files vaadukovachu.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config  # noqa: E402  (imported after sys.path is fixed)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sample_dir() -> Path:
    """The bundled sample dataset (TXT + MD + PDF)."""
    return config.SAMPLE_DOCUMENTS_DIR


@pytest.fixture()
def docs_dir(tmp_path: Path) -> Path:
    """An empty temporary 'documents' folder for each test."""
    folder = tmp_path / "documents"
    folder.mkdir()
    return folder


@pytest.fixture()
def index_dir(tmp_path: Path) -> Path:
    """An empty temporary 'index_store' folder for each test."""
    folder = tmp_path / "index_store"
    folder.mkdir()
    return folder


@pytest.fixture()
def sample_docs_dir(docs_dir: Path, sample_dir: Path) -> Path:
    """A documents folder pre-filled with the sample dataset (copied)."""
    for source in sample_dir.iterdir():
        if source.is_file() and source.suffix.lower() in config.SUPPORTED_EXTENSIONS:
            shutil.copy2(source, docs_dir / source.name)
    return docs_dir


@pytest.fixture()
def index_manager(docs_dir: Path, index_dir: Path):
    """IndexManager that writes only inside tmp_path (never the real folders)."""
    from indexer import IndexManager

    return IndexManager(documents_dir=docs_dir, index_dir=index_dir)


@pytest.fixture()
def sample_index_manager(sample_docs_dir: Path, index_dir: Path):
    """IndexManager already pointing at the sample dataset."""
    from indexer import IndexManager

    return IndexManager(documents_dir=sample_docs_dir, index_dir=index_dir)


class FakeUpload:
    """Small helper so tests can call save_uploaded_file() without Streamlit."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


@pytest.fixture()
def fake_upload():
    return FakeUpload
