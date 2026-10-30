"""
Tests for app.py - the Streamlit interface.

Purpose: check the page really starts, the example-question flow really produces
an answer, and the upload validation rejects bad files with a clear message.
We use Streamlit's official AppTest helper, so no browser is needed.
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = BASE_DIR / "sample_documents"


@pytest.fixture()
def app_env(tmp_path, monkeypatch):
    """Point the app at temporary folders and load its module."""
    docs = tmp_path / "documents"
    docs.mkdir()
    shutil.copy2(SAMPLE_DIR / "tfidf_and_retrieval.txt", docs / "tfidf_and_retrieval.txt")
    monkeypatch.setenv("DOCS_ASSISTANT_DOCUMENTS", str(docs))
    monkeypatch.setenv("DOCS_ASSISTANT_INDEX", str(tmp_path / "index_store"))

    import app as app_module

    return importlib.reload(app_module), docs


# ---------------------------------------------------------------------------
# Upload validation (pure Python, no browser)
# ---------------------------------------------------------------------------
def test_valid_upload_is_saved(app_env, fake_upload):
    app_module, docs = app_env

    status, message = app_module.save_uploaded_file(
        fake_upload("My Notes.MD", b"# Notes\n\nStreamlit makes dashboards easy.")
    )

    assert status == "ok"
    assert (docs / "My Notes.md").exists()
    assert "My Notes.md" in message


def test_unsupported_upload_is_rejected(app_env, fake_upload):
    app_module, docs = app_env

    status, message = app_module.save_uploaded_file(fake_upload("photo.png", b"\x89PNG"))

    assert status == "error"
    assert "unsupported format" in message.lower()
    assert list(docs.glob("*.png")) == []


def test_empty_upload_is_rejected(app_env, fake_upload):
    app_module, docs = app_env

    status, message = app_module.save_uploaded_file(fake_upload("empty.txt", b""))

    assert status == "error"
    assert "empty" in message.lower()
    assert not (docs / "empty.txt").exists()


def test_broken_pdf_upload_is_rejected_and_removed(app_env, fake_upload):
    app_module, docs = app_env

    status, message = app_module.save_uploaded_file(fake_upload("broken.pdf", b"%PDF-1.4 nope"))

    assert status == "error"
    assert not (docs / "broken.pdf").exists()


def test_duplicate_upload_gets_a_new_name(app_env, fake_upload):
    app_module, docs = app_env

    status, _ = app_module.save_uploaded_file(fake_upload("tfidf_and_retrieval.txt", b"new content"))

    assert status == "ok"
    assert (docs / "tfidf_and_retrieval (1).txt").exists()


def test_copy_sample_documents_does_not_overwrite(app_env):
    app_module, docs = app_env

    copied, skipped = app_module.copy_sample_documents()

    assert "python_basics.md" in copied
    assert "tfidf_and_retrieval.txt" in skipped
    assert (docs / "project_faq.pdf").exists()


# ---------------------------------------------------------------------------
# The real Streamlit page
# ---------------------------------------------------------------------------
def test_index_is_reused_on_second_page_load(app_env):
    from streamlit.testing.v1 import AppTest

    first = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=120)
    first.run()
    built_at = first.session_state["index"].built_at

    second = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=120)
    second.run()

    assert not second.exception
    assert second.session_state["index"].built_at == built_at
    assert second.session_state["index_loaded_from_disk"] is True


def test_page_starts_and_lists_documents(app_env):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=120)
    at.run()

    assert not at.exception
    headings = " ".join(element.value for element in at.markdown)
    assert "Self Updating Documentation Assistant" in headings
    assert at.session_state["index"].file_count == 1
    # the index was saved to disk, ready for the next start
    assert at.session_state["index"].chunk_count > 0


def test_example_question_produces_an_answer_with_source(app_env):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=120)
    at.run()

    at.button(key="example_What is TF-IDF?").click().run()

    assert not at.exception
    answer = at.session_state["last_answer"]
    assert answer.status == "answered"
    assert answer.source == "tfidf_and_retrieval.txt"
    assert answer.hits[0].score > 0

    page_text = " ".join(element.value for element in at.markdown)
    assert "Answer from your documents" in page_text


def test_unrelated_question_shows_no_answer(app_env):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=120)
    at.run()

    # Widgets inside a form are sent to the backend only when the form is
    # submitted, so set the value AND click in the same run.
    at.text_input(key="question_input").set_value("best recipe for chocolate biryani")
    submit_buttons = [button for button in at.button if button.label == "Get answer"]
    assert submit_buttons, "the ask form should have a submit button"
    submit_buttons[0].click().run()

    assert not at.exception
    answer = at.session_state["last_answer"]
    assert answer.status == "no_answer"
    assert "none of the words" in answer.message.lower()
    assert at.session_state["history"][-1]["status"] == "no_answer"
