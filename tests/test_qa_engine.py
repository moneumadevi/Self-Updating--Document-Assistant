"""
Tests for qa_engine.py - cosine similarity search and answer building.

Purpose: verify that a real question finds the correct document, that the
source filename is returned, and that an unrelated question honestly returns
"no answer" instead of a wrong passage.
"""

from __future__ import annotations

import pytest

import config
from indexer import build_index
from qa_engine import (
    STATUS_ANSWERED,
    STATUS_EMPTY_INDEX,
    STATUS_EMPTY_QUESTION,
    STATUS_NO_ANSWER,
    answer_question,
    confidence_label,
    highlight_passage,
    normalise_question,
    question_terms,
    search_chunks,
)


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    """Build the index once for this module (faster than per test)."""
    import shutil

    folder = tmp_path_factory.mktemp("qa_docs")
    for source in config.SAMPLE_DOCUMENTS_DIR.iterdir():
        if source.is_file() and source.suffix.lower() in config.SUPPORTED_EXTENSIONS:
            shutil.copy2(source, folder / source.name)
    built, _ = build_index(folder)
    return built


def test_question_about_tfidf_returns_tfidf_document(index):
    answer = answer_question(index, "What is TF-IDF?")

    assert answer.status == STATUS_ANSWERED
    assert answer.source == "tfidf_and_retrieval.txt"
    assert "tf-idf" in answer.text.lower()
    assert answer.confidence in {"high", "medium"}
    assert answer.hits[0].cosine > config.DEFAULT_MIN_SCORE


def test_answer_quotes_the_document_instead_of_inventing_text(index):
    answer = answer_question(index, "What is cosine similarity?")

    text_lower = answer.text.lower()
    assert "cosine" in text_lower
    # Extract-based: the answer words really exist in the source passage.
    source_words = set(answer.hits[0].chunk.text.lower().split())
    answer_words = set(text_lower.split())
    assert answer_words.issubset(source_words)


def test_git_question_returns_git_cheatsheet(index):
    answer = answer_question(index, "How do I undo the last commit with git reset?")

    assert answer.status == STATUS_ANSWERED
    assert answer.source == "git_cheatsheet.md"
    assert "reset" in answer.text.lower()


def test_pdf_question_returns_pdf_source_and_page(index):
    answer = answer_question(index, "How do I reset the search index?")

    assert answer.status == STATUS_ANSWERED
    assert answer.source == "project_faq.pdf"
    assert answer.hits[0].chunk.page == 2
    assert "page 2" in answer.message


def test_streamlit_question_returns_streamlit_guide(index):
    answer = answer_question(index, "How do I run a Streamlit app?")

    assert answer.status == STATUS_ANSWERED
    assert answer.source == "streamlit_guide.md"


def test_unrelated_question_returns_no_answer(index):
    answer = answer_question(index, "Who won the football world cup in 1998?")

    assert answer.status == STATUS_NO_ANSWER
    assert answer.text == ""
    assert answer.best_score == 0.0


def test_question_with_no_shared_words_explains_why(index):
    answer = answer_question(index, "recipe for chocolate biryani")

    assert answer.status == STATUS_NO_ANSWER
    assert "none of the words" in answer.message.lower()


def test_very_high_threshold_refuses_weak_matches(index):
    answer = answer_question(index, "What is TF-IDF?", min_score=0.99)

    assert answer.status == STATUS_NO_ANSWER
    assert "no confident answer" in answer.message.lower()
    assert answer.hits, "closest passages are still offered for transparency"


def test_empty_question(index):
    assert answer_question(index, "   ").status == STATUS_EMPTY_QUESTION
    assert answer_question(index, "").status == STATUS_EMPTY_QUESTION


def test_empty_index(tmp_path):
    from indexer import DocumentIndex

    answer = answer_question(DocumentIndex(), "What is TF-IDF?")

    assert answer.status == STATUS_EMPTY_INDEX
    assert "no indexed documents" in answer.message.lower()


def test_none_index():
    assert answer_question(None, "anything").status == STATUS_EMPTY_INDEX


def test_search_returns_scored_hits_best_first(index):
    hits = search_chunks(index, "cosine similarity", top_k=5)

    assert len(hits) == 5
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    # cosine similarity itself always stays between 0 and 1
    assert all(0 <= hit.cosine <= 1.0001 for hit in hits)
    assert all(hit.percent <= 100 for hit in hits)
    assert all(hit.chunk.filename for hit in hits)


def test_top_k_is_clamped_to_available_chunks(index):
    hits = search_chunks(index, "python", top_k=9999)

    assert len(hits) <= min(len(index.chunks), config.MAX_TOP_K)


def test_source_list_is_deduplicated(index):
    answer = answer_question(index, "python function", top_k=5)

    sources = answer.sources()
    assert len(sources) == len(set(sources))
    assert answer.source in sources


def test_normalise_question():
    assert normalise_question("  What\n  is   TF-IDF?  ") == "What is TF-IDF?"


def test_question_terms_drop_question_words():
    terms = question_terms("What is the soft margin in SVM?")

    assert "soft" in terms
    assert "margin" in terms
    assert "what" not in terms


def test_confidence_labels():
    assert confidence_label(0.9) == "high"
    assert confidence_label(0.2) == "medium"
    assert confidence_label(0.05) == "low"
    assert confidence_label(0.0) == "none"


def test_highlight_passage_bolds_keywords():
    marked = highlight_passage("TF-IDF is a weighting method.", "What is TF-IDF?")

    assert "**TF-IDF**" in marked


def test_highlight_passage_escapes_markdown_characters():
    marked = highlight_passage("Uses *stars* and _underscores_.", "stars")

    assert "**stars**" in marked
    assert "\\*" in marked  # the raw asterisks stay escaped, so the page cannot break
    assert "\\_underscores\\_" in marked


def test_answer_text_is_punctuated(index):
    answer = answer_question(index, "What is a chunk?")

    assert answer.text
    assert answer.text[-1] in ".!?\"')"


def test_definition_question_prefers_the_definition_sentence(index):
    answer = answer_question(index, "What is TF-IDF?")

    assert answer.text.lower().startswith("tf-idf stands for")


def test_definition_question_detection():
    from qa_engine import is_definition_question

    assert is_definition_question("What is TF-IDF?")
    assert is_definition_question("Define cosine similarity")
    assert not is_definition_question("How do I reset the index?")


def test_query_coverage_counts_meaningful_words(index):
    from qa_engine import query_coverage

    tfidf_chunk = next(c for c in index.chunks if "TF-IDF stands for" in c.text)

    assert query_coverage("What is TF-IDF?", tfidf_chunk) == 1.0
    assert query_coverage("chocolate biryani recipe", tfidf_chunk) == 0.0
    # stop words alone carry no meaning
    assert query_coverage("what is the", tfidf_chunk) == 0.0


def test_phrase_match_bonus_lifts_exact_heading(index):
    from qa_engine import search_chunks

    hits = search_chunks(index, "What is TF-IDF?", top_k=1)

    assert "what is tf-idf" in hits[0].chunk.text.lower()
    assert hits[0].score > 0.2
