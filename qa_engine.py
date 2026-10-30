"""
qa_engine.py
============
Answers a question by finding the most similar document passages.

How it works (honest version - no magic):
    1. The question is cleaned and converted to a TF-IDF vector using the SAME
       vectorizer that was fit on the documents.
    2. cosine similarity is computed between the question vector and every
       chunk vector. 1.0 = identical wording direction, 0.0 = nothing shared.
    3. The best chunks are shown as evidence, with the source filename.
    4. The short answer is EXTRACTIVE: we pick the sentences of those passages
       that best match the question. The app never invents text - it quotes
       your own documentation. That is why there is no hallucination and no API.

Beginner note (Telugu-English):
    Cosine similarity = rendu vectors madhya angle. Angle chinna ga unte
    (score 1 ki daggaraga) meaning daggaraga untundi. So highest score unna
    chunk = best matching passage. Score chala takkuva unte (default 0.10
    kanna takkuva) manam "answer dorakaledu" ani cheptam - wrong answer ivvam.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import config
from indexer import STOP_WORDS_NORMALISED, DocumentIndex
from text_processor import Chunk, split_sentences, tokenize

STATUS_ANSWERED = "answered"
STATUS_NO_ANSWER = "no_answer"
STATUS_EMPTY_INDEX = "empty_index"
STATUS_EMPTY_QUESTION = "empty_question"

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'_-]+")

# "What is ..." / "Define ..." questions usually want a definition sentence.
_DEFINITION_QUESTION = re.compile(
    r"^\s*(what|who)\s+(is|are|was|were|does)\b|^\s*(define|explain|describe)\b", re.I
)
_DEFINITION_CUE = re.compile(
    r"\b(is|are|was|were|means|stands? for|refers? to|measures|describes|defines|represents)\b",
    re.I,
)
DEFINITION_BONUS = 0.12          # sentence defines something related
DEFINITION_STRONG_BONUS = 0.30   # sentence STARTS with the asked term and defines it
# Small prior so a full sentence beats a short heading fragment like
# "Why Python for machine learning." when both match the question.
SENTENCE_LENGTH_PRIOR = 0.05
SENTENCE_SWEET_SPOT = (60, 320)


@dataclass
class SearchHit:
    """
    One matching passage with two numbers:

    * `cosine` - the pure cosine similarity (0.0 - 1.0), used for honesty and
      thresholds.
    * `score`  - cosine plus the small phrase bonus, used only for ranking.
    """

    chunk: Chunk
    score: float
    cosine: float = 0.0

    @property
    def source(self) -> str:
        return self.chunk.filename

    @property
    def percent(self) -> int:
        """Cosine similarity as an easy-to-read percentage for the UI."""
        return int(round(max(0.0, min(self.cosine, 1.0)) * 100))


@dataclass
class Answer:
    """The final result shown to the user."""

    question: str
    status: str
    text: str = ""
    hits: list[SearchHit] = field(default_factory=list)
    best_score: float = 0.0
    coverage: float = 0.0             # share of question keywords found in the passage
    confidence: str = "none"          # high / medium / low / none
    message: str = ""
    source: str | None = None

    @property
    def answered(self) -> bool:
        return self.status == STATUS_ANSWERED

    def sources(self) -> list[str]:
        """Unique source files of the shown evidence, in score order."""
        seen: list[str] = []
        for hit in self.hits:
            if hit.source not in seen:
                seen.append(hit.source)
        return seen

    def score_breakdown(self) -> dict[str, int]:
        return {hit.source: hit.percent for hit in self.hits}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalise_question(question: str) -> str:
    """Trim and collapse whitespace so '  what   is TF-IDF? ' works too."""
    return re.sub(r"\s+", " ", (question or "").replace("\n", " ")).strip()


def confidence_label(score: float) -> str:
    if score >= config.HIGH_CONFIDENCE:
        return "high"
    if score >= config.MEDIUM_CONFIDENCE:
        return "medium"
    if score > 0:
        return "low"
    return "none"


def query_coverage(question: str, chunk: Chunk) -> float:
    """
    Share (0.0 - 1.0) of the question's meaningful words that really appear in
    the passage.

    Why: cosine similarity can be fooled by one common word. If only a small
    part of the question is in the passage, we prefer to say "no answer" rather
    than show a misleading quote.
    """
    query_terms = {term for term in tokenize(question) if term not in STOP_WORDS_NORMALISED}
    if not query_terms:
        return 0.0
    chunk_terms = set(tokenize(chunk.text))
    return len(query_terms & chunk_terms) / len(query_terms)


def is_definition_question(question: str) -> bool:
    """True for questions like 'What is TF-IDF?' or 'Define cosine similarity'."""
    return bool(_DEFINITION_QUESTION.search(question or ""))


def question_terms(question: str) -> list[str]:
    """Content words of a question, used for highlighting + explanations."""
    stop = {
        "the", "is", "are", "a", "an", "of", "to", "in", "and", "or", "for",
        "what", "which", "who", "how", "why", "when", "where", "does", "do",
        "can", "i", "you", "me", "my", "it", "this", "that", "explain", "tell",
        "about", "please", "give", "with", "on", "be", "was", "were", "there",
    }
    terms: list[str] = []
    for word in _WORD.findall(question or ""):
        lower = word.lower()
        if len(lower) < 3 or lower in stop:
            continue
        if lower not in terms:
            terms.append(lower)
    return terms


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------
def search_chunks(
    index: DocumentIndex,
    question: str,
    top_k: int = config.DEFAULT_TOP_K,
) -> list[SearchHit]:
    """
    Return the top_k most similar chunks (score > 0), best first.

    cosine_similarity comes from scikit-learn and works on sparse matrices, so
    even thousands of chunks are fast.
    """
    if index is None or index.is_empty:
        return []

    query = normalise_question(question)
    if not query:
        return []

    from sklearn.metrics.pairwise import cosine_similarity

    query_vector = index.vectorizer.transform([query])
    if query_vector.nnz == 0:
        # No word of the question exists in the documents' vocabulary.
        return []

    scores = cosine_similarity(query_vector, index.matrix).ravel()
    top_k = max(1, min(int(top_k), len(index.chunks), config.MAX_TOP_K))

    # Look at a slightly bigger pool, then rerank with a phrase bonus.
    pool_size = min(len(scores), max(top_k * 3, top_k + 5))

    # argpartition is O(n) instead of sorting everything - matters for big docs.
    if len(scores) > pool_size:
        candidate_idx = scores.argpartition(-pool_size)[-pool_size:]
    else:
        candidate_idx = range(len(scores))

    # Reranking bonus: if the whole question phrase appears in the passage
    # (common when documentation has headings like "What is TF-IDF?"), that
    # passage is clearly very relevant. Base ranking stays cosine similarity.
    phrase = re.sub(r"[^a-z0-9\s-]", "", query.lower()).strip()
    use_phrase = len(phrase.split()) >= 2

    hits = []
    for i in candidate_idx:
        if scores[i] <= 0:
            continue
        bonus = 0.0
        if use_phrase and phrase in " ".join(index.chunks[i].text.lower().split()):
            bonus = config.PHRASE_MATCH_BONUS
        cosine = float(scores[i])
        hits.append(SearchHit(chunk=index.chunks[i], score=cosine + bonus, cosine=cosine))

    hits.sort(key=lambda hit: (-hit.score, hit.chunk.filename.lower(), hit.chunk.index))
    return hits[:top_k]


def _is_complete_sentence(sentence: str) -> bool:
    """True when a sentence is not cut off by a chunk boundary."""
    return sentence.rstrip()[-1:] in {".", "!", "?", '"', "'"}


def _context_sentence(
    sentences: list[str], start: int, query_terms: set[str]
) -> str | None:
    """
    Next sentence after `start`, but only when it is complete, long enough, and
    still talks about the question (shares a keyword). This keeps answers short
    and avoids jumping into an unrelated section of the document.
    """
    for position in range(start, len(sentences)):
        candidate = sentences[position]
        if len(candidate) < config.MIN_SENTENCE_CHARS:
            continue
        if not _is_complete_sentence(candidate):
            continue
        if query_terms & set(tokenize(candidate)):
            return candidate
        return None
    return None


def _build_extractive_answer(question: str, index: DocumentIndex, hits: list[SearchHit]) -> str:
    """
    Build a short, readable answer that is QUOTED from the best passage.

    Steps:
      1. Split the strongest passages into sentences.
      2. Score each sentence with cosine similarity to the question.
      3. For "What is ...?" questions, give a small bonus to definition sentences
         ("TF-IDF stands for ...", "Cosine similarity measures ...").
      4. Return the winning sentence plus its next sentence as context.

    Nothing is generated: every word comes from the uploaded documents.
    """
    if not hits:
        return ""

    from sklearn.metrics.pairwise import cosine_similarity

    query = normalise_question(question)
    query_vector = index.vectorizer.transform([query])
    query_term_set = {term for term in tokenize(query) if term not in STOP_WORDS_NORMALISED}
    definitional = is_definition_question(query)

    best: tuple[float, int, int, str] | None = None  # (value, rank, position, sentence)

    for rank, hit in enumerate(hits):
        if rank > 0 and hit.score < hits[0].score * 0.6:
            continue  # only strong passages can supply the answer sentence
        sentences = split_sentences(hit.chunk.text)
        candidates = [
            (position, sentence)
            for position, sentence in enumerate(sentences)
            if len(sentence) >= config.MIN_SENTENCE_CHARS
            and _is_complete_sentence(sentence)
        ]
        if not candidates:
            continue

        scores = cosine_similarity(
            query_vector, index.vectorizer.transform([sentence for _, sentence in candidates])
        ).ravel()

        for (position, sentence), score in zip(candidates, scores):
            value = float(score)
            if rank == 0:
                value += 0.02  # tiny tie-breaker in favour of the best passage
            if SENTENCE_SWEET_SPOT[0] <= len(sentence) <= SENTENCE_SWEET_SPOT[1]:
                value += SENTENCE_LENGTH_PRIOR
            else:
                value -= SENTENCE_LENGTH_PRIOR
            if definitional and _DEFINITION_CUE.search(sentence):
                sentence_terms = tokenize(sentence)
                if sentence_terms and sentence_terms[0] in query_term_set:
                    # e.g. "TF-IDF stands for ..." / "Cosine similarity measures ..."
                    value += DEFINITION_STRONG_BONUS
                elif query_term_set & set(sentence_terms):
                    value += DEFINITION_BONUS
            if best is None or value > best[0]:
                best = (value, rank, position, sentence)

    if best is None:
        return _clean_answer(hits[0].chunk.text.strip())

    _, rank, position, primary = best
    sentences = split_sentences(hits[rank].chunk.text)

    parts = [primary]
    if config.MAX_ANSWER_SENTENCES > 1:
        context = _context_sentence(sentences, position + 1, query_term_set)
        if context and context != primary:
            parts.append(context)

    return _clean_answer(" ".join(parts))


def _clean_answer(text: str) -> str:
    """Tidy spacing and make sure the passage ends with punctuation."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    if text and text[-1] not in ".!?\"')":
        text += "."
    return text


def answer_question(
    index: DocumentIndex | None,
    question: str,
    top_k: int = config.DEFAULT_TOP_K,
    min_score: float = config.DEFAULT_MIN_SCORE,
) -> Answer:
    """
    Full pipeline: clean question -> TF-IDF -> cosine similarity -> answer.

    Returns an Answer object; it never raises for user input problems.
    """
    query = normalise_question(question)
    if not query:
        return Answer(
            question=question,
            status=STATUS_EMPTY_QUESTION,
            message="Type a question first, for example: 'What is TF-IDF?'",
        )

    if index is None or index.is_empty:
        return Answer(
            question=query,
            status=STATUS_EMPTY_INDEX,
            message=(
                "There are no indexed documents yet. Upload a TXT/MD/PDF file or add "
                "one to the documents folder, then ask again."
            ),
        )

    hits = search_chunks(index, query, top_k=top_k)
    if not hits:
        return Answer(
            question=query,
            status=STATUS_NO_ANSWER,
            message=(
                "None of the words in your question appear in the uploaded documents. "
                "Try different keywords, or upload a document about this topic."
            ),
            best_score=0.0,
            confidence="none",
        )

    best = hits[0]
    coverage = query_coverage(query, best.chunk)

    if best.cosine < min_score or coverage < config.MIN_QUERY_COVERAGE:
        if best.cosine < min_score:
            reason = (
                f"best similarity {best.cosine:.2f} is below the {min_score:.2f} threshold"
            )
        else:
            reason = (
                f"only {coverage:.0%} of your question keywords appear in that passage "
                f"(similarity {best.cosine:.2f})"
            )
        return Answer(
            question=query,
            status=STATUS_NO_ANSWER,
            hits=hits,
            best_score=best.cosine,
            coverage=coverage,
            confidence="low",
            message=(
                f"No confident answer found: {reason}. The closest passage is "
                f"from '{best.source}', but it probably does not answer your "
                "question. Try rephrasing with words used in the document, upload "
                "a document about this topic, or lower the sensitivity in the sidebar."
            ),
            source=best.source,
        )

    answer_text = _build_extractive_answer(query, index, hits)
    return Answer(
        question=query,
        status=STATUS_ANSWERED,
        text=answer_text,
        hits=hits,
        best_score=best.cosine,
        coverage=coverage,
        confidence=confidence_label(best.cosine),
        source=best.source,
        message=f"Answered from '{best.source}', {best.chunk.location}.",
    )


# ---------------------------------------------------------------------------
# Display helper (nice highlighting in the Streamlit UI)
# ---------------------------------------------------------------------------
def highlight_passage(text: str, question: str) -> str:
    """
    Bold the question keywords inside a passage (markdown string for Streamlit).
    Markdown special characters are escaped first so the text renders cleanly.
    """
    safe = text.replace("\\", "\\\\").replace("*", "\\*").replace("_", "\\_")
    safe = safe.replace("[", "\\[").replace("`", "\\`")
    safe = safe.replace("<", "&lt;").replace(">", "&gt;")

    terms = sorted(question_terms(question), key=len, reverse=True)
    for term in terms[:12]:
        pattern = re.compile(rf"(?i)\b({re.escape(term)}\w*)")
        safe = pattern.sub(lambda match: f"**{match.group(1)}**", safe, count=2)
    return safe


__all__ = [
    "SearchHit",
    "Answer",
    "STATUS_ANSWERED",
    "STATUS_NO_ANSWER",
    "STATUS_EMPTY_INDEX",
    "STATUS_EMPTY_QUESTION",
    "normalise_question",
    "confidence_label",
    "query_coverage",
    "is_definition_question",
    "question_terms",
    "search_chunks",
    "answer_question",
    "highlight_passage",
]
