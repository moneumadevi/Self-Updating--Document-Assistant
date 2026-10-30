"""
app.py
======
Self Updating Documentation Assistant - the Streamlit web interface.

Run with:
    streamlit run app.py

Beginner note (Telugu-English):
    Ee file lo UI matrame undi - search logic antha vera files lo undi
    (document_loader.py, text_processor.py, indexer.py, qa_engine.py).
    Streamlit prathi interaction ki script ni top-to-bottom run chestundi, so
    costly pani (index build) ni session_state lo cache chestunnam.
"""

from __future__ import annotations

import html
import os
import shutil
import time
from pathlib import Path

import streamlit as st

import config
from document_loader import (
    DocumentError,
    EmptyDocumentError,
    is_supported_file,
    load_document,
    safe_upload_filename,
    unique_destination,
)
from indexer import DocumentIndex, IndexManager
from qa_engine import (
    STATUS_ANSWERED,
    STATUS_EMPTY_INDEX,
    answer_question,
    highlight_passage,
)
from text_processor import text_preview
from watcher import FolderWatcher, scan_folder

# ---------------------------------------------------------------------------
# Paths (read at runtime so an env var / test can point somewhere else)
# ---------------------------------------------------------------------------
DOCUMENTS_DIR = Path(os.environ.get("DOCS_ASSISTANT_DOCUMENTS", str(config.DOCUMENTS_DIR)))
INDEX_DIR = Path(os.environ.get("DOCS_ASSISTANT_INDEX", str(config.INDEX_DIR)))

MAX_UPLOAD_MB = 10

EXAMPLES = [
    "What is TF-IDF?",
    "How do I undo the last Git commit?",
    "How do I reset the search index?",
    "How do I run a Streamlit app?",
]

CSS = """
<style>
    .block-container {padding-top: 2rem; max-width: 1150px;}
    .hero-title {
        font-size: 2.3rem; font-weight: 800; margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #4f46e5 0%, #0ea5e9 60%, #14b8a6 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-sub {color: #64748b; font-size: 0.98rem; margin-bottom: 1.1rem;}
    .answer-card {
        background: #f5f7ff; border: 1px solid #dbe4ff; border-left: 6px solid #4f46e5;
        border-radius: 12px; padding: 1rem 1.2rem; margin: 0.4rem 0 0.9rem 0;
        color: #1f2937; font-size: 1.02rem; line-height: 1.6;
    }
    .answer-label {font-size: 0.72rem; letter-spacing: .08em; text-transform: uppercase;
        color: #6366f1; font-weight: 700; margin-bottom: .35rem;}
    .badge {display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 700; margin-right: 6px;}
    .badge-high {background: #dcfce7; color: #166534;}
    .badge-medium {background: #fef9c3; color: #854d0e;}
    .badge-low {background: #fee2e2; color: #991b1b;}
    .badge-neutral {background: #e2e8f0; color: #334155;}
    .source-line {color: #475569; font-size: 0.87rem; margin-top: 0.5rem;}
    .pill {display: inline-block; background: #eef2ff; color: #3730a3; border-radius: 8px;
        padding: 2px 9px; font-size: 0.78rem; font-weight: 600; margin-right: 6px;}
    .hint-box {background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 10px;
        padding: 0.8rem 1rem; color: #475569; font-size: 0.9rem;}
    .passage {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 0.75rem 0.9rem; color: #1f2937; font-size: 0.94rem; line-height: 1.55;
    }
    div[data-testid="stSidebar"] .stButton button {width: 100%;}
</style>
"""


# ---------------------------------------------------------------------------
# Cached manager (one IndexManager for the whole server process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_manager(documents_dir: str, index_dir: str) -> IndexManager:
    return IndexManager(documents_dir=documents_dir, index_dir=index_dir)


def manager() -> IndexManager:
    return get_manager(str(DOCUMENTS_DIR), str(INDEX_DIR))


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def init_session() -> None:
    ss = st.session_state
    config.ensure_directories()
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if "index" not in ss:
        ss.index = None
        ss.last_report = None
        ss.index_loaded_from_disk = False
        ss.history = []
        ss.pending_question = None
        ss.uploader_key = 0
        ss.watcher = FolderWatcher(DOCUMENTS_DIR)
        ss.watch_message = ""
        ss.toast_message = ""
        ss.upload_messages = []

    if "auto_watch" not in ss:
        ss.auto_watch = True
    if "top_k" not in ss:
        ss.top_k = config.DEFAULT_TOP_K
    if "min_score" not in ss:
        ss.min_score = config.DEFAULT_MIN_SCORE


def ensure_index(force: bool = False) -> DocumentIndex:
    """
    Always up to date, always cheap:
      * index on disk + folder unchanged -> just load it (few milliseconds)
      * folder changed / force          -> re-index, re-using unchanged files
    """
    ss = st.session_state
    index, report = manager().get(force_rebuild=force)

    if report is not None:
        ss.index = index
        ss.last_report = report
        ss.index_loaded_from_disk = False
        if report.errors:
            ss.toast_message = f"Rebuild finished with {len(report.errors)} problem(s)."
        elif report.has_changes:
            ss.toast_message = "Documents changed - search index rebuilt automatically."
    elif ss.index is None:
        ss.index = index
        ss.index_loaded_from_disk = True

    # After any full app run the watcher baseline equals the folder, so the
    # 3-second auto-check does not report the same change twice.
    ss.watcher.reset()
    return ss.index


def rebuild(force_report: bool = True) -> None:
    index, report = manager().get(force_rebuild=force_report)
    st.session_state.index = index
    st.session_state.last_report = report
    st.session_state.watcher.reset()


def copy_sample_documents() -> tuple[list[str], list[str]]:
    """Copy the sample dataset into the watched folder (never overwrites)."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    skipped: list[str] = []
    if not config.SAMPLE_DOCUMENTS_DIR.is_dir():
        return copied, skipped
    for source in sorted(config.SAMPLE_DOCUMENTS_DIR.iterdir()):
        if not source.is_file() or not is_supported_file(source.name):
            continue
        destination = DOCUMENTS_DIR / source.name
        if destination.exists():
            skipped.append(source.name)
            continue
        shutil.copy2(source, destination)
        copied.append(source.name)
    return copied, skipped


def save_uploaded_file(uploaded) -> tuple[str, str]:
    """
    Validate + save one uploaded file into the documents folder.

    Returns (status, message) where status is "ok" | "skipped" | "error".
    """
    name = getattr(uploaded, "name", "uploaded_file")
    try:
        safe_name = safe_upload_filename(name)
    except Exception:
        return "error", f"'{name}': the file name is not valid."

    if not is_supported_file(safe_name):
        return (
            "error",
            f"'{name}': unsupported format. Allowed: {config.human_supported_formats()}.",
        )

    data = uploaded.getvalue()
    if not data:
        return "error", f"'{name}': the file is empty (0 bytes)."
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return "error", f"'{name}': bigger than {MAX_UPLOAD_MB} MB. Please upload a smaller file."

    destination = unique_destination(DOCUMENTS_DIR, safe_name)
    try:
        destination.write_bytes(data)
    except OSError as exc:
        return "error", f"'{name}': could not be saved ({exc})."

    # Read it once now so the user gets instant feedback (empty/corrupt PDF etc.).
    try:
        load_document(destination)
    except EmptyDocumentError as exc:
        destination.unlink(missing_ok=True)
        return "error", f"'{name}': {exc}"
    except DocumentError as exc:
        destination.unlink(missing_ok=True)
        return "error", f"'{name}': {exc}"
    except Exception as exc:  # safety net, never crash the page
        destination.unlink(missing_ok=True)
        return "error", f"'{name}': could not be read ({exc})."

    if destination.name != safe_name:
        return "ok", f"'{safe_name}' saved as '{destination.name}' (a file with that name already existed)."
    return "ok", f"'{destination.name}' added to the documents folder."


# ---------------------------------------------------------------------------
# Small UI building blocks
# ---------------------------------------------------------------------------
def badge(label: str, kind: str = "neutral") -> str:
    return f'<span class="badge badge-{kind}">{html.escape(label)}</span>'


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def render_header() -> None:
    st.markdown('<div class="hero-title">Self Updating Documentation Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Upload your documentation, ask a question, and get the exact '
        "passage that answers it - with the source file name. TF-IDF + cosine similarity, "
        "100% local, no API key.</div>",
        unsafe_allow_html=True,
    )


def render_sidebar(index: DocumentIndex) -> None:
    ss = st.session_state
    with st.sidebar:
        st.markdown("### 📚 Documentation Assistant")
        st.caption("TF-IDF retrieval · offline · no paid APIs")

        st.divider()
        st.markdown("**Index status**")
        col_a, col_b = st.columns(2)
        col_a.metric("Documents", index.file_count if index else 0)
        col_b.metric("Passages", index.chunk_count if index else 0)
        col_c, col_d = st.columns(2)
        col_c.metric("Vocabulary", index.vocabulary_size if index else 0)
        col_d.metric("Words", index.total_words if index else 0)

        if ss.index_loaded_from_disk:
            st.caption("Loaded the saved index from `index_store/` without re-reading files.")
        if index and index.built_at:
            st.caption(f"Last build: {index.built_at.replace('T', ' ')}")

        report = ss.last_report
        if report is not None:
            if report.duration_seconds:
                st.caption(
                    f"Build took {report.duration_seconds:.2f}s · "
                    f"{len(report.unchanged)} unchanged file(s) reused from cache"
                )
            if report.has_changes:
                st.info(f"Latest rebuild: {report.summary()}", icon="🔄")
            elif report.errors:
                st.warning(report.summary(), icon="⚠️")

        st.divider()
        st.markdown("**Live folder watch**")
        st.toggle(
            "Auto-detect changes",
            key="auto_watch",
            help=f"Checks '{DOCUMENTS_DIR.name}/' every {config.AUTO_REFRESH_SECONDS} seconds.",
        )
        if st.button("Check folder now", width="stretch"):
            with st.spinner("Checking the documents folder..."):
                changes = ss.watcher.poll()
                if changes.has_changes:
                    ss.watch_message = changes.summary()
                    rebuild()
                else:
                    ss.watch_message = "No changes detected. Index is up to date."
            st.rerun()

        if st.button("Rebuild index from scratch", width="stretch"):
            with st.spinner("Re-reading every document..."):
                index_new, report_new = manager().build_fresh()
                ss.index = index_new
                ss.last_report = report_new
                ss.watcher.reset()
            st.rerun()

        if st.button("Clear chat history", width="stretch"):
            ss.history = []
            st.rerun()

        if ss.watch_message:
            st.caption(ss.watch_message)

        st.divider()
        st.markdown("**Answer settings**")
        st.slider(
            "Passages to show",
            min_value=1,
            max_value=config.MAX_TOP_K,
            key="top_k",
        )
        st.slider(
            "Minimum similarity",
            min_value=0.0,
            max_value=0.9,
            step=0.01,
            key="min_score",
            help="Higher = stricter. If you see 'no answer found', lower this a little.",
        )

        st.divider()
        st.markdown("**Sample data**")
        if st.button("Load sample documents", width="stretch"):
            copied, skipped = copy_sample_documents()
            if copied:
                rebuild()
                ss.watch_message = f"Added {len(copied)} sample file(s) and rebuilt the index."
                ss.toast_message = f"{len(copied)} sample file(s) added."
                st.rerun()
            elif skipped:
                st.info(f"All {len(skipped)} sample file(s) are already in the folder.")
            else:
                st.warning("Sample folder is missing or empty.")

        st.caption(f"Watching: `{DOCUMENTS_DIR}`")
        st.caption(f"Index stored in: `{INDEX_DIR}`")


def render_no_answer(answer) -> None:
    kind = "info" if answer.status == STATUS_EMPTY_INDEX else "warning"
    icon = "📂" if answer.status == STATUS_EMPTY_INDEX else "🤔"
    getattr(st, kind)(answer.message, icon=icon)

    if answer.hits:
        st.markdown("**Closest passages (below the confidence threshold)**")
        for hit in answer.hits:
            with st.expander(f"{hit.source} · {hit.chunk.location} · {hit.percent}% match"):
                st.markdown(highlight_passage(hit.chunk.text, answer.question))


def render_answer(answer) -> None:
    if answer.status != STATUS_ANSWERED:
        render_no_answer(answer)
        return

    sources = answer.sources()
    confidence = answer.confidence if answer.confidence in {"high", "medium", "low"} else "neutral"
    st.markdown(
        '<div class="answer-card">'
        '<div class="answer-label">Answer from your documents</div>'
        f"{html.escape(answer.text)}"
        '<div class="source-line">'
        f'<span class="pill">📄 {html.escape(sources[0])}</span>'
        f'<span class="pill">📍 {html.escape(answer.hits[0].chunk.location)}</span>'
        f'<span class="pill">🎯 {answer.hits[0].percent}% match</span>'
        f"{badge(answer.confidence + ' confidence', confidence)}"
        "</div></div>",
        unsafe_allow_html=True,
    )

    meta_a, meta_b, meta_c = st.columns(3)
    meta_a.caption(f"Cosine similarity: {answer.best_score:.3f}")
    meta_b.caption(f"Question keywords found: {answer.coverage:.0%}")
    meta_c.caption(f"Source file(s): {len(sources)}")

    st.markdown("**Evidence used**")
    for rank, hit in enumerate(answer.hits, start=1):
        header = f"{rank}. {hit.source} · {hit.chunk.location} · {hit.percent}% match"
        with st.expander(header, expanded=(rank == 1)):
            st.markdown(
                f'<div class="passage">{highlight_passage(hit.chunk.text, answer.question)}</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"{hit.chunk.word_count} words · cosine similarity {hit.cosine:.4f}"
            )


def render_history(index: DocumentIndex) -> None:
    history = st.session_state.history
    if not history:
        return
    st.divider()
    head_a, head_b = st.columns([4, 1])
    head_a.markdown(f"**Recent questions** ({len(history)})")
    if head_b.button("Clear", key="clear_history_inline"):
        st.session_state.history = []
        st.rerun()

    for entry in reversed(history):
        with st.expander(f"❓ {entry['question']}", expanded=False):
            if entry["status"] == STATUS_ANSWERED:
                st.markdown(f"> {entry['answer']}")
                st.caption(
                    f"Source: {entry['sources'][0] if entry['sources'] else 'unknown'} · "
                    f"{entry['confidence']} confidence · asked at {entry['time']}"
                )
            else:
                st.caption(entry["answer"] or entry["message"])


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def render_ask_tab(index: DocumentIndex) -> None:
    ss = st.session_state

    if index is None or index.is_empty:
        st.info(
            "No documents are indexed yet. Open the **Documents** tab to upload a "
            "TXT / MD / PDF file, or press **Load sample documents** in the sidebar.",
            icon="📂",
        )

    # Example questions: set the input box and ask automatically on the next run.
    st.markdown("**Try an example question**")
    example_cols = st.columns(len(EXAMPLES))
    for column, example in zip(example_cols, EXAMPLES):
        if column.button(example, key=f"example_{example}", width="stretch"):
            ss.pending_question = example
            st.rerun()

    if ss.pending_question:
        ss.question_input = ss.pending_question
        ss.pending_question = None
        auto_ask = True
    else:
        auto_ask = False

    with st.form("ask_form", clear_on_submit=False):
        question = st.text_input(
            "Your question",
            key="question_input",
            placeholder="e.g. What is cosine similarity and why is it used?",
        )
        submitted = st.form_submit_button("Get answer", type="primary", width="stretch")

    if submitted or auto_ask:
        query = (question or "").strip()
        if not query:
            st.warning("Please type a question first.", icon="✍️")
        else:
            with st.spinner("Searching your documents..."):
                answer = answer_question(
                    index,
                    query,
                    top_k=st.session_state.get("top_k", config.DEFAULT_TOP_K),
                    min_score=st.session_state.get("min_score", config.DEFAULT_MIN_SCORE),
                )
            st.session_state.history.append(
                {
                    "question": query,
                    "status": answer.status,
                    "answer": answer.text,
                    "message": answer.message,
                    "confidence": answer.confidence,
                    "sources": answer.sources(),
                    "time": time.strftime("%H:%M:%S"),
                }
            )
            st.session_state.last_answer = answer

    answer = st.session_state.get("last_answer")
    if answer is not None:
        st.divider()
        st.caption(f"Question: **{answer.question}**")
        render_answer(answer)

    render_history(index)


def render_documents_tab(index: DocumentIndex) -> None:
    ss = st.session_state
    st.subheader("Upload documentation")
    st.caption(
        f"Accepted formats: {config.human_supported_formats()} · max {MAX_UPLOAD_MB} MB per file · "
        f"files are saved in `{DOCUMENTS_DIR.name}/` and indexed automatically."
    )

    uploaded_files = st.file_uploader(
        "Choose one or more files",
        type=[ext.replace(".", "") for ext in sorted(config.SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
        key=f"uploader_{ss.uploader_key}",
        help="TXT, Markdown and text-based PDF files work best. Scanned PDFs need OCR first.",
    )

    for status, message in ss.upload_messages:
        if status == "ok":
            st.success(message, icon="✅")
        else:
            st.error(message, icon="🚫")
    ss.upload_messages = []

    if uploaded_files and st.button("Add to documents folder", type="primary"):
        results = [save_uploaded_file(uploaded) for uploaded in uploaded_files]
        saved = [message for status, message in results if status == "ok"]
        errors = [message for status, message in results if status == "error"]

        ss.upload_messages = results
        if saved:
            rebuild()
            ss.uploader_key += 1  # reset the widget so the uploaded list is empty again
            ss.toast_message = f"{len(saved)} file(s) added and indexed."
        if errors and not saved:
            ss.toast_message = f"{len(errors)} file(s) could not be added."
        st.rerun()

    st.divider()
    st.subheader("Indexed documents")

    files = index.file_names() if index else []
    if not files:
        st.markdown(
            '<div class="hint-box">Nothing indexed yet. Upload a file above, or add files '
            f"directly to the <code>{html.escape(DOCUMENTS_DIR.name)}/</code> folder - the app "
            "detects them automatically.</div>",
            unsafe_allow_html=True,
        )
        return

    rows = []
    for name in files:
        record = index.files[name]
        rows.append(
            {
                "File": name,
                "Type": Path(name).suffix.replace(".", "").upper(),
                "Size": format_size(record.size),
                "Pages": record.page_count or 1,
                "Passages": record.chunk_count,
                "Status": "indexed" if record.status == "indexed" else "error",
                "Problem": record.error or "",
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)

    error_files = {name: rec.error for name, rec in index.files.items() if rec.status == "error"}
    if error_files:
        st.warning("Some files could not be indexed. They are skipped, the rest still work.", icon="⚠️")
        for name, error in error_files.items():
            st.caption(f"**{name}** · {error}")

    st.divider()
    st.subheader("Browse passages")
    selected = st.selectbox(
        "File", options=[name for name in files if index.files[name].status == "indexed"] or files
    )
    if selected:
        chunks = index.chunks_for(selected)
        st.caption(f"{len(chunks)} passage(s) from `{selected}` - this is exactly what the search uses.")
        for chunk in chunks[:25]:
            with st.expander(f"{chunk.location} · {chunk.word_count} words · {text_preview(chunk.text, 10)}"):
                st.write(chunk.text)
        if len(chunks) > 25:
            st.caption(f"... and {len(chunks) - 25} more passages (hidden to keep the page fast).")

    st.divider()
    st.subheader("Delete a document")
    st.caption(
        "Deleting here removes the file from the folder, so the index rebuilds and the "
        "content is no longer used in answers."
    )
    delete_target = st.selectbox("File to delete", options=files, key="delete_target")
    confirm = st.checkbox(f"Yes, delete `{delete_target}` from the documents folder", key="confirm_delete")
    if st.button("Delete selected document", disabled=not confirm):
        path = DOCUMENTS_DIR / delete_target
        try:
            path.unlink(missing_ok=True)
            for key in ("delete_target", "confirm_delete"):
                if key in st.session_state:
                    del st.session_state[key]
            rebuild()
            st.success(f"Deleted '{delete_target}'. The index was rebuilt.")
            st.rerun()
        except OSError as exc:
            st.error(f"Could not delete '{delete_target}': {exc}")


def render_watch_tab(index: DocumentIndex) -> None:
    ss = st.session_state
    st.subheader("Automatic document updates")
    st.markdown(
        f"""
The app watches the **`{DOCUMENTS_DIR.name}/`** folder. When you **add**, **edit** or
**delete** a supported file there, the search index is rebuilt automatically.
Unchanged files are re-used from the local cache, so only the edited file is
processed again.
"""
    )

    st.code(str(DOCUMENTS_DIR), language="text")

    col_1, col_2 = st.columns(2)
    with col_1:
        st.markdown("**Settings**")
        st.caption(
            "Auto-detect is "
            + ("**ON**" if st.session_state.get("auto_watch", True) else "**OFF**")
            + " - change it with the toggle in the sidebar."
        )
        if st.button("Check for changes now", width="stretch"):
            changes = ss.watcher.poll()
            ss.watch_message = changes.summary()
            if changes.has_changes:
                rebuild()
                ss.toast_message = "Index rebuilt after folder changes."
            st.rerun()
    with col_2:
        st.markdown("**Manual rebuild**")
        st.caption("Use this after fixing a broken file or changing chunk settings.")
        if st.button("Force full rebuild", width="stretch"):
            with st.spinner("Processing every document..."):
                new_index, report = manager().build_fresh()
                ss.index = new_index
                ss.last_report = report
                ss.watcher.reset()
            st.success(report.summary())
            st.rerun()

    st.divider()
    st.markdown("**Detected files**")
    snapshot = scan_folder(DOCUMENTS_DIR)
    if not snapshot:
        st.info(f"The folder `{DOCUMENTS_DIR}` has no supported files yet.")
    else:
        table = []
        for name, snap in sorted(snapshot.items(), key=lambda item: item[0].lower()):
            record = index.files.get(name) if index else None
            if record is None:
                state = "not indexed yet"
            elif record.status == "error":
                state = "error while reading"
            elif record.size != snap.size or record.mtime_ns != snap.mtime_ns:
                state = "changed since last build"
            else:
                state = "indexed"
            table.append(
                {
                    "File": name,
                    "Size": format_size(snap.size),
                    "Modified": time.strftime(
                        "%d-%b-%Y %H:%M:%S", time.localtime(snap.mtime_ns / 1_000_000_000)
                    ),
                    "State": state,
                }
            )
        st.dataframe(table, width="stretch", hide_index=True)

    if ss.watch_message:
        st.info(ss.watch_message)

    st.divider()
    st.markdown("**Run the watcher from the terminal (optional)**")
    st.markdown(
        "You can rebuild the index without opening the web page. This is handy when you "
        "edit Markdown files in VS Code:"
    )
    st.code("python watcher.py", language="bash")
    st.code("python watcher.py --once", language="bash")

    manifest = manager().read_manifest()
    if manifest:
        with st.expander("Saved index manifest (index_store/manifest.json)"):
            st.json(manifest)


def render_about_tab(index: DocumentIndex) -> None:
    st.subheader("How this project works")
    st.markdown(
        """
1. **Extract** - `document_loader.py` reads TXT/Markdown with Python's built-in
   `open()` and PDFs with `pypdf`. Empty or unsupported files raise clear errors.
2. **Clean and chunk** - `text_processor.py` removes Markdown noise and splits each
   page into ~120-word passages with a 20-word overlap.
3. **Vectorise (this is the "learning" step)** - `scikit-learn`'s `TfidfVectorizer`
   is **fit** on your passages. Fitting learns the vocabulary and the IDF weights
   from your own documents. It is classic NLP maths, not a neural network, and it
   runs offline in milliseconds.
4. **Store** - the vectorizer, the sparse TF-IDF matrix and the passages are saved
   to `index_store/index.pkl`; a readable summary goes to `index_store/manifest.json`.
5. **Retrieve** - your question is transformed with the same vectorizer and compared
   to every passage using **cosine similarity**. The highest scores win.
6. **Answer** - the best matching sentences are quoted back with the source file name
   and location. If the best score is below the threshold, the app honestly says
   **no relevant answer found** instead of guessing.
7. **Stay updated** - `watcher.py` compares each file's hash/size/modified-time. Any
   change triggers a rebuild, and unchanged files are re-used, not re-processed.

**What is NOT used:** no paid API, no API key, no downloaded model, no GPU. The app
never claims to generate text - answers are extractive quotes from your documents.
        """
    )

    if index is not None and not index.is_empty:
        st.divider()
        st.markdown("**Current index details**")
        st.json(
            {
                "built_at": index.built_at,
                "index_version": index.version,
                "files": index.file_count,
                "passages": index.chunk_count,
                "vocabulary_terms": index.vocabulary_size,
                "words": index.total_words,
                "settings": index.stats,
            }
        )

    st.divider()
    st.markdown("**Project files**")
    st.dataframe(
        [
            {"File": "app.py", "Purpose": "Streamlit web interface: upload, ask, documents, watch tabs."},
            {"File": "config.py", "Purpose": "All settings: folders, chunk size, TF-IDF and answer thresholds."},
            {"File": "document_loader.py", "Purpose": "Reads TXT/MD/PDF and raises clear, specific errors."},
            {"File": "text_processor.py", "Purpose": "Cleans text, splits into overlapping chunks, splits sentences."},
            {"File": "indexer.py", "Purpose": "Builds/loads/saves the TF-IDF index and caches unchanged files."},
            {"File": "qa_engine.py", "Purpose": "Cosine similarity search + extractive answer with sources."},
            {"File": "watcher.py", "Purpose": "Detects added/edited/deleted files; also a terminal watcher."},
            {"File": "tests/", "Purpose": "pytest test cases for every module and the Streamlit UI."},
            {"File": "sample_documents/", "Purpose": "Ready-made TXT, MD and PDF data for testing."},
            {"File": "documents/", "Purpose": "The watched folder. Uploads and your own files live here."},
            {"File": "index_store/", "Purpose": "Saved index (auto-created). Safe to delete to force a rebuild."},
        ],
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Auto-watch fragment (runs while the page is idle)
# ---------------------------------------------------------------------------
@st.fragment(run_every=f"{config.AUTO_REFRESH_SECONDS}s")
def auto_watch_panel() -> None:
    if not st.session_state.get("auto_watch", True):
        return
    watcher: FolderWatcher = st.session_state.get("watcher")
    if watcher is None:
        return

    changes = watcher.poll()
    if changes.has_changes:
        st.session_state.watch_message = changes.summary()
        st.toast("Documents changed - rebuilding the index...", icon="🔄")
        st.rerun(scope="app")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Self Updating Documentation Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    init_session()
    index = ensure_index()

    render_header()
    render_sidebar(index)

    tab_ask, tab_docs, tab_watch, tab_about = st.tabs(
        ["💬 Ask a question", "📄 Documents", "🔄 Auto updates", "ℹ️ How it works"]
    )
    with tab_ask:
        render_ask_tab(index)
    with tab_docs:
        render_documents_tab(index)
    with tab_watch:
        render_watch_tab(index)
    with tab_about:
        render_about_tab(index)

    auto_watch_panel()

    if st.session_state.get("toast_message"):
        st.toast(st.session_state.toast_message, icon="✅")
        st.session_state.toast_message = ""


if __name__ == "__main__":
    main()
