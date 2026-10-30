# 📚 Self Updating Documentation Assistant

A beginner-friendly **Machine Learning / NLP project** built with **Python + Streamlit**.
Upload your documentation (TXT, Markdown, PDF), ask a question in plain English, and the
app returns the **best matching passage from your own documents**, together with the
**source file name**.

It also **watches a local `documents/` folder**. When you add, edit or delete a file,
the search index rebuilds automatically and only the changed file is re-processed.

> **No paid API. No API key. No internet needed at run time. No GPU.**
> Everything runs offline on a normal student laptop.

---

## 1. What this project does (short version)

```
TXT / MD / PDF  ──►  text extract  ──►  clean + split into chunks  ──►  TF-IDF vectors
                                                                          │
      your question  ──►  TF-IDF vector  ──►  cosine similarity  ◄─────────┘
                                                   │
                                    best passages + short quoted answer
                                     with the source file name shown
```

Example questions you can ask with the included sample data:

| Question | Answer comes from |
|---|---|
| What is TF-IDF? | `tfidf_and_retrieval.txt` |
| What is cosine similarity? | `tfidf_and_retrieval.txt` |
| How do I undo the last commit? | `git_cheatsheet.md` |
| How do I reset the search index? | `project_faq.pdf` (page 2) |
| How do I run a Streamlit app? | `streamlit_guide.md` |
| How do I create a virtual environment? | `python_basics.md` |

If the documents do not contain the answer, the app **says so honestly** instead of
guessing.

---

## 2. Features (requirement checklist)

| # | Requirement | How it is done in this project |
|---|---|---|
| 1 | Python + VS Code | Pure Python 3.10+, ready-made `.vscode/launch.json` |
| 2 | Streamlit web interface | `app.py` with tabs, sidebar, cards and metrics |
| 3 | Upload TXT / MD / PDF | `st.file_uploader` + `document_loader.py` |
| 4 | Extract / process text | `open()` for text, `pypdf` for PDF, cleaning in `text_processor.py` |
| 5 | Split into searchable chunks | Sliding window: 120 words, 20 words overlap |
| 6 | TF-IDF + cosine similarity | `scikit-learn` `TfidfVectorizer` + `cosine_similarity` |
| 7 | Ask questions | Ask tab / form, example question buttons |
| 8 | Answers with source file | Quoted sentences + filename, page, chunk, match % |
| 9 | Detect added / edited / deleted files | `watcher.py` (size + modified time, then SHA-256) |
| 10 | Rebuild index automatically | Auto-watch toggle (`st.fragment` polling every 3 s) |
| 11 | Store index locally, skip unchanged files | `index_store/index.pkl` + per-file SHA-256 cache |
| 12 | Sample documentation dataset | `sample_documents/` (3 × MD, 1 × TXT, 1 × PDF) |
| 13 | requirements.txt, README, all Python files | This repository |
| 14 | Clear error handling | Empty file, unsupported format, broken PDF, no answer, index errors |
| 15 | Simple attractive UI | Coloured hero header, cards, tabs, badges, metrics |
| 16 | No paid services | All libraries are free and open source |
| 17 | Windows instructions | Section 5 below + `setup_windows.bat` / `run_app.bat` |
| 18 | Test cases + file explanation | `tests/` (96 tests) + Section 3 and Section 9 |
| 19 | Verified code | Tests run with `python -m pytest` (see Section 8) |

---

## 3. Project files and their purpose

```
self-updating-documentation-assistant/
│
├── app.py                  # The Streamlit web page (UI only). Tabs: Ask, Documents,
│                           # Auto updates, How it works. Handles uploads + rebuilds.
├── config.py               # ALL settings in one place: folders, chunk size, TF-IDF
│                           # options, answer thresholds, index version.
├── document_loader.py      # Reads TXT/MD/PDF. Raises clear errors:
│                           # UnsupportedFileTypeError, EmptyDocumentError,
│                           # DocumentReadError.
├── text_processor.py       # Cleans text, removes Markdown noise, normalises word
│                           # endings (install/installing), splits into chunks and
│                           # sentences.
├── indexer.py              # THE CORE: fits TF-IDF, builds the matrix, caches files by
│                           # SHA-256, saves/loads index.pkl + manifest.json, decides
│                           # when a rebuild is needed.
├── qa_engine.py            # Cosine similarity search, keyword coverage guard,
│                           # extractive answer builder, source list, highlighting.
├── watcher.py              # Detects added/edited/deleted documents. Also a terminal
│                           # watcher: python watcher.py
│
├── requirements.txt        # Packages to install
├── conftest.py             # Makes imports work for pytest + shared fixtures
├── setup_windows.bat       # One-time Windows setup (venv + pip install)
├── run_app.bat             # Start the app on Windows
├── README.md               # This file
│
├── .vscode/
│   ├── launch.json         # Press F5 to run the app / watcher / tests
│   └── settings.json       # pytest discovery for the Testing panel
├── .streamlit/config.toml  # Streamlit theme + server settings
│
├── sample_documents/       # Ready-made dataset for testing
│   ├── python_basics.md
│   ├── git_cheatsheet.md
│   ├── streamlit_guide.md
│   ├── tfidf_and_retrieval.txt
│   └── project_faq.pdf     # 2-page PDF
│
├── documents/              # THE WATCHED FOLDER. Ships with copies of the 5 sample
│                           # files so the app works immediately. Uploads go here too.
│
├── index_store/            # Auto-created. Contains:
│   ├── index.pkl           #   vectorizer + TF-IDF matrix + chunk text (binary)
│   └── manifest.json       #   readable summary: files, hashes, chunk counts, build time
│
└── tests/
    ├── test_document_loader.py   # TXT/MD/PDF reading + every error case
    ├── test_text_processor.py    # cleaning, word normalisation, chunking, sentences
    ├── test_indexer.py           # TF-IDF fit, caching, save/load, change detection
    ├── test_qa_engine.py         # retrieval accuracy, no-answer behaviour, sources
    ├── test_watcher.py           # add/edit/delete detection + CLI
    └── test_app.py               # Streamlit UI (AppTest), upload validation
```

`documents/` and `index_store/` are created automatically on first run.
`index_store/` is in `.gitignore` because it is generated data. `documents/` ships
with copies of the sample files so the project runs immediately; anything you
upload there is just a normal file in a normal folder on your computer.

---

## 4. How it works (beginner explanation — Telugu-English mix)

**Step 1 — Text extraction.**
`document_loader.py` file ni chadutundi. TXT/MD ki Python `open()` chalu. PDF ki
`pypdf` library text ni teestundi (page by page). Empty file or wrong format unte
manam clear error isthunnam — app crash avvadu.

**Step 2 — Cleaning.**
Markdown lo `#`, `**`, links, code fences lanti noise ni `text_processor.py`
remove chestundi, kani actual words ni unchutundi. Chinnaga unna heading lines ki
full stop add chestundi, so answer lo heading body lo kalipoyi weird ga chudadu.

**Step 3 — Chunking.**
Oka pedda document ni direct ga compare cheyalem. So 120 words unna chunks ga
split chestam, 20 words overlap tho (so sentence madhyalo cut avvadu). Prathi chunk
ki filename, page number, chunk number attach chestam — ade answer lo "source".

**Step 4 — TF-IDF (idi real ga fit avuthundi).**
`TfidfVectorizer` ni mana chunks meeda **fit** chestam. Fit ante vectorizer mana
documents nunchi **vocabulary** and **IDF weights** learn chestundi: prathi
document lo common ga unna word ki takkuva weight, rare ga unna word ki ekkuva
weight. Taruvata prathi chunk ni numbers vector ga **transform** chestundi.
Neural network ledu, GPU ledu, downloaded model ledu — idi classical NLP maths,
milliseconds lo jarugutundi. (Ee vishayam README lo honest ga chepthunnam, enduku
ante "model train chesam" ani cheppadam correct kaadu.)

Idi kuda cheyagalam: chinnaga suffix stripper (`normalise_term`) valla
"install", "installing", "installed" anni okate term ga treat avuthayi.

**Step 5 — Cosine similarity.**
Question ni same vectorizer tho vector ga marchi, prathi chunk vector tho angle
similarity compute chestam. Score 1 ki daggaraga unte meaning daggaraga untundi.
Highest scores unna chunks = best passages.

**Step 6 — Answer (extractive, no hallucination).**
Best passage lo unna sentences lo, question ki best match ayye sentence ni
teesukuni, next sentence ni context ga add chestam. Answer lo unna prathi word
mee document nunchi ne vasthundi — app kotha text create cheyadu.

**Step 7 — Honest "no answer".**
Rendu conditions: (a) best similarity score chala takkuva (< 0.10), or
(b) question lo unna meaningful words lo sagam kanna takkuva passage lo kanipinchindi.
Ee cases lo "No confident answer found" ani cheptam, plus closest passage ni
chupistam. Wrong answer ivvakunda undadam better.

**Step 8 — Self updating index.**
`watcher.py` prathi 3 seconds ki folder lo files size + modified time compare
chestundi. Change unte index rebuild avuthundi. Rebuild lo **marchina file matrame**
malli chaduvutundi; migatha files chunks cache nunchi reuse avuthayi (SHA-256 hash
compare). Anduke "unnecessary ga process cheyyadam" Avoid chestunnam.

---

## 5. Windows installation — step by step

> Already have Python? Skip to **Step 3**.

### Step 1 — Install Python

1. Open <https://www.python.org/downloads/windows/>.
2. Download **Python 3.12** (64-bit) — any version from 3.10 to 3.13 works.
3. Run the installer and **tick the checkbox “Add python.exe to PATH”** (very important).
4. Click *Install Now*.

Check it worked — open **Command Prompt** and type:

```bat
python --version
```

You should see something like `Python 3.12.3`.

### Step 2 — Install VS Code (recommended)

1. Install **Visual Studio Code** from <https://code.visualstudio.com/>.
2. Open VS Code → Extensions (`Ctrl+Shift+X`) → install **Python** (by Microsoft).
3. (Optional but nice) install **Pylance** and **Even Better TOML**.

### Step 3 — Open the project

In VS Code: **File → Open Folder…** → select the
`self-updating-documentation-assistant` folder
(the folder that contains `app.py` and `requirements.txt`).

### Step 4 — Create a virtual environment

Open the VS Code terminal: **Terminal → New Terminal** (`Ctrl+` `).
Run:

```bat
python -m venv .venv
```

Activate it:

```bat
.venv\Scripts\activate
```

You should now see `(.venv)` at the start of the prompt.

> If PowerShell shows *“running scripts is disabled on this system”*, either use
> **Command Prompt** (cmd) instead of PowerShell, or run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Step 5 — Install the packages

```bat
pip install --upgrade pip
pip install -r requirements.txt
```

This installs Streamlit, scikit-learn, pypdf and pytest. It needs internet **once**.

### Step 6 — Start the app

```bat
streamlit run app.py
```

Your browser opens automatically at **http://localhost:8501**.
To stop the app press `Ctrl+C` in the terminal.

**Shortcuts:** you can also double-click `setup_windows.bat` (first time only) and
then `run_app.bat` (every time) instead of typing the commands.

**In VS Code:** press `F5` and choose **“Run the app (Streamlit)”**.

---

## 6. First run — using the app

1. The sidebar shows **Index status** (documents, passages, vocabulary, words).
2. The 5 sample files are already inside `documents/`, so the index is built on the
   first page load (about 1 second). If you deleted them, click **“Load sample
   documents”** in the sidebar to copy them back.
3. Go to the **💬 Ask a question** tab.
4. Click an example question, or type your own and press **Get answer**.
5. You get:
   * a short **quoted answer** (from your documents),
   * the **source file name**, page and chunk,
   * the **match %** and a confidence label,
   * expandable **evidence passages** with your keywords highlighted.
6. Under the answer you can see **recent questions** (chat history).

### Uploading your own files

**📄 Documents** tab → choose TXT / MD / PDF files → **Add to documents folder**.
Files are saved into `documents/`, validated immediately, and the index rebuilds.
Unsupported formats, empty files and broken PDFs show a **clear red error message**
and are not added.

### Demonstrating the self-updating behaviour (great for a viva)

1. Open the **🔄 Auto updates** tab — leave *Auto-detect changes* ON.
2. In File Explorer, open the `documents/` folder.
3. **Add** a file: create `new_note.txt` with a sentence like
   *“Docker packages an application with its dependencies into a container.”*
   Wait ~3 seconds → the app shows a toast and the sidebar counts increase.
4. Ask *“What is Docker?”* → answered from `new_note.txt`.
5. **Edit** the file (change the sentence) → rebuilt automatically.
6. **Delete** the file → the app notices and removes it from the index.

You can also run the watcher without opening the web page:

```bat
python watcher.py            REM watch forever (Ctrl+C to stop)
python watcher.py --once     REM check once, rebuild only if needed
```

### Where is the index stored?

```
index_store/index.pkl      -> TfidfVectorizer + sparse matrix + chunks (binary)
index_store/manifest.json  -> readable summary, including each file's SHA-256 hash
```

Deleting `index_store/` is safe — the app rebuilds it on the next start.
The **Rebuild index from scratch** button does the same thing.

---

## 7. Settings you can change (`config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE_WORDS` | 120 | Passage size. Smaller = sharper matching, shorter quotes. |
| `CHUNK_OVERLAP_WORDS` | 20 | Overlap between neighbouring chunks. |
| `DEFAULT_MIN_SCORE` | 0.10 | Below this cosine similarity → “no answer found”. |
| `MIN_QUERY_COVERAGE` | 0.5 | At least this share of question keywords must appear in the passage. |
| `DEFAULT_TOP_K` | 3 | Passages shown as evidence (also a sidebar slider). |
| `PHRASE_MATCH_BONUS` | 0.10 | Small rerank boost when the whole question phrase appears in a passage. |
| `INDEX_VERSION` | 5 | Bump this after changing chunk/TF-IDF settings to force a rebuild. |

The sidebar also has sliders for **passages to show** and **minimum similarity**
(lower it if you see “no confident answer” too often).

---

## 8. Test cases — what they check and how to run them

```bat
python -m pytest -v
```

Expected result:

```
96 passed
```

| Test file | What it proves |
|---|---|
| `tests/test_document_loader.py` | TXT/Markdown/PDF text is extracted correctly; page numbers work; **unsupported format, empty file, whitespace-only file, missing file and corrupted PDF all raise clear errors**; uploaded file names are made safe; duplicates do not overwrite. |
| `tests/test_text_processor.py` | Markdown cleaning keeps words; headings get a full stop; chunk size and 20-word overlap are exact; short documents become one chunk; PDF page numbers survive chunking; word normalisation makes install/installing the same token. |
| `tests/test_indexer.py` | The vectorizer is **really fit** (vocabulary and IDF weights exist, matrix shape is chunks × terms); unchanged files are **not re-read**; edited files drop old chunks; deleted files disappear; a broken file is reported but does not break the rest; saved index loads back and is searchable; corrupted/old version indexes are rejected; manifest is readable. |
| `tests/test_qa_engine.py` | A real question finds the right file; the answer is quoted from that passage (no invented words); the PDF page is correct; “What is TF-IDF?” returns the definition; an unrelated question returns **no answer**; empty question and empty index are handled; top-k ordering and deduplication. |
| `tests/test_watcher.py` | Added / edited / deleted files are detected; the watcher ignores untouched files; folder is compared against the saved index; `python watcher.py --once` works from the command line. |
| `tests/test_app.py` | The Streamlit page starts without exceptions, builds/reuses the index, the example-question flow returns an answer with the correct source, unrelated questions show “no answer”, and upload validation rejects PNG, empty files and broken PDFs. |

To run only one file:

```bat
python -m pytest tests/test_qa_engine.py -v
```

In VS Code you can also open the **Testing** (flask) icon in the left sidebar and
click ▶.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Python is not on PATH. Reinstall Python and tick **Add python.exe to PATH**, or use the `py` command instead of `python`. |
| `'streamlit' is not recognized` | The virtual environment is not active. Run `.venv\Scripts\activate` first, or use `python -m streamlit run app.py`. |
| PowerShell blocks activation | Use **Command Prompt**, or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. |
| `ModuleNotFoundError: No module named 'sklearn'` | Packages were installed outside the venv. Activate the venv, then `pip install -r requirements.txt`. |
| `No text found in 'file.pdf'` | The PDF is scanned images, not text. Use OCR (or re-save it as a text PDF), or upload TXT/MD instead. |
| “No confident answer found” | Your wording may not appear in the docs. Use words from the document, or lower **Minimum similarity** in the sidebar. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| App shows 0 documents | Click **Load sample documents**, or put TXT/MD/PDF files into `documents/` (the folder next to `app.py`). |
| Index looks outdated | Sidebar → **Rebuild index from scratch** (or delete `index_store/`). |
| PDF upload says “password-protected” | Remove the PDF password in a PDF viewer and upload again. |

---

## 10. Honesty section — what is *actually* trained here?

This matters for your report/viva, so it is stated plainly:

* ✅ **Yes, the TF-IDF vectorizer is fit on your documents.** `TfidfVectorizer.fit()`
  learns the vocabulary, document frequencies and IDF weights **from your uploaded
  files**. That is the "learning" step, and the learned index is saved to disk.
* ✅ Each chunk is transformed into a vector; retrieval uses **cosine similarity**.
* ❌ **No neural network is trained.** No transformer, no embeddings model, no
  deep-learning fine-tuning. TF-IDF is classical NLP.
* ❌ **No pre-trained model is downloaded** and **no API is called** — no OpenAI,
  no paid service, no API key, no internet at run time.
* ❌ The answer text is **not generated**. It is a quotation of the best matching
  sentences, which is why answers never contain invented facts.

If your report needs an ML algorithm section, describe it as:
**“Unsupervised information retrieval using TF-IDF vectorisation and cosine
similarity, with an extractive answer builder.”**

### Known limitations (be honest in your viva too)

* TF-IDF matches **words, not meaning**. “car” vs “automobile” may not match.
* Spelling mistakes reduce accuracy (no spell correction).
* Scanned/image PDFs need OCR before upload.
* A passage that shares many words but not the meaning can still score high
  (the 50 % keyword coverage rule reduces this, but cannot remove it).
* Very large folders (thousands of files) will make rebuilds slower; the app is
  designed for student-sized documentation sets.

### Possible future improvements

* Sentence embeddings (e.g. a free local model) for synonym matching.
* Chat-style history export, per-document filters, and page preview.
* Support for DOCX and HTML, and OCR for scanned PDFs.
* A nicer score chart, answer feedback buttons, and multi-user accounts.

---

## 11. Quick viva questions and answers

**Q: Where is the “machine learning” part?**
A: The TF-IDF vectorizer is *fit* on the corpus — it learns vocabulary and IDF
weights. Retrieval uses cosine similarity. It is classical NLP, not deep learning,
and I do not claim a neural model is trained.

**Q: Why chunks instead of whole documents?**
A: A long document averages many topics, so its vector matches everything weakly.
A ~120-word chunk covers one idea, so the similarity score is sharper and the
quoted answer stays short.

**Q: Why cosine and not Euclidean distance?**
A: Cosine ignores document length — a short and a long passage about the same
topic still score high. Euclidean distance punishes long passages.

**Q: How do you avoid wrong answers?**
A: Two guards — a minimum cosine score and a keyword-coverage rule. If either
fails, the app says “no confident answer” and shows the closest passage instead.

**Q: How does the index update itself?**
A: The watcher compares file size + modified time against the saved index. On any
change the index rebuilds, re-using the chunks of files whose SHA-256 did not
change.

**Q: What happens with an empty file or a PNG?**
A: `document_loader.py` raises `EmptyDocumentError` / `UnsupportedFileTypeError`.
The UI shows a red message and the bad file is not added to the index.

**Q: Where is data stored?**
A: Everything stays on the local machine — `documents/` for files and
`index_store/` for the index. Nothing is uploaded to any server.

---

## 12. Other operating systems (quick note)

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

The `.bat` files are Windows-only; the Python code is cross-platform.

---

Built as a student project. Free and open libraries only:
Streamlit, scikit-learn, pypdf, pytest.
