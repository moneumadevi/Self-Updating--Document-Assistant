# Streamlit Quick Guide

Streamlit is a free Python library that turns a Python script into an interactive web
app. You do not need HTML, CSS or JavaScript to build a student project dashboard.

## Install and run

```
pip install streamlit
streamlit run app.py
```

The app opens at `http://localhost:8501`. Every time you save the Python file,
Streamlit shows a "Rerun" option in the browser, so you can see your changes quickly.

## Rerun model

Streamlit runs your script from top to bottom on every interaction. To keep expensive
work from repeating, use `st.cache_data` for data and `st.cache_resource` for things
like database connections. Anything stored in `st.session_state` survives a rerun.

## Common widgets

- `st.title("My App")` shows a large heading.
- `st.write(x)` prints almost anything: text, numbers, tables, charts.
- `st.text_input("Your question")` creates a single-line text box.
- `st.file_uploader("Upload", type=["txt", "pdf"])` lets users upload files.
- `st.button("Ask")` returns True only in the run where it was clicked.
- `st.slider("Top results", 1, 10, 3)` creates a slider.
- `st.expander("Details")` hides long content until the user opens it.
- `st.download_button("Download", data)` offers a file for download.

## Layout

`st.sidebar` puts widgets in the left sidebar. `st.columns(2)` creates side-by-side
columns. `st.tabs(["Ask", "Documents"])` groups content into tabs. `st.metric`
displays a single statistic with a label.

## Showing progress and messages

`st.success`, `st.info`, `st.warning` and `st.error` show coloured message boxes.
`st.spinner("Searching...")` shows a loading indicator while a slow block runs.

## Auto refresh

`st.fragment(run_every="3s")` reruns only that small function every three seconds.
This is how a page can poll a folder for new documents without reloading everything.

## Deploying

Streamlit Community Cloud can host a public app for free from a GitHub repository.
You need a `requirements.txt` file so the server installs the same packages.

## Tips for a clean project

Keep the interface simple: one clear question box, one answer area and a small
sidebar for settings. Show the source filename with every answer so the user can
trust the result.
