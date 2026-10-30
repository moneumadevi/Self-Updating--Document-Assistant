# Python Basics for Beginners

Python is a popular programming language created by Guido van Rossum. It is easy to
read because it uses plain English words and indentation instead of curly braces.

## Installing Python on Windows

1. Download Python from python.org.
2. Run the installer and tick the checkbox **Add python.exe to PATH**.
3. Open Command Prompt and type `python --version`. You should see something like
   `Python 3.12.3`.

## Variables and data types

A variable stores a value. In Python you do not need to declare the type:

```python
name = "Ravi"        # string
age = 21             # integer
height = 5.9         # float
is_student = True    # boolean
```

Common built-in data types are `str`, `int`, `float`, `bool`, `list`, `tuple`,
`dict` and `set`.

## Lists and dictionaries

A list keeps items in order and is written with square brackets: `marks = [78, 91, 65]`.
A dictionary stores key-value pairs and is written with curly braces:
`student = {"name": "Ravi", "branch": "CSE"}`.

You can loop over a list with `for mark in marks:` and read a dictionary value with
`student["name"]`. Use `student.get("city", "unknown")` to avoid a KeyError.

## Functions

A function is a reusable block of code:

```python
def add(a, b):
    return a + b

print(add(2, 3))   # 5
```

Functions make your program shorter and easier to test. Give functions a clear name
and a short docstring explaining what they do.

## Virtual environments

A virtual environment keeps the packages of one project separate from other projects.
Create one with `python -m venv .venv`, activate it on Windows with
`.venv\Scripts\activate`, and install packages using `pip install -r requirements.txt`.

## Reading a text file

```python
with open("notes.txt", "r", encoding="utf-8") as handle:
    text = handle.read()
```

The `with` statement closes the file automatically, even when an error happens.
Always pass `encoding="utf-8"` on Windows so special characters do not break.

## Common beginner errors

- `IndentationError`: the spaces at the start of a line are not consistent.
- `NameError`: you used a variable before creating it, or spelled it differently.
- `ModuleNotFoundError`: the package is not installed in the active environment.

## Why Python for machine learning

Python has free libraries such as NumPy, pandas, scikit-learn and Streamlit. They let
students build real projects without paid tools. Most machine learning tutorials use
Python, so help is easy to find.
