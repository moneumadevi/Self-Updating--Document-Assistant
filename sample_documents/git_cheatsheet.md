# Git and GitHub Cheat Sheet

Git is a version control system. It records changes to your files so you can go back
to an older version or work with a team. GitHub is a website that hosts Git
repositories online.

## First time setup

```
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

## Create a repository

`git init` creates a new repository in the current folder. `git clone <url>` copies
an existing repository from GitHub to your computer.

## Daily workflow

1. `git status` shows changed files.
2. `git add .` stages all changes.
3. `git commit -m "message"` saves a snapshot with a message.
4. `git push origin main` uploads commits to GitHub.

`git diff` shows the exact lines you changed but did not stage yet.

## Undo mistakes

- `git restore file.py` throws away uncommitted changes in one file.
- `git restore --staged file.py` unstages a file but keeps the edits.
- `git commit --amend -m "new message"` fixes the message of the last commit.
- `git reset --soft HEAD~1` removes the last commit but keeps your changes staged.
- `git reset --hard HEAD~1` deletes the last commit and its changes permanently.

Be careful with `--hard`: it cannot be undone easily.

## Branches

`git branch feature-login` creates a branch, `git switch feature-login` moves to it,
and `git merge feature-login` merges it back into the current branch. A branch lets
you try a new idea without breaking the main code.

## Working with GitHub

`git remote add origin <url>` connects your local repository to GitHub.
`git pull` downloads the latest commits. `git push` uploads yours. If both sides
changed the same lines you get a merge conflict; open the file, keep the correct
lines, remove the conflict markers `<<<<<<<`, `=======`, `>>>>>>>`, then run
`git add` and `git commit`.

## Pull requests

A pull request asks the project owner to review and merge your branch. Write a clear
title, describe what you changed, and link the issue it fixes.

## Ignoring files

Add a `.gitignore` file and list patterns such as `.venv/`, `__pycache__/`, `*.pyc`,
`index_store/` and `.env`. Git will then skip those files, keeping secrets and large
generated files out of the repository.

## Useful extra commands

- `git log --oneline` shows a compact commit history.
- `git stash` temporarily saves uncommitted work.
- `git tag v1.0` marks a release.
