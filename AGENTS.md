# instructions

- A repository for mloom: a Python library that excludes unavailable sensors, selects features, and tunes the hyperparameters of a user-chosen model on Parquet data, then saves the fitted Pipeline as a pickle. A thin CLI and an agent Skill (`skills/`) call the library.
- Always respond in Japanese.
- `docs/spec.md` holds the requirements, design decisions (with their reasons) and verified facts. Read the relevant sections before changing code.
  - When you are making a change on a branch and a decision is made or a fact is verified, update `docs/spec.md` in the same change.
  - For investigation, questions, or review requests, do NOT edit `docs/spec.md`. Report findings in your reply (or an Issue comment) instead.
- This repository is in development (0.x); backward compatibility is NOT required.
  - NEVER add or keep compatibility layers, deprecated aliases, or readers for older `run.json` formats.
- Coding rules for every language (keeping the code easy to change): [.agents/docs/coding.md](.agents/docs/coding.md)

## Responsibilities of the library and the Skill

- The Python API is the only place that decides which columns are excluded, how models are evaluated, and what is saved.
- The CLI only turns its arguments into the API's config, calls the API, and saves the result. NEVER put processing logic in the CLI.
- The Skill (`SKILL.md`) only turns the user's request into CLI arguments and explains `run.json`. NEVER make it decide columns, evaluation, or search spaces.

## Working tree

- NEVER edit, create, or delete tracked files on the local `main` branch unless the user explicitly asks for that edit on `main`.
  - Investigation-only requests are read-only: no file edits at all, except scratch files under `tmp/<task-name>/`.
  - If a code or doc change seems necessary, propose it and wait for instructions, or create a branch or worktree first.

## Python

- Always use `uv run -m`.
- NEVER use `python -m`.
- Coding rules: [.agents/docs/python.md](.agents/docs/python.md)
- Running tests: [.agents/docs/pytest.md](.agents/docs/pytest.md)

### Verification commands

- Run lint: `uv run -m ruff check .`
- Run type check: `uv run -m ty check`
- Run tests: `uv run -m pytest --tb=short -q`
- CI runs all three on every pull request to main.

## Github

- Commit: [.agents/docs/commit.md](.agents/docs/commit.md)

## Temporary files

- Create all temporary scripts and investigation files under `tmp/<task-name>/` at the repository root (one subdirectory per task), never directly in `tmp/`.
- When the task is complete, delete only the files you created, then the empty `tmp/<task-name>/` with `rmdir`. NEVER delete `tmp/` itself or other subdirectories: other agents may be using them.
