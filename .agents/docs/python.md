# Python coding

Follow [coding.md](coding.md) first; this file adds what is specific to Python.

## Rules

- Add docstrings for all functions and classes with NumPy style.
- Add type hints for all function parameters and return types.

## Structure

- State each module's responsibility in its module docstring.
- `src/mloom/__init__.py` is a thin, explicit re-export of the public API (`__all__`), so that users can write `mloom.optimize_model`. Keep it to the docstring, the imports of the public names, and `__all__`.
  - When `__all__` changes, update `README.md`, `docs/spec.md` and the tests in the same change.
- Keep every other `__init__.py` to a docstring. NEVER re-export names from them.
- Choose the module for new code from "パッケージ構成" in `docs/spec.md`. When adding a module, add it to that table.

## Lint

- ruff checks the limits of coding.md (C901, PLR0912, PLR0915) with the values in `pyproject.toml`, and fails on a stale `noqa` (RUF100). Existing debt is marked `# noqa: <codes>` on the `def` line.
- ruff also flags unused arguments (ARG; not in tests, where pytest passes fixtures). An argument a caller's signature requires stays, with `# noqa: ARG001 - <why>`.

## Tests

- Before adding a helper or fixture, search the `conftest.py` files under `tests/` and the test file you are editing.
- Share setup through pytest fixtures scoped as narrowly as the tests that use them.
- Make test data with small synthetic Parquet or DataFrames in the test; NEVER read files under `data/`.
