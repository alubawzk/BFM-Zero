# Repository Guidelines

## Project Structure & Module Organization

`humanoidverse/` is the main Python package. Training and inference entry points live at its top level (`train.py`, `tracking_inference.py`, `goal_inference.py`, and `reward_inference.py`). Agent implementations are under `humanoidverse/agents/`; environments and robot helpers are under `envs/`; simulator adapters are grouped by backend in `simulator/`; shared math, motion, and logging code belongs in `utils/`. Hydra configuration is organized in `humanoidverse/config/`. Robot descriptions and motion datasets live in `humanoidverse/data/`, while example checkpoints and generated inference artifacts are under `model/`. Documentation images belong in `static/images/`.

## Build, Test, and Development Commands

- `git lfs pull` fetches tracked motion datasets and other large artifacts after cloning.
- `uv sync` creates the Python 3.10 environment and installs runtime plus Ruff dependencies.
- `uv run python -m humanoidverse.train` starts the default training pipeline.
- `uv run python -m humanoidverse.tracking_inference --help` shows CLI options; substitute `goal_inference` or `reward_inference` for the other workflows.
- `uv run ruff check .` runs lint and import-order checks. Use `uv run ruff check . --fix` only after reviewing the proposed edits.
- `uv build` verifies that the `humanoidverse` package and bundled config/data files can be packaged.

## Coding Style & Naming Conventions

Use four-space indentation and keep lines within Ruff's 140-character limit. Follow Python conventions: `snake_case` for modules, functions, and variables; `PascalCase` for classes and Pydantic configuration models; `UPPER_SNAKE_CASE` for constants. Add type hints to public interfaces and keep simulator-specific behavior inside its backend directory. Prefer Hydra YAML overrides over hard-coded experiment values, and use two-space indentation in YAML.

## Testing Guidelines

No dedicated automated test suite or coverage threshold is currently configured. Before submitting, run Ruff and execute the affected training or inference entry point with the relevant simulator (MuJoCo is the lightweight option). New tests should go in `tests/`, mirror the package layout, and use names such as `test_reward_inference.py`; add any required test dependency and configuration in the same change.

## Commit & Pull Request Guidelines

Recent history uses short, imperative subjects such as `Add MuJoCo configuration file` and `Ensure output directory is created before saving`. Keep each commit focused. Pull requests should explain the behavior change, list commands run, identify simulator/GPU assumptions, and link related issues. Include screenshots or short clips for viewer/rendering changes and call out new LFS assets, checkpoints, or configuration migrations explicitly. Never commit credentials, local experiment logs, or untracked large binaries; route datasets through Git LFS.
