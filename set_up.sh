#!/usr/bin/env bash
set -euo pipefail

# One-shot environment setup for BFM-Zero on a server.
#
# Default usage:
#   bash set_up.sh
#
# Common overrides:
#   SKIP_CUDA_CHECK=1 bash set_up.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"

SKIP_CUDA_CHECK="${SKIP_CUDA_CHECK:-0}"

mkdir -p "${MPLCONFIGDIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed."
  echo "Install uv first, then rerun:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo "[1/3] Syncing Python environment with uv..."
uv sync --frozen

echo "[2/3] Verifying Mini3 motion data..."
uv run --frozen python - <<'PY'
from pathlib import Path

motion_dir = Path("humanoidverse/data/lafan1_mini3")
if not motion_dir.exists():
    raise SystemExit(
        f"Missing Mini3 motion directory: {motion_dir}\n"
        "This repository no longer stores motion datasets in Git/LFS. "
        "Copy lafan1_mini3 into this path before training."
    )

motion_files = sorted(motion_dir.glob("*.pkl"))
if not motion_files:
    raise SystemExit(
        f"No .pkl motion files found under: {motion_dir}\n"
        "Copy the Mini3 .pkl motion files into this directory before training."
    )

print(f"Mini3 motion files: {len(motion_files)}")
PY

if [[ "${SKIP_CUDA_CHECK}" != "1" ]]; then
  echo "[3/3] Checking CUDA/PyTorch..."
  uv run --frozen python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Set SKIP_CUDA_CHECK=1 only if this is intentional.")

print("gpu:", torch.cuda.get_device_name(0))
PY
else
  echo "[3/3] Skipping CUDA check."
fi

echo "Setup complete. Start training with:"
echo "  bash mini3_train.sh"
