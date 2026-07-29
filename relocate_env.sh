#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
venv_dir="${project_dir}/.venv"
python_dir="${project_dir}/.python-runtime"
old_project_dir="/home/amax/Desktop/robot/BFM-Zero"
old_python_dir="/home/amax/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu"

if [[ ! -d "${venv_dir}" || ! -x "${python_dir}/bin/python3.10" ]]; then
    echo "Error: extract the bundle at the BFM-Zero project root, then run this script." >&2
    exit 1
fi

rm -f "${venv_dir}/bin/python" "${venv_dir}/bin/python3" "${venv_dir}/bin/python3.10"
ln -s "${python_dir}/bin/python3.10" "${venv_dir}/bin/python"
ln -s python "${venv_dir}/bin/python3"
ln -s python "${venv_dir}/bin/python3.10"

PROJECT_DIR="${project_dir}" \
VENV_DIR="${venv_dir}" \
PYTHON_DIR="${python_dir}" \
OLD_PROJECT_DIR="${old_project_dir}" \
OLD_PYTHON_DIR="${old_python_dir}" \
"${python_dir}/bin/python3.10" - <<'PY'
import os
from pathlib import Path

project_dir = Path(os.environ["PROJECT_DIR"])
venv_dir = Path(os.environ["VENV_DIR"])
python_dir = Path(os.environ["PYTHON_DIR"])
replacements = {
    os.environ["OLD_PROJECT_DIR"].encode(): str(project_dir).encode(),
    os.environ["OLD_PYTHON_DIR"].encode(): str(python_dir).encode(),
}

targets = [venv_dir / "pyvenv.cfg"]
targets.extend(path for path in (venv_dir / "bin").iterdir() if path.is_file())
site_packages = venv_dir / "lib" / "python3.10" / "site-packages"
targets.extend(site_packages.glob("__editable__*.py"))
targets.extend(site_packages.glob("*.dist-info/direct_url.json"))

patched = 0
for path in targets:
    try:
        data = path.read_bytes()
    except OSError:
        continue
    updated = data
    for old, new in replacements.items():
        updated = updated.replace(old, new)
    if updated != data:
        path.write_bytes(updated)
        patched += 1

cfg_path = venv_dir / "pyvenv.cfg"
cfg_lines = cfg_path.read_text().splitlines()
cfg_lines = [
    f"home = {python_dir / 'bin'}" if line.startswith("home = ") else line
    for line in cfg_lines
]
cfg_path.write_text("\n".join(cfg_lines) + "\n")
print(f"Relocated environment to {project_dir} ({patched} files patched).")
PY

"${venv_dir}/bin/python" - <<'PY'
import sys
import torch
import humanoidverse

print(f"Python: {sys.version.split()[0]}")
print(f"Environment: {sys.prefix}")
print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
print(f"humanoidverse: {humanoidverse.__path__}")
PY

echo "Environment relocation completed."
echo "Activate with: source \"${venv_dir}/bin/activate\""
