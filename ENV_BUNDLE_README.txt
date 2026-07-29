BFM-Zero portable environment bundle

Platform:
  Linux x86_64
  Python 3.10.20
  PyTorch 2.7.0+cu128
  CUDA runtime 12.8
  Isaac Sim 4.5.0 / Isaac Lab 2.0.2

Requirements on the remote server:
  - Linux x86_64
  - A glibc version compatible with the source machine (Ubuntu 22.04,
    glibc 2.35) and a compatible NVIDIA driver
  - The archive must be extracted directly into the BFM-Zero project root

Usage:
  tar --zstd -xf bfm-zero-env-linux-x86_64-py310-cu128.tar.zst -C /path/to/BFM-Zero
  cd /path/to/BFM-Zero
  bash relocate_env.sh
  source .venv/bin/activate

The relocation script updates the Python link, virtual-environment metadata,
entry-point shebangs, activation scripts, and the editable project install.
