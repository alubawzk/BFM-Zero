#!/usr/bin/env bash
set -euo pipefail

# Unpack a uv cache archive created by scripts/pack_uv_cache.sh.
#
# Usage on the server:
#   bash scripts/unpack_uv_cache.sh /home/wzk/bfm-zero-uv-cache.tar.gz

archive="${1:-}"
if [[ -z "${archive}" ]]; then
  echo "Usage: bash scripts/unpack_uv_cache.sh /path/to/bfm-zero-uv-cache.tar.gz" >&2
  exit 2
fi

if [[ ! -f "${archive}" ]]; then
  echo "ERROR: archive does not exist: ${archive}" >&2
  exit 1
fi

cache_dir="$(uv cache dir)"
mkdir -p "${cache_dir}"

echo "Unpacking uv cache:"
echo "  archive: ${archive}"
echo "  target:  ${cache_dir}"

tar -C "${cache_dir}" -xzf "${archive}"

echo "Done."
