#!/usr/bin/env bash
set -euo pipefail

# Pack the local uv cache so it can be copied to a weak-network server.
#
# Usage:
#   bash scripts/pack_uv_cache.sh
#   bash scripts/pack_uv_cache.sh /tmp/bfm-zero-uv-cache.tar.gz

output="${1:-/tmp/bfm-zero-uv-cache.tar.gz}"
cache_dir="$(uv cache dir)"

if [[ ! -d "${cache_dir}" ]]; then
  echo "ERROR: uv cache directory does not exist: ${cache_dir}" >&2
  exit 1
fi

mkdir -p "$(dirname "${output}")"

echo "Packing uv cache:"
echo "  source: ${cache_dir}"
echo "  output: ${output}"

tar -C "${cache_dir}" -czf "${output}" .

echo "Done:"
du -h "${output}"
