#!/usr/bin/env bash
# uv duoc cai tai ~/.local/bin - dam bao tim duoc tu moi shell
PATH="$HOME/.local/bin:$PATH"
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
  [ -z "${KAGGLE_API_TOKEN:-}" ] && unset KAGGLE_API_TOKEN
  [ -z "${KAGGLE_USERNAME:-}" ] && unset KAGGLE_USERNAME
  [ -z "${KAGGLE_KEY:-}" ] && unset KAGGLE_KEY
fi
# Push a notebook folder (kernel-metadata.json + code file) to Kaggle.
# Usage: ./push_kernel.sh notebooks/<folder>
set -euo pipefail
dir="${1:?Usage: push_kernel.sh <notebook_dir>}"
[ -f "$dir/kernel-metadata.json" ] || { echo "ERROR: missing $dir/kernel-metadata.json"; exit 1; }
uv run kaggle kernels push -p "$dir"
