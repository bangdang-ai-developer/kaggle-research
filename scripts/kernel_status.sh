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
# Check Kaggle kernel run status (complete/error/running/queued).
# Usage: ./kernel_status.sh <username/slug>
set -euo pipefail
slug="${1:?Usage: kernel_status.sh <username/slug>}"
uv run kaggle kernels status "$slug"
