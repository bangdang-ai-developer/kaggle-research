#!/usr/bin/env bash
# uv duoc cai tai ~/.local/bin - dam bao tim duoc tu moi shell
PATH="$HOME/.local/bin:$PATH"
# Check Kaggle kernel run status (complete/error/running/queued).
# Usage: ./kernel_status.sh <username/slug>
set -euo pipefail
slug="${1:?Usage: kernel_status.sh <username/slug>}"
uv run kaggle kernels status "$slug"
