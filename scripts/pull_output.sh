#!/usr/bin/env bash
# uv duoc cai tai ~/.local/bin - dam bao tim duoc tu moi shell
PATH="$HOME/.local/bin:$PATH"
# Pull the latest output files of a Kaggle kernel into outputs/<slug>/.
# Usage: ./pull_output.sh <username/slug>
set -euo pipefail
slug="${1:?Usage: pull_output.sh <username/slug>}"
dest="outputs/${slug##*/}"
mkdir -p "$dest"
uv run kaggle kernels output "$slug" -p "$dest"
echo "Output saved to $dest"
