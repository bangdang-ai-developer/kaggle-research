#!/usr/bin/env bash
# uv duoc cai tai ~/.local/bin - dam bao tim duoc tu moi shell
PATH="$HOME/.local/bin:$PATH"
# Download a Kaggle competition or dataset for local EDA.
# Usage: ./fetch_data.sh competition <name>     -> data/raw/<name>/
#        ./fetch_data.sh dataset <owner/slug>   -> data/raw/<owner>---<slug>/
set -euo pipefail
kind="${1:?Usage: fetch_data.sh competition|dataset <id>}"
id="${2:?Usage: fetch_data.sh competition|dataset <id>}"
case "$kind" in
  competition)
    dest="data/raw/$id"
    mkdir -p "$dest"
    uv run kaggle competitions download -c "$id" -p "$dest"
    # competition download tra ve 1 zip lon -> giai nen tai cho
    (cd "$dest" && for z in *.zip *.tar.gz; do [ -e "$z" ] && tar -xf "$z" 2>/dev/null || unzip -o -q "$z" 2>/dev/null || true; done)
    ;;
  dataset)
    dest="data/raw/${id/\//---}"
    mkdir -p "$dest"
    uv run kaggle datasets download "$id" -p "$dest" --unzip
    ;;
  *)
    echo "Unknown kind: $kind (use competition|dataset)"; exit 1 ;;
esac
echo "Downloaded to $dest"
