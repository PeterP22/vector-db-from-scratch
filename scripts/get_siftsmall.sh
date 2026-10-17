#!/usr/bin/env bash
# Download SIFT-small (TEXMEX) into data/siftsmall/ and verify its checksum.
set -euo pipefail

URL="https://huggingface.co/datasets/vecdata/siftsmall/resolve/main/siftsmall.tar.gz"
SHA256="987b526d24e749082ba27ee8068003836eb17a61b34f09b4db865750f8a43487"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${1:-$ROOT/data}"
ARCHIVE="$DATA/siftsmall.tar.gz"

mkdir -p "$DATA"

if [ -f "$DATA/siftsmall/siftsmall_base.fvecs" ]; then
    echo "siftsmall already present in $DATA/siftsmall"
    exit 0
fi

if [ ! -f "$ARCHIVE" ]; then
    echo "downloading $URL"
    curl -fL --retry 3 -o "$ARCHIVE.part" "$URL"
    mv "$ARCHIVE.part" "$ARCHIVE"
fi

echo "$SHA256  $ARCHIVE" | sha256sum -c -

tar -xzf "$ARCHIVE" -C "$DATA" --no-same-owner
ls -l "$DATA/siftsmall"
