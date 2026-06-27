#!/usr/bin/env bash
# Fetch the ONNX model weights svgit's ML layer needs. Weights are large and
# license-encumbered, so they're git-ignored and pulled on demand instead.
#
# Usage:  scripts/fetch-models.sh            # into ./models
#         SVGIT_MODEL_DIR=/path scripts/fetch-models.sh
set -euo pipefail

DEST="${SVGIT_MODEL_DIR:-models}"
mkdir -p "$DEST"

# name  url  sha256
fetch() {
  local name="$1" url="$2" want="$3"
  local out="$DEST/$name"
  if [[ -f "$out" ]] && have_sha "$out" "$want"; then
    echo "✓ $name already present and verified"
    return
  fi
  echo "↓ fetching $name …"
  curl -fSL --retry 3 -o "$out" "$url"
  if ! have_sha "$out" "$want"; then
    echo "✗ $name checksum mismatch — got $(sha "$out"), want $want" >&2
    rm -f "$out"
    exit 1
  fi
  echo "✓ $name verified"
}

sha() { shasum -a 256 "$1" | awk '{print $1}'; }
have_sha() { [[ "$(sha "$1")" == "$2" ]]; }

# u2netp — lightweight (~4.6 MB) salient-object net used for background removal.
fetch "u2netp.onnx" \
  "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx" \
  "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8"

echo "All models ready in $DEST/"
