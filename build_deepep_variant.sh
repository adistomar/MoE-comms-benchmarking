#!/bin/bash
# Build a patched copy of DeepEP next to the original, leaving the original checkout untouched.
# Run from a SINGLE task inside the container (like install_deepep_ngc.sh):
#
#   bash build_deepep_variant.sh <patch> <dst dir>
#   e.g. bash build_deepep_variant.sh patches/deepep_dispatch_total_warps.patch ../DeepEP-warps
#
# The copy is the source tree of $DEEPEP_DIR's HEAD (git archive, tracked files only) with the
# third-party/fmt submodule symlinked from the original, plus the patch. Use it with
# DEEPEP_DIR=<dst dir> (deepep_env.sh then takes the prebuilt fast path).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PATCH="$(realpath "$1")"
DST="$(realpath -m "$2")"
SRC="${DEEPEP_DIR:-$(cd .. && pwd)/DeepEP}"

if [ ! -f "$DST/setup.py" ]; then
    mkdir -p "$DST"
    git -C "$SRC" archive HEAD | tar -x -C "$DST"
    rm -rf "$DST/third-party/fmt"
    ln -s "$SRC/third-party/fmt" "$DST/third-party/fmt"
fi
if (cd "$DST" && git apply --reverse --check "$PATCH" >/dev/null 2>&1); then
    echo "[build_deepep_variant] patch already applied in $DST"
else
    (cd "$DST" && git apply "$PATCH")
    echo "[build_deepep_variant] applied $(basename "$PATCH") in $DST"
fi
DEEPEP_DIR="$DST" bash ./install_deepep_ngc.sh
