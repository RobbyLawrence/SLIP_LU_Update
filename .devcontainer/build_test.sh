#!/usr/bin/env bash
# Compile a small standalone program against the SLIP LU library built inside
# this devcontainer. Run from the repo root or from anywhere; paths resolve
# relative to this script.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

if [ ! -f "$REPO/lib/libsliplu.so" ] && [ ! -f "$REPO/lib/libsliplu.a" ]; then
    echo "SLIP LU library not built yet. Run 'make' in $REPO first." >&2
    exit 1
fi

OUT="$HERE/test_link"

gcc -O2 -Wall -Wextra \
    -I"$REPO/SLIP_LU/Include" \
    -I"$REPO/SuiteSparse_config" \
    "$HERE/test_link.c" \
    -L"$REPO/lib" -Wl,-rpath,"$REPO/lib" \
    -lsliplu -lamd -lcolamd -lsuitesparseconfig \
    -lmpfr -lgmp -lm \
    -o "$OUT"

echo "Built: $OUT"
"$OUT"
