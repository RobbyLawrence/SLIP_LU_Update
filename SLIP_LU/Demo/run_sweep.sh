#!/bin/bash
# run_sweep.sh <bases_root_dir> [csv_out] [max_bases_per_problem]
#
# Iterates over every subdirectory of bases_root_dir that looks like a
# _Bases directory (contains basis_k*_B.txt files) and runs the sdipps
# --sweep mode against the first max_bases_per_problem basis files in
# each.  CSV is appended (single-header at the top).
#
# defaults:
#   csv_out = sdipps_sweep.csv
#   max_bases_per_problem = 2

set -e

if [ $# -lt 1 ]; then
    echo "usage: $0 <bases_root_dir> [csv_out] [max_bases_per_problem]"
    echo "  e.g. $0 . results.csv 2"
    exit 1
fi

ROOT="$1"
CSV="${2:-sdipps_sweep.csv}"
MAX="${3:-2}"

SDIPPS="./bin/sdipps"
if [ ! -x "$SDIPPS" ]; then
    echo "sdipps binary not found at $SDIPPS  (build with 'make bin/sdipps')"
    exit 1
fi

echo "========================================"
echo "[run_sweep] root=$ROOT  csv=$CSV  max_bases_per_problem=$MAX"
echo "========================================"

# Find all _Bases dirs (any directory containing at least one basis_k*_B.txt)
DIRS=$(find "$ROOT" -mindepth 1 -maxdepth 2 -type d -name '*_Bases' | sort)
if [ -z "$DIRS" ]; then
    echo "[run_sweep] no *_Bases directories found under $ROOT"
    exit 1
fi

TOTAL=0
for d in $DIRS; do
    BASES=$(ls "$d"/basis_k*_B.txt 2>/dev/null | sort -V | head -n "$MAX")
    if [ -z "$BASES" ]; then continue; fi
    for b in $BASES; do
        TOTAL=$((TOTAL+1))
        echo ""
        echo "########## [$TOTAL] $b ##########"
        # log everything to stdout so user can tail -f
        "$SDIPPS" --sweep "$b" "$CSV" 2>&1 | grep -v "\[SLIP_LU_factorize\]"
    done
done

echo ""
echo "========================================"
echo "[run_sweep] all done — $TOTAL basis files processed"
echo "[run_sweep] CSV: $CSV"
echo "========================================"
wc -l "$CSV"
