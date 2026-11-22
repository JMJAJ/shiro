#!/usr/bin/env bash
# Hybrid research eval on the test box: online NN -> side file -> CM eval with fusion.
#   tools/hybrid.sh <file> <cm_bits> [online.py args...]
# Logs both the NN-only run and the fused CM run to results/runs.jsonl.
set -euo pipefail
file=$1; bits=$2; shift 2
py=${XZIP_PY:-$HOME/xzip-venv/bin/python}
rev=$(cat .rev 2>/dev/null || echo nogit)
side=/tmp/xzip_$(basename "$file")_$$.side
note="hybrid $*"
"$py" nn/online.py "$file" --side "$side" --log results/runs.jsonl --note "$note" "$@"
./target/release/xzip eval "$file" --bits "$bits" --side "$side" \
  --log results/runs.jsonl --rev "$rev" --note "$note" 2>/dev/null
rm -f "$side"
