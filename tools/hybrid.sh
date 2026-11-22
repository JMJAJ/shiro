#!/usr/bin/env bash
# Hybrid research eval on the test box: online NN -> side file -> CM eval with fusion.
#   tools/hybrid.sh <file> <cm_bits> [online.py args...]
# Logs both the NN-only run and the fused CM run to results/runs.jsonl.
# The side file is kept (default ~/xzip-side/<file>_<pid>.side, or $XZIP_SIDE) so the
# CM fusion can be re-run without retraining:  xzip eval <file> --bits N --side <path>
set -euo pipefail
file=$1; bits=$2; shift 2
py=${XZIP_PY:-$HOME/xzip-venv/bin/python}
rev=$(cat .rev 2>/dev/null || echo nogit)
mkdir -p "$HOME/xzip-side"
side=${XZIP_SIDE:-$HOME/xzip-side/$(basename "$file")_$$.side}
note="hybrid $*"
"$py" nn/online.py "$file" --side "$side" --log results/runs.jsonl --note "$note" "$@"
./target/release/xzip eval "$file" --bits "$bits" --side "$side" \
  --log results/runs.jsonl --rev "$rev" --note "$note" 2>/dev/null
echo "side kept: $side"
