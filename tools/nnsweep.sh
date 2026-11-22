#!/usr/bin/env bash
# Sequential NN-only runs on one file; waits for any running nn/online.py first.
#   tools/nnsweep.sh <file> "<args1>" "<args2>" ...
# Each run is logged to results/runs.jsonl (kind "nn", note "nnsweep <args>").
set -u
file=$1; shift
py=${XZIP_PY:-$HOME/xzip-venv/bin/python}
while ps aux | grep -q '[n]n/online.py'; do sleep 10; done
for a in "$@"; do
  # shellcheck disable=SC2086
  r=$("$py" nn/online.py "$file" --log results/runs.jsonl --note "nnsweep $a" $a 2>&1 | grep '^NN-only')
  echo "$r  |  ${a:-(default)}"
done
