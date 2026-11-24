#!/usr/bin/env bash
# CM cost of the block-interleaved order a real hybrid decoder must use.
#   tools/decodable.sh <file> <cm_bits> <B> [T] [extra SHIRO_TUNE]
# Runs, in parallel: file order (reference), naive interleaved, interleaved with
# per-stream history, and per-stream history + per-stream indirect table.
# Prints one line per variant with bytes and the change vs file order.
set -euo pipefail
file=$1; bits=$2; B=$3; T=${4:-256}; extra=${5:-}
py=${SHIRO_PY:-$HOME/xzip-venv/bin/python}
mkdir -p "$HOME/xzip-side"
order="$HOME/xzip-side/$(basename "$file")_B${B}_T${T}.order"
[ -f "$order" ] || "$py" tools/schedule.py "$file" "$order" --B "$B" --T "$T"
tmp=$(mktemp -d)
run() {  # name tune args...
  local name=$1 tune=$2; shift 2
  SHIRO_TUNE="$tune" ./target/release/shiro eval "$file" --bits "$bits" "$@" 2>/dev/null \
    | awk -v n="$name" '/^bytes/{print n "\t" $2}' > "$tmp/$name"
}
t0="sf=0,ind2s=0${extra:+,$extra}"
run 1-file "$t0" &
run 2-naive "$t0" --order "$order" &
run 3-streams "$t0" --order "$order" --streams "$B" &
run 4-streams+ind2s "$t0,ind2s=1" --order "$order" --streams "$B" &
wait
cat "$tmp"/* | sort | awk -F'\t' 'NR==1{b=$2} {printf "%-18s %10d  %+6.2f%%\n", substr($1,3), $2, ($2-b)*100/b}'
rm -rf "$tmp"
