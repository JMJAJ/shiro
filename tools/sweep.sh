#!/usr/bin/env bash
# Run several XZIP_TUNE configs in parallel on one file; print a sorted table.
#   tools/sweep.sh <file> <bits> "<spec1>" "<spec2>" ...      (spec "" = defaults)
# Every run is appended to results/runs.jsonl with note "sweep".
set -u
file=$1; bits=$2; shift 2
rev=$(cat .rev 2>/dev/null || echo nogit)
tmp=$(mktemp -d)
i=0
for spec in "$@"; do
  ( XZIP_TUNE="$spec" ./target/release/xzip eval "$file" --bits "$bits" \
      --log results/runs.jsonl --rev "$rev" --note "sweep" 2>/dev/null \
      | awk -v s="$spec" '/^bytes/{print $2 "\t" s}' > "$tmp/$i" ) &
  i=$((i+1))
done
wait
cat "$tmp"/* | sort -n | awk -F'\t' 'NR==1{b=$1} {printf "%10d  %+7.3f%%  %s\n", $1, ($1-b)*100/b, ($2==""?"(default)":$2)}'
rm -rf "$tmp"
