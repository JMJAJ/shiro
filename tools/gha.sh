#!/usr/bin/env bash
# Dispatch an A/B sweep on GitHub Actions, wait for it, print the summary table.
#   tools/gha.sh <enwik6|enwik7|enwik8|e30m> <bits> "<ref> [tune]" ["<ref> [tune]" ...]
# Env: ORDER=B,T (decodable order), LABEL=text, NOWAIT=1 (just dispatch).
# The first run is the baseline. Refs must be pushed. Results also land in
# results/gha/<run id>.jsonl.
set -euo pipefail
file=$1; bits=$2; shift 2
runs=$(printf '%s\n' "$@")
since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
gh workflow run sweep.yml -f file="$file" -f bits="$bits" -f runs="$runs" \
  -f order="${ORDER:-}" -f label="${LABEL:-}" >/dev/null
id=""
for _ in $(seq 30); do
  id=$(gh run list --workflow sweep.yml --event workflow_dispatch -L 5 --json databaseId,createdAt \
       -q "[.[] | select(.createdAt >= \"$since\")][-1].databaseId // empty")
  [ -n "$id" ] && break
  sleep 2
done
[ -n "$id" ] || { echo "could not find the dispatched run" >&2; exit 1; }
url=$(gh run view "$id" --json url -q .url)
echo "run $id  $url"
[ -n "${NOWAIT:-}" ] && exit 0
gh run watch "$id" --interval 30 >/dev/null 2>&1 || true
tmp=$(mktemp -d)
gh run download "$id" -n summary -D "$tmp"
mkdir -p results/gha
cp "$tmp/all.jsonl" "results/gha/$id.jsonl"
cat "$tmp/table.md"
rm -rf "$tmp"
