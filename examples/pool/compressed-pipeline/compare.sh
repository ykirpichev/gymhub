#!/bin/sh
# Every compression on one machine (two processes over loopback); one JSON line per run in results.jsonl.
# Usage: sh compare.sh [extra pipeline.py arguments]
set -eu
cd "$(dirname "$0")"
: > results.jsonl
for mode in none bf16 int8 "int8 --compress-grad topk:0.1" "topk:0.25" "topk:0.1"; do
  # shellcheck disable=SC2086
  python3 pipeline.py --compress $mode "$@" | tail -n 1 | tee -a results.jsonl
done
