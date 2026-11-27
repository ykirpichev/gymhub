#!/bin/bash
# prime-rl reverse-text inside a pool job. "0,0" makes the launcher give inference and the trainer the same GPU.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,0 PRL_OUTPUT_DIR="${POOL_OUTPUTS:-/outputs}/runs" HF_HOME=/tmp/hf
cd /tmp
rl @ /work/rl.toml "$@" 2>&1 | tee /tmp/rl.log
cp /tmp/rl.log "${POOL_OUTPUTS:-/outputs}/rl.log"
