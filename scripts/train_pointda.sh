#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Usage: bash scripts/train_pointda.sh scannet /path/PointDA_data outputs/scannet
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MAX_JOBS="${MAX_JOBS:-2}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
target="$1"; root="$2"; output="$3"
shift 3
python examples/classification/main.py --cfg "cfgs/pointda/$target.yaml" --data-root "$root" --output "$output" "$@"
