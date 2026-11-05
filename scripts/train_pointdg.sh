#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Usage: bash scripts/train_pointdg.sh D /path/PDG3to1 outputs/pdg_D
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MAX_JOBS="${MAX_JOBS:-2}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
target="$1"; root="$2"; output="$3"
shift 3
python examples/classification/main.py --cfg "cfgs/pointdg/$target.yaml" --data-root "$root" --output "$output" "$@"
