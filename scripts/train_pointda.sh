#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Usage: bash scripts/train_pointda.sh scannet /path/PointDA_data outputs/scannet
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MAX_JOBS="${MAX_JOBS:-2}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
target="$1"; root="$2"; output="$3"
shift 3
case "${POINT_VRWKV_OPERATOR:-wkv6}" in
  wkv6) config="cfgs/pointda/$target.yaml" ;;
  bi_wkv) config="cfgs/bi_wkv/pointda/$target.yaml" ;;
  *) printf '%s\n' 'POINT_VRWKV_OPERATOR must be wkv6 or bi_wkv' >&2; exit 2 ;;
esac
python examples/classification/main.py --cfg "$config" --data-root "$root" --output "$output" "$@"
