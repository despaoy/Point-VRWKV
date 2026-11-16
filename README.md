# Point-VRWKV

Vision-RWKV adapted to point-cloud classification on **PointDA-10** and
**PointDG-3to1**, organized with a compact OpenPoints framework and YAML configs.

## Overview

Both point-cloud baselines share FPS/KNN patch encoding, geometric positional
embeddings, 3D Q-Shift, channel mixing, mean pooling and a linear classifier.
The default model uses 12 layers, width 192 and 256 tokens from 1024 points.

| Baseline | PointDA-10 presets | PointDG-3to1 presets |
| --- | --- | --- |
| Point-VRWKV6 (WKV6, three heads) | `cfgs/pointda/` | `cfgs/pointdg/` |
| Vision-RWKV Bi-WKV (normalized exponential weighting) | `cfgs/bi_wkv/pointda/` | `cfgs/bi_wkv/pointdg/` |

ModelNet uses six-axis grid-snake ordering. Other targets use Morton-xyz.
Each family keeps its own training recipe. Checkpoints are operator-specific.
Bi-WKV CUDA adaptations are described in `THIRD_PARTY_NOTICES.md`.

## Installation

Requirements include Linux, Python 3.10+, a CUDA-compatible PyTorch installation,
the CUDA toolkit with nvcc, and a compatible C++ compiler.

```bash
pip install -r requirements.txt
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Bi-WKV (or WKV6) and input-FPS extensions compile on first use.

## Datasets

Prepare the benchmarks using the data resources provided by
[PointDGMamba](https://github.com/yxltya/PointDGMamba).
Each domain contains `train_pts.npy`, `train_label.npy`, `test_pts.npy` and
`test_label.npy`. Input features are XYZ coordinates.

```text
data/
├── PointDA_data/
│   ├── modelnet/
│   ├── shapenet/
│   └── scannet/
└── PDG3to1/
    ├── A/   # ModelNet-5
    ├── B/   # ScanNet-5
    ├── C/   # ShapeNet-5
    └── D/   # 3D-FUTURE-Completion
```

Use consistent zero-based labels across domains, in the following class order.

- PointDA-10 — bathtub, bed, bookshelf, cabinet, chair, lamp, monitor, plant, sofa, table.
- PointDG-3to1 — cabinet, chair, lamp, sofa, table.

Evaluation uses FPS1024, unit-sphere normalization and a single view. Original
PointDA runs used cached FPS inputs. Supply the matching unnormalized
`[samples,1024,3]` NPY with `--eval-points`, preserving the test-label object order.

## Training

### PointDA-10

```bash
bash scripts/train_pointda.sh scannet data/PointDA_data outputs/scannet
bash scripts/train_pointda.sh shapenet data/PointDA_data outputs/shapenet
bash scripts/train_pointda.sh modelnet data/PointDA_data outputs/modelnet
```

### PointDG-3to1

```bash
bash scripts/train_pointdg.sh D data/PDG3to1 outputs/ABC_D
bash scripts/train_pointdg.sh C data/PDG3to1 outputs/ABD_C
bash scripts/train_pointdg.sh B data/PDG3to1 outputs/ACD_B
bash scripts/train_pointdg.sh A data/PDG3to1 outputs/BCD_A
```

The shell scripts default to WKV6. Set `POINT_VRWKV_OPERATOR=bi_wkv` to
select the Bi-WKV presets, and choose a GPU with `CUDA_VISIBLE_DEVICES`.

```bash
POINT_VRWKV_OPERATOR=bi_wkv CUDA_VISIBLE_DEVICES=0 \
  bash scripts/train_pointda.sh modelnet data/PointDA_data outputs/modelnet_bi_wkv

CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py \
  --cfg cfgs/bi_wkv/pointdg/D.yaml --data-root data/PDG3to1 \
  --output outputs/ABC_D_bi_wkv seed=42 sampling_seed=42
```

Optimization uses source-domain train labels. Per-task recipes include
domain-balanced sampling, optional batch-class balancing, smoothing, augmentation,
warmup and cosine decay. Best checkpoints are selected using target OA.
Early stopping and evaluation settings are defined by each recipe. Effective
settings are stored in the run's `cfg.yaml`.

## Evaluation and resume

```bash
python evaluate.py --checkpoint outputs/ABC_D/best_target.pt \
  --data-root data/PDG3to1 --output outputs/ABC_D_eval

python examples/classification/main.py --cfg cfgs/pointdg/D.yaml \
  --data-root data/PDG3to1 --output outputs/ABC_D \
  --resume outputs/ABC_D/last.pt
```

Resume with the original configuration and data paths. Outputs include
`best_target.pt`, `last.pt`, training curves, per-sample predictions and confusion
matrices. Evaluation runs in FP32 with a single view.

## Acknowledgements

The implementation builds on [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV)
and the OpenPoints framework used by
[PointDGMamba](https://github.com/yxltya/PointDGMamba).
Repository organization also follows [OptDE](https://github.com/azuki-miho/OptDE).
Source and license notices are in `THIRD_PARTY_NOTICES.md`.
