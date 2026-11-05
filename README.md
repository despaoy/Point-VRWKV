# Point-VRWKV

Vision-RWKV6 adapted to point-cloud classification on **PointDA-10** and
**PointDG-3to1**, organized with a compact OpenPoints framework and YAML configs.

## Overview

The model uses FPS/KNN patch encoding, geometric positional embeddings,
six-direction 3D Q-Shift, bidirectional WKV6, channel mixing, mean pooling and a
linear classification head. The default backbone has 12 layers, width 192,
3 heads and 256 tokens from 1024 input points.

ModelNet uses six-axis grid-snake ordering. The other tasks use Morton-xyz.

## Installation

Requirements include Linux, Python 3.10+, a CUDA-compatible PyTorch installation,
the CUDA toolkit with nvcc, and a compatible C++ compiler.

```bash
pip install -r requirements.txt
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

WKV6 and input-FPS extensions compile on first use.

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

Choose a GPU with `CUDA_VISIBLE_DEVICES`. Each task uses its corresponding YAML
file under `cfgs/`. The OpenPoints entry also supports dotted overrides.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py \
  --cfg cfgs/pointdg/D.yaml --data-root data/PDG3to1 \
  --output outputs/ABC_D_seed42 seed=42 sampling_seed=42
```

Optimization uses source-domain train labels. Per-task recipes include
domain-balanced sampling, optional batch-class balancing, smoothing, augmentation,
warmup and cosine decay. Best checkpoints are selected using target OA.
ShapeNet also uses a target-OA threshold stop. Settings are recorded in each run's
`cfg.yaml`.

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

## Results

Historical single-seed best OA from the original experimental runs, with
per-target training recipes and target-selected checkpoints.

| Target | Paper V-RWKV′ OA (%) | Local best OA (%) |
|---|---:|---:|
| ScanNet | 49.52 | 50.1413 |
| ShapeNet | 81.90 | 84.7913 |
| ModelNet | 85.49 | 86.4486 |
| ABC→D | 73.42 | 76.5191 |
| ABD→C | 92.12 | 93.7500 |
| ACD→B | 57.88 | 59.2567 |
| BCD→A | 88.18 | 89.9015 |

This is an independent adaptation of the V-RWKV′ baseline idea. Historical
weights, cached inputs and the ScanNet saved initialization are separate
experimental assets.

## Acknowledgements

The implementation builds on [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV)
and the OpenPoints framework used by
[PointDGMamba](https://github.com/yxltya/PointDGMamba).
Repository organization also follows [OptDE](https://github.com/azuki-miho/OptDE).
Source and license notices are in `THIRD_PARTY_NOTICES.md`.
