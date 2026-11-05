# Point-VRWKV

A compact Vision-RWKV6 adaptation for point-cloud classification on PointDA-10 and
PointDG-3to1. This is a local reproduction of the **V-RWKV′ baseline idea**, not the
unreleased baseline implementation of the PointDGRWKV authors.

This release uses the **OpenPoints framework subset from PointDGMamba**:
`EasyConfig` YAML loading, `MODELS`/`DATASETS` registries, configuration-driven
model/dataset builders, and the AdamW optimizer-factory interface. Framework
modules are pruned and adapted; this is not the complete upstream framework.
The classification training loop retains this baseline's existing balanced-domain
sampling, augmentation, step-based schedule and target evaluation. It does not
adopt PointDGMamba's model, loss or training defaults.

The model contains FPS/KNN patch encoding, geometric positional embeddings,
parameter-free six-direction 3D Q-Shift, bidirectional WKV6, channel mixing, mean
pooling and a linear classifier. Defaults: 1024 XYZ points, 256 tokens, 32 neighbors
per patch, 12 layers, width 192 and 3 heads. There is no AGT-Shift, CD-KDA,
distillation, domain discriminator, expert head, PartField or output ensemble.
ModelNet additionally uses a six-axis grid-snake serialization schedule; other
presets use Morton-xyz. These are distinct disclosed baseline configurations.

## Install

Use Linux, Python 3.10+, a CUDA-compatible PyTorch installation, the CUDA toolkit
(including nvcc), and a compatible C++ compiler. Install PyTorch for your GPU first:

```bash
pip install -r requirements.txt
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The first actual run builds the bundled WKV6 and input-FPS extensions. There is no
CPU training backend and no requirement for mmcv, mmcls, Mamba, EMD or SciPy.
The token-center FPS and preprocessing FPS have different historical algorithms;
they are deliberately kept separate. WKV runs in FP32 even during BF16 training.

## Data

Obtain the prepared benchmarks from their original providers. Supply the directory
containing the domain directories, each with `train_pts.npy`, `train_label.npy`,
`test_pts.npy`, `test_label.npy`. Only XYZ is used. Labels must already follow the
same class mapping in every domain; the loader does not guess or remap labels.

```text
PointDA_data/               PDG3to1/
  modelnet/                  A/  # ModelNet-5
  shapenet/                  B/  # ScanNet-5
  scannet/                   C/  # ShapeNet-5
                             D/  # 3D-FUTURE-Completion
```

PointDA classes: bathtub, bed, bookshelf, cabinet, chair, lamp, monitor, plant,
sofa, table. PointDG classes: cabinet, chair, lamp, sofa, table. IDs start at zero.
Expected target counts: ModelNet 856, ShapeNet 2492, ScanNet 1769; A/B/C/D
406/1426/2192/971. Counts alone do not establish semantic correctness or object order.

Evaluation defaults to raw test XYZ, original input-FPS1024 if needed, then unit
sphere normalization and yaw0. **Historical PointDA results used an existing FPS
cache.** To reproduce that input protocol, explicitly pass `--eval-points` with
the trusted cache exported to an unnormalized `[samples,1024,3]` NPY in exactly
the label/object order. Do not silently substitute a different cache or pick an
input protocol because it scores higher. Pickle caches are not loaded here.
The evaluated normalized inputs and labels are saved in `evaluation_inputs.npz`.

## Train

```bash
bash scripts/train_pointda.sh scannet /path/PointDA_data outputs/scannet
bash scripts/train_pointda.sh shapenet /path/PointDA_data outputs/shapenet
bash scripts/train_pointda.sh modelnet /path/PointDA_data outputs/modelnet
bash scripts/train_pointdg.sh D /path/PDG3to1 outputs/pdg_D
# Other five-class targets: C, B, A (each trains on the other three domains).
```

Append `--eval-points /path/trusted_test_fps1024.npy` when needed. Each invocation
starts one task on the selected GPU (`CUDA_VISIBLE_DEVICES`, default 0). There is
no server scheduler or automatic parallel launch. A new run requires a new output
directory. Data paths are arguments, never machine-specific constants.

Presets include historical training techniques: equal-domain batches, optional
inverse batch-class-count loss weights, smoothing, augmentation, warmup and cosine
decay. Class balancing is `mean(CE_i / batch_count[label_i])`, including its loss
scale change. Generic augmentation is yaw/scale, local deletion/refill, jitter,
point dropout and translation; standard/mild use yaw/scale/jitter and
FPS1200->random1024. Source optimization only reads source **train** labels.

| Target | Batch | Steps | LR / warmup | WD | Smoothing | Balance | Augment / precision |
|---|---:|---:|---|---|---:|---|---|
| ScanNet | 24 | 4000 | 3e-4 / 1000 | 1e-4, all | .1 | no | generic / FP32 |
| ShapeNet | 24 | up to 20000 | 3e-4 / 1000 | .05, matrix | 1/3 | no | mild / FP32 |
| ModelNet | 192 | 6000 | 3e-4 / 125 | .05, matrix | 1/3 | yes | standard / BF16 + recompute |
| PointDG four tasks | 24 | 20000 | 3e-4 / 1000 | .05, matrix | .375 | yes | standard / FP32 |

All schedules decay to 1e-5. Matrix-only WD exempts 1D and bias parameters.
Seeds: ScanNet3442 (sampling3441), ShapeNet3443, ModelNet/PointDG3444.
ShapeNet retains its historical target threshold stop at 81.9%; other presets end
at their configured budget. Target checks occur each balanced source epoch and
at the endpoint; ScanNet also every500 steps. No source-test/whole_source training.

## Checkpoints and evaluation

```bash
python examples/classification/main.py --cfg cfgs/pointdg/D.yaml --data-root /path/PDG3to1 \
  --output outputs/pdg_D --resume outputs/pdg_D/last.pt
python evaluate.py --checkpoint outputs/pdg_D/best_target.pt \
  --data-root /path/PDG3to1 --output outputs/pdg_D_eval
```

Resume uses the original configuration, data paths and RNG/optimizer/stream state;
do not change the schedule while claiming an exact resume. Training saves
`initial.pt`, `best_target.pt`, `last.pt`, training/evaluation curves, full
predictions and confusion matrices. Evaluation is FP32, single-view. Only load
checkpoints you trust; release checkpoints use the restricted PyTorch loader.

The presets explicitly permit **target-selected exploration**: target accuracy
selects the best checkpoint and the ShapeNet threshold stop, but target labels
never enter the training loss. These are not source-selected formal DG results.

## Historical reference results

| Target | Paper V-RWKV′ OA | Local historical best OA |
|---|---:|---:|
| ScanNet | 49.52 | 50.1413 |
| ShapeNet | 81.90 | 84.7913 |
| ModelNet | 85.49 | 86.4486 |
| ABC→D | 73.42 | 76.5191 |
| ABD→C | 92.12 | 93.7500 |
| ACD→B | 57.88 | 59.2567 |
| BCD→A | 88.18 | 89.9015 |

These are prior target-selected single-seed best scores from different recipes,
not multi-seed means or new results from this cleaned release. Paper-exact
architecture equivalence is unconfirmed. The ScanNet run reused a saved untrained
initial state; this release seeds scratch initialization instead. Historical
PointDA caches and weights are not bundled. Do not claim bit-for-bit reproduction
of those scores without the actual weights, inputs and configuration.

Validated after the OpenPoints migration with PyYAML 6.0.3. All seven YAML
configurations preserve the preceding preset values. Dotted overrides, registry
construction, the upstream model-builder interface and Python syntax pass local
checks. Numerical model and CUDA execution validation is pending because the
packaging environment lacks PyTorch and CUDA.

## Sources

- [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV): WKV6 and model basis;
  upstream commit `0e0e38a6fd4475ae3817e14420ab85fab08d9097`.
- [PointDGMamba](https://github.com/yxltya/PointDGMamba): benchmarks and training/FPS references.
- [OptDE](https://github.com/azuki-miho/OptDE): reference for compact research repository organization.

See `THIRD_PARTY_NOTICES.md` for source and license scope. No data or pretrained
weights are included. This repository is an independent adaptation, not an
official PointDGRWKV release.

## YAML configuration and OpenPoints entry

Seven task configurations are in `cfgs/pointda/*.yaml` and
`cfgs/pointdg/*.yaml`; there are no JSON task configuration files.
Models are instantiated with `model.NAME: PointVRWKV`, datasets with
`dataset.NAME: SourceStream`, and optimizers through `optimizer.NAME: adamw`.
`SourceStream` is the registered resumable domain-balanced source stream, rather
than upstream's default shuffled DataLoader; this preserves the approved sampler.
The same registered model builder is used for training and evaluation.

```bash
python examples/classification/main.py --cfg cfgs/pointdg/D.yaml \
  --data-root /path/PDG3to1 --output outputs/pdg_D
# Optional OpenPoints-style dotted configuration overrides:
python examples/classification/main.py --cfg cfgs/pointdg/D.yaml \
  --data-root /path/PDG3to1 --output outputs/pdg_D_seed42 seed=42 sampling_seed=42
```

The fully resolved settings are saved to `cfg.yaml` in each run. Metrics and
prediction metadata remain JSON/JSONL, which are result records, not task configs.
Checkpoints from the preceding independent-JSON release are not resume-compatible
with the new configuration schema. Model parameter names have not been changed.
Post-migration numerical CUDA validation remains pending.
