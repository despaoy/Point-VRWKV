# Third-party notices

`openpoints/cpp/cuda/bi_wkv` uses the original normalized Bi-WKV from
OpenGVLab/Vision-RWKV `classification/mmcls_custom/models/backbones/cuda_new`
at commit 0e0e38a6fd4475ae3817e14420ab85fab08d9097. Local adaptations replace deprecated `k.type()` dispatch with `k.scalar_type()`,
correct split-summary distance-minus-one offsets and self-term derivatives, and
use synchronized shared-memory gathering in forward (as in the upstream legacy
kernel). The implementation targets the original normalized exponential formula;
it is not byte-identical to upstream CUDA. `openpoints/models/spatial_mix.py` adapts the corresponding `vrwkv.py`
SpatialMix, substituting the existing 3D Q-Shift for image-grid Q-Shift.
The upstream Apache-2.0 license is included alongside the kernel.

`openpoints/cpp/cuda/wkv6/wkv6_op.cpp` and `wkv6_cuda.cu` are retained from Vision-RWKV,
commit 0e0e38a6fd4475ae3817e14420ab85fab08d9097. The original Apache-2.0
license is supplied alongside these files as `openpoints/cpp/cuda/wkv6/LICENSE`.
The adapted spatial/channel mixing follows that implementation; modified Python
files remove diagnostics and use geometric point-cloud Q-Shift/tokenization.

`openpoints/cpp/cuda/fps` retains the input-FPS implementation used by the existing local
PointDGMamba-derived experiments. Original source headers identify Shaoshuai Shi,
2018, and PointNet++. Unused gather kernels/wrappers/declarations were removed;
the FPS kernel and reduction/tie behavior were retained. These files retain their
original notices. No blanket relicensing of third-party files is implied.

The local PointDGMamba snapshot contains no top-level license. Its inclusion in
past experiments is not proof of redistribution permission. Before public release,
resolve the FPS files' applicable upstream license/permission; the current package
is a local release candidate, not a statement that all third-party rights have
been cleared. Dataset licenses remain with their providers.


The OpenPoints subset is adapted from the local PointDGMamba snapshot:
`utils/config.py`, `utils/registry.py`, `models/build.py`, `dataset/build.py`,
and the AdamW path/interface of `optim/optim_factory.py`. Registry scope trees,
deprecated APIs, unused optimizer families, distributed logging, external trackers
and unrelated models/transforms were removed. The original registry credits MMCV;
the optimizer factory credits timm. Source attribution does not replace upstream
license requirements, and the unresolved snapshot-license note above also applies
to these newly retained framework portions.
