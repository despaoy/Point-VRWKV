from pathlib import Path
import torch
from torch.utils.cpp_extension import load
_FPS = None

def fps(points, count):
    global _FPS
    if _FPS is None:
        root = Path(__file__).parent / 'cuda/fps'
        _FPS = load(name='point_vrwkv_input_fps', sources=[str(root/x) for x in
                    ('binding.cpp', 'sampling.cpp', 'sampling_gpu.cu')],
                    extra_cflags=['-O2'], extra_cuda_cflags=['-O2'], verbose=False)
    assert points.is_cuda and points.dtype == torch.float32
    b, n, _ = points.shape
    assert 1 <= count <= n
    indices = torch.zeros((b, count), dtype=torch.int32, device=points.device)
    distance = torch.full((b, n), 1e10, device=points.device)
    _FPS.fps(b, n, count, points.contiguous(), distance, indices)
    return indices.long()
