"""Original Vision-RWKV exponential, normalized bidirectional WKV."""
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXTENSION = None


def _extension():
    global _EXTENSION
    if _EXTENSION is None:
        root = Path(__file__).parent / 'cuda/bi_wkv'
        _EXTENSION = load(
            name='point_vrwkv_original_bi_wkv',
            sources=[str(root / 'bi_wkv.cpp'), str(root / 'bi_wkv_kernel.cu')],
            extra_cuda_cflags=['-O3', '--use_fast_math', '--maxrregcount=60'],
            verbose=False,
        )
    return _EXTENSION


class _BiWKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        ctx.input_dtypes = (w.dtype, u.dtype, k.dtype, v.dtype)
        w, u, k, v = [x.float().contiguous() for x in (w, u, k, v)]
        ctx.save_for_backward(w, u, k, v)
        return _extension().bi_wkv_forward(w, u, k, v)

    @staticmethod
    def backward(ctx, grad_output):
        grads = _extension().bi_wkv_backward(
            *ctx.saved_tensors, grad_output.float().contiguous())
        return tuple(g.to(dtype) for g, dtype in zip(grads, ctx.input_dtypes))


def run_bi_wkv(w, u, k, v):
    if (not k.is_cuda or k.ndim != 3 or v.shape != k.shape
            or k.shape[1] < 32 or k.shape[1] % 32
            or k.shape[2] % 32 or w.shape != (k.shape[2],) or u.shape != w.shape):
        raise ValueError('Original Bi-WKV expects CUDA [B,T,C], T/C multiples of 32, and [C] w/u')
    return _BiWKV.apply(w, u, k, v)
