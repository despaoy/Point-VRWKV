from pathlib import Path
from typing import Any
import torch
from torch.utils.cpp_extension import load
HEAD_SIZE = 64
_EXTENSION = None

def _extension():
    global _EXTENSION
    if _EXTENSION is None:
        root = Path(__file__).parent / 'cuda/wkv6'
        _EXTENSION = load(name='point_vrwkv_wkv6_t256_h64',
            sources=[str(root/'wkv6_op.cpp'), str(root/'wkv6_cuda.cu')],
            extra_cuda_cflags=['-res-usage', '--use_fast_math', '-O3', '-Xptxas', '-O3',
                              '--extra-device-vectorization', '-D_N_=64', '-D_T_=256'],
            verbose=False)
    return _EXTENSION


class _OfficialWKV6(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        extension = _extension()
        batch, tokens, channels = r.shape
        heads = u.shape[0]
        if channels // heads != HEAD_SIZE:
            raise ValueError("Official WKV6 requires head size 64")
        ew = (-torch.exp(w.float())).contiguous()
        r = r.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        u = u.float().contiguous()
        ctx.shape = (batch, tokens, channels, heads)
        ctx.save_for_backward(r, k, v, ew, u)
        output = torch.empty_like(r, dtype=torch.float32)
        extension.forward(batch, tokens, channels, heads, r, k, v, ew, u, output)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        extension = _extension()
        batch, tokens, channels, heads = ctx.shape
        r, k, v, ew, u = ctx.saved_tensors
        grad_output = grad_output.float().contiguous()
        grad_r = torch.empty_like(r)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        grad_w = torch.empty_like(ew)
        grad_u_batch = torch.empty(
            (batch, channels), device=r.device, dtype=torch.float32
        )
        extension.backward(
            batch,
            tokens,
            channels,
            heads,
            r,
            k,
            v,
            ew,
            u,
            grad_output,
            grad_r,
            grad_k,
            grad_v,
            grad_w,
            grad_u_batch,
        )
        grad_u = grad_u_batch.sum(dim=0).view(heads, channels // heads)
        return grad_r, grad_k, grad_v, grad_w, grad_u

def run_wkv6(r, k, v, w, u):
    if not r.is_cuda or not 3 <= r.shape[1] <= 256:
        raise ValueError('WKV6 requires CUDA and 3..256 tokens')
    return _OfficialWKV6.apply(r, k, v, w, u)
