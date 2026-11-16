"""Vision-RWKV SpatialMix, with the existing point-cloud Q-Shift adapter.

Adapted from OpenGVLab/Vision-RWKV vrwkv.py (Apache-2.0).
The original per-channel decay, first-token term, sigmoid receptance and
exponentially normalized Bi-WKV are preserved, including division by T.
"""
import math

import torch
from torch import nn

from openpoints.cpp.bi_wkv import run_bi_wkv
from .qshift import directional_qshift


class PointSpatialMix(nn.Module):
    def __init__(self, embed_dim, depth, layer_id):
        super().__init__()
        ratio = layer_id / max(depth - 1, 1)
        remaining = 1.0 - layer_id / depth
        ramp = torch.arange(embed_dim, dtype=torch.float32) / embed_dim
        decay = -5 + 8 * (torch.arange(embed_dim, dtype=torch.float32)
                          / max(embed_dim - 1, 1)).pow(0.7 + 1.3 * ratio)
        zigzag = ((torch.arange(embed_dim) + 1) % 3 - 1) * 0.5
        self.spatial_decay = nn.Parameter(decay)
        self.spatial_first = nn.Parameter(torch.full((embed_dim,), math.log(0.3)) + zigzag)
        self.spatial_mix_k = nn.Parameter(ramp.pow(remaining).view(1, 1, -1))
        self.spatial_mix_v = nn.Parameter((ramp.pow(remaining) + 0.3 * ratio).view(1, 1, -1))
        self.spatial_mix_r = nn.Parameter(ramp.pow(0.5 * remaining).view(1, 1, -1))
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.receptance = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, tokens, donors):
        shifted = directional_qshift(tokens, donors)
        xk = tokens * self.spatial_mix_k + shifted * (1 - self.spatial_mix_k)
        xv = tokens * self.spatial_mix_v + shifted * (1 - self.spatial_mix_v)
        xr = tokens * self.spatial_mix_r + shifted * (1 - self.spatial_mix_r)
        k, v = self.key(xk), self.value(xv)
        receptance = torch.sigmoid(self.receptance(xr))
        count = tokens.shape[1]
        response = run_bi_wkv(self.spatial_decay / count, self.spatial_first / count, k, v)
        return self.output(receptance * response)
