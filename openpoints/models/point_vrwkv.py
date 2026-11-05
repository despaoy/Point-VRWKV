from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from openpoints.cpp.wkv6 import run_wkv6
from openpoints.utils import EasyConfig
from .build import MODELS
from .tokenizer import PointPatchTokenizer
from .qshift import directional_qshift, plan_directional_donors, remap_directional_donors
from .serialization import AXIS_ORDERS, gather_tokens, morton_permutation, serialize

class PointSpatialMixV6(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, depth: int, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self._init_time_parameters()
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.receptance = nn.Linear(embed_dim, embed_dim, bias=False)
        self.gate = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ln_x = nn.GroupNorm(num_heads, embed_dim, eps=1e-05)

    def _init_time_parameters(self) -> None:
        with torch.no_grad():
            ratio_0_to_1 = self.layer_id / max(self.depth - 1, 1)
            ratio_1_to_almost0 = 1.0 - self.layer_id / self.depth
            channel_ramp = torch.arange(self.embed_dim, dtype=torch.float32) / self.embed_dim
            channel_ramp = channel_ramp.view(1, 1, self.embed_dim)
            base = torch.pow(channel_ramp, ratio_1_to_almost0)
            self.time_maa_x = nn.Parameter(1.0 - base)
            self.time_maa_w = nn.Parameter(1.0 - base)
            self.time_maa_k = nn.Parameter(1.0 - base)
            self.time_maa_v = nn.Parameter(1.0 - (base + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(channel_ramp, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(channel_ramp, 0.5 * ratio_1_to_almost0))
            mix_extra = 32
            self.time_maa_w1 = nn.Parameter(torch.empty(self.embed_dim, mix_extra * 5).uniform_(-0.0001, 0.0001))
            self.time_maa_w2 = nn.Parameter(torch.empty(5, mix_extra, self.embed_dim).uniform_(-0.0001, 0.0001))
            decay = torch.empty(self.embed_dim)
            for channel in range(self.embed_dim):
                fraction = channel / max(self.embed_dim - 1, 1)
                decay[channel] = -6 + 5 * fraction ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay.view(1, 1, self.embed_dim))
            decay_extra = 64
            self.time_decay_w1 = nn.Parameter(torch.empty(self.embed_dim, decay_extra).uniform_(-0.0001, 0.0001))
            self.time_decay_w2 = nn.Parameter(torch.empty(decay_extra, self.embed_dim).uniform_(-0.0001, 0.0001))
            bonus = torch.empty(self.embed_dim)
            for channel in range(self.embed_dim):
                zigzag = ((channel + 1) % 3 - 1) * 0.1
                bonus[channel] = ratio_0_to_1 * (1.0 - channel / max(self.embed_dim - 1, 1)) + zigzag
            self.time_faaaa = nn.Parameter(bonus.view(self.num_heads, self.embed_dim // self.num_heads))

    def _project(self, tokens: torch.Tensor, donors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch, token_count, channels = tokens.shape
        shifted_delta = directional_qshift(tokens, donors) - tokens
        dynamic_input = tokens + shifted_delta * self.time_maa_x
        dynamic = torch.tanh(dynamic_input @ self.time_maa_w1)
        dynamic = dynamic.view(batch * token_count, 5, -1).transpose(0, 1)
        dynamic = torch.bmm(dynamic, self.time_maa_w2).view(5, batch, token_count, channels)
        mix_w, mix_k, mix_v, mix_r, mix_g = dynamic.unbind(dim=0)
        xw = tokens + shifted_delta * (self.time_maa_w + mix_w)
        xk = tokens + shifted_delta * (self.time_maa_k + mix_k)
        xv = tokens + shifted_delta * (self.time_maa_v + mix_v)
        xr = tokens + shifted_delta * (self.time_maa_r + mix_r)
        xg = tokens + shifted_delta * (self.time_maa_g + mix_g)
        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g_raw = self.gate(xg)
        g_effective = F.silu(g_raw)
        dynamic_decay = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + dynamic_decay
        return (r, k, v, g_effective, w)

    def forward(self, tokens: torch.Tensor, donors: torch.Tensor) -> torch.Tensor:
        r, k, v, g_effective, w = self._project(tokens, donors)
        response = run_wkv6(r, k, v, w, self.time_faaaa)
        batch, token_count, channels = response.shape
        response = self.ln_x(response.view(batch * token_count, channels))
        response = response.view(batch, token_count, channels)
        return self.output(response * g_effective)

class PointChannelMixV6(nn.Module):

    def __init__(self, embed_dim: int, depth: int, layer_id: int, hidden_rate: int) -> None:
        super().__init__()
        ratio = 1.0 - layer_id / depth
        ramp = torch.arange(embed_dim, dtype=torch.float32) / embed_dim
        ramp = ramp.view(1, 1, embed_dim)
        self.spatial_mix_k = nn.Parameter(torch.pow(ramp, ratio))
        self.spatial_mix_r = nn.Parameter(torch.pow(ramp, ratio))
        hidden = hidden_rate * embed_dim
        self.key = nn.Linear(embed_dim, hidden, bias=False)
        self.receptance = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(hidden, embed_dim, bias=False)

    def forward(self, tokens: torch.Tensor, donors: torch.Tensor) -> torch.Tensor:
        shifted = directional_qshift(tokens, donors)
        key_input = tokens * self.spatial_mix_k + shifted * (1.0 - self.spatial_mix_k)
        receptance_input = tokens * self.spatial_mix_r + shifted * (1.0 - self.spatial_mix_r)
        key = torch.relu(self.key(key_input)).square()
        value = self.value(key)
        return torch.sigmoid(self.receptance(receptance_input)) * value

class PointVRWKV6Block(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, depth: int, layer_id: int, hidden_rate: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(embed_dim)
        self.att = PointSpatialMixV6(embed_dim, num_heads, depth, layer_id)
        self.ffn = PointChannelMixV6(embed_dim, depth, layer_id, hidden_rate)

    def forward(self, tokens: torch.Tensor, donors: torch.Tensor) -> torch.Tensor:
        if self.layer_id == 0:
            tokens = self.ln0(tokens)
        spatial_delta = self.att(self.ln1(tokens), donors)
        tokens = tokens + spatial_delta
        tokens = tokens + self.ffn(self.ln2(tokens), donors)
        return tokens

@MODELS.register_module()
class PointVRWKV(nn.Module):

    def __init__(self, num_classes, num_tokens, group_size, embed_dim, num_heads,
                 depth, hidden_rate, order, recompute) -> None:
        super().__init__()
        config = EasyConfig()
        config.update(dict(num_classes=num_classes, num_tokens=num_tokens,
                           group_size=group_size, embed_dim=embed_dim, num_heads=num_heads,
                           depth=depth, hidden_rate=hidden_rate, order=order, recompute=recompute))
        self.config = config
        if config.num_heads < 1 or config.embed_dim != config.num_heads*64:
            raise ValueError('The WKV6 kernel uses 64 channels per head')
        if not 3 <= config.num_tokens <= 256 or config.depth < 1:
            raise ValueError('Expected 3..256 tokens and positive depth')
        self.tokenizer = PointPatchTokenizer(config.num_tokens, config.group_size, config.embed_dim)
        self.layers = nn.ModuleList([PointVRWKV6Block(config.embed_dim, config.num_heads, config.depth, layer_id, config.hidden_rate) for layer_id in range(config.depth)])
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        self.apply(self._initialize_module)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, points):
        bundle = self.tokenizer(points)
        canonical_donors = plan_directional_donors(bundle.centers)
        if self.config.order == 'six_axis_snake':
            serials = [serialize(bundle.centers, axis) for axis in AXIS_ORDERS]
        else:
            serials = [morton_permutation(bundle.centers, self.config.order)]
        donors = [remap_directional_donors(canonical_donors, s) for s in serials]
        tokens = bundle.tokens
        if len(serials) == 1:
            tokens = gather_tokens(tokens, serials[0].permutation)
        for i, layer in enumerate(self.layers):
            j = i % len(serials)
            if len(serials) > 1:
                tokens = gather_tokens(tokens, serials[j].permutation)
            if self.config.recompute and self.training:
                tokens = checkpoint(layer, tokens, donors[j], use_reentrant=False)
            else:
                tokens = layer(tokens, donors[j])
            if len(serials) > 1:
                tokens = gather_tokens(tokens, serials[j].inverse_permutation)
        return self.head(self.ln1(tokens).mean(dim=1))
