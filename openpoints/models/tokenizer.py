from dataclasses import dataclass
import torch
from torch import nn
from openpoints.cpp.pointops import deterministic_fps, exact_knn, index_points

@dataclass(frozen=True)
class PointTokenBundle:
    tokens: torch.Tensor
    centers: torch.Tensor



class PointPatchEncoder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=1),
        )
        self.second = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, embed_dim, kernel_size=1),
        )

    def forward(self, neighborhoods: torch.Tensor) -> torch.Tensor:
        if neighborhoods.ndim != 4 or neighborhoods.shape[-1] != 3:
            raise ValueError("neighborhoods must have shape [B,T,K,3]")
        features = self.first(neighborhoods.permute(0, 3, 1, 2))
        global_features = features.amax(dim=-1, keepdim=True).expand_as(features)
        features = self.second(torch.cat((features, global_features), dim=1))
        return features.amax(dim=-1).transpose(1, 2).contiguous()

class PointPatchTokenizer(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        group_size: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.group_size = group_size
        self.encoder = PointPatchEncoder(embed_dim)
        hidden = max(64, embed_dim // 2)
        self.position = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, points: torch.Tensor) -> PointTokenBundle:
        if points.ndim != 3 or points.shape[-1] < 3:
            raise ValueError("points must have shape [B,N,3+] ")
        xyz = points[..., :3].contiguous()
        if xyz.shape[1] < max(self.num_tokens, self.group_size):
            raise ValueError("input point count is smaller than tokenizer requirements")
        center_indices = deterministic_fps(xyz, self.num_tokens)
        centers = index_points(xyz, center_indices)
        group_indices = exact_knn(xyz, centers, self.group_size)
        neighborhoods = index_points(xyz, group_indices) - centers.unsqueeze(2)
        tokens = self.encoder(neighborhoods) + self.position(centers)
        return PointTokenBundle(
            tokens=tokens,
            centers=centers,
        )
