import torch

def square_distance(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.ndim != 3 or target.ndim != 3 or source.shape[-1] != target.shape[-1]:
        raise ValueError("source and target must have shapes [B,N,C] and [B,M,C]")
    distance = (
        source.square().sum(dim=-1, keepdim=True)
        + target.square().sum(dim=-1).unsqueeze(1)
        - 2.0 * source @ target.transpose(1, 2)
    )
    return distance.clamp_min_(0.0)

def index_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if points.ndim != 3 or indices.ndim not in (2, 3):
        raise ValueError("points must be [B,N,C] and indices [B,S] or [B,S,K]")
    if points.shape[0] != indices.shape[0]:
        raise ValueError("batch dimensions do not match")
    batch_shape = (points.shape[0],) + (1,) * (indices.ndim - 1)
    batch = torch.arange(points.shape[0], device=points.device).view(batch_shape)
    batch = batch.expand_as(indices)
    return points[batch, indices]

def deterministic_fps(points: torch.Tensor, count: int) -> torch.Tensor:
    """Deterministic farthest-point sampling with a centroid-based first point."""
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B,N,3]")
    batch, total, _ = points.shape
    if not 1 <= count <= total:
        raise ValueError("FPS count must be in [1, N]")

    indices = torch.empty((batch, count), dtype=torch.long, device=points.device)
    distances = torch.full((batch, total), float("inf"), device=points.device)
    selected = torch.zeros((batch, total), dtype=torch.bool, device=points.device)
    centroid = points.mean(dim=1, keepdim=True)
    first = (points - centroid).square().sum(dim=-1).argmax(dim=1)
    farthest = first
    batch_ids = torch.arange(batch, device=points.device)
    for step in range(count):
        indices[:, step] = farthest
        selected[batch_ids, farthest] = True
        center = points[batch_ids, farthest].unsqueeze(1)
        current = (points - center).square().sum(dim=-1)
        distances = torch.minimum(distances, current)
        if step + 1 < count:
            # A cloud can contain duplicate XYZ rows. Masking selected indices
            # keeps FPS indices unique even when every remaining distance is 0.
            farthest = distances.masked_fill(selected, -1.0).argmax(dim=1)
    return indices

def exact_knn(reference: torch.Tensor, queries: torch.Tensor, count: int) -> torch.Tensor:
    if not 1 <= count <= reference.shape[1]:
        raise ValueError("KNN count must be in [1, reference points]")
    distance = square_distance(queries, reference)
    return torch.argsort(distance, dim=-1, stable=True)[..., :count]
