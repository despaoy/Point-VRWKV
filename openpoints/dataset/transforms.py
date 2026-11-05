"""Point-cloud augmentation with explicit random generators."""
import math
import torch

def random_yaw(points, generator):
    """Rotate each object around Y, retaining its point identities and lengths."""
    angle = torch.empty((len(points), 1)).uniform_(0, 2 * math.pi, generator=generator)
    cosine, sine = (angle.cos(), angle.sin())
    output = points.clone()
    output[:, :, 0] = points[:, :, 0] * cosine + points[:, :, 2] * sine
    output[:, :, 2] = points[:, :, 2] * cosine - points[:, :, 0] * sine
    return output

def local_delete_mask(points, generator, fraction=0.2):
    """Delete a fixed count of nearest points to one random anchor per object."""
    batch, count, _ = points.shape
    delete_count = min(count - 1, max(1, round(count * fraction)))
    anchors = torch.randint(count, (batch,), generator=generator)
    anchor_xyz = points[torch.arange(batch), anchors]
    distances = (points - anchor_xyz[:, None]).square().sum(-1)
    ids = distances.topk(delete_count, dim=1, largest=False, sorted=False).indices
    mask = torch.zeros((batch, count), dtype=torch.bool)
    return mask.scatter_(1, ids, True)

def delete_and_refill(points, deleted, generator):
    """Keep every survivor once, then sample survivors with replacement to N."""
    batch, count, channels = points.shape
    counts = deleted.sum(1)
    if deleted.shape != (batch, count) or deleted.dtype != torch.bool:
        raise ValueError('deleted must be a boolean [B,N] mask')
    if not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError('Batched refill requires equal deletion counts')
    removed = int(counts[0])
    retained = count - removed
    if retained < 1:
        raise ValueError('At least one point must survive deletion')
    survivors = points[~deleted].reshape(batch, retained, channels)
    if removed == 0:
        return survivors.clone()
    sampled = torch.randint(retained, (batch, removed), generator=generator)
    copies = survivors.gather(1, sampled[:, :, None].expand(-1, -1, channels))
    return torch.cat([survivors, copies], dim=1)

def random_point_dropout(points, generator, object_probability=0.2, fraction=0.2):
    """For selected objects, delete random points and refill only from survivors."""
    selected = torch.rand(len(points), generator=generator) < object_probability
    output = points.clone()
    if not selected.any():
        return output
    subset = points[selected]
    count = points.shape[1]
    removed = min(count - 1, max(1, round(count * fraction)))
    scores = torch.rand((len(subset), count), generator=generator)
    ids = scores.topk(removed, dim=1, largest=False, sorted=False).indices
    deleted = torch.zeros((len(subset), count), dtype=torch.bool).scatter_(1, ids, True)
    output[selected] = delete_and_refill(subset, deleted, generator)
    return output

def generic_augment(points: torch.Tensor, generator: torch.Generator, *, translation_generator: torch.Generator | None=None) -> torch.Tensor:
    """Yaw, scale, deletion/refill, jitter, dropout, translation."""
    if points.ndim != 3 or points.shape[1:] != (1024, 3) or len(points) == 0:
        raise ValueError('points must have shape [B>0,1024,3]')
    if points.device.type != 'cpu' or points.dtype != torch.float32:
        raise ValueError('augment expects CPU float32 points')
    if generator.device.type != 'cpu':
        raise ValueError('augment requires a CPU torch.Generator')
    translation_rng = generator if translation_generator is None else translation_generator
    if translation_rng.device.type != 'cpu':
        raise ValueError('translation_generator must be a CPU torch.Generator')
    if not torch.isfinite(points).all():
        raise ValueError('points must be finite')
    batch = len(points)
    output = random_yaw(points, generator)
    scale = torch.empty((batch, 1, 3)).uniform_(0.8, 1.2, generator=generator)
    output = normalize(output * scale)
    output = delete_and_refill(output, local_delete_mask(output, generator), generator)
    noise = torch.randn(output.shape, generator=generator).mul_(0.01).clamp_(-0.05, 0.05)
    output = output + noise
    output = random_point_dropout(output, generator)
    translation = torch.empty((batch, 1, 3)).uniform_(-0.1, 0.1, generator=translation_rng)
    return output + translation
from openpoints.cpp.fps import fps

def normalize(x):
    x = x - x.mean(1, keepdim=True)
    radius = x.square().sum(-1, keepdim=True).sqrt().amax(1, keepdim=True)
    if (radius <= 1e-12).any():
        raise ValueError('Degenerate cloud')
    return x / radius

def fps_augment(x, rng, mild=False):
    angle = torch.rand((len(x), 1), generator=rng) * (2 * math.pi)
    c, s = (angle.cos(), angle.sin())
    y = x.clone()
    y[:, :, 0] = x[:, :, 0] * c + x[:, :, 2] * s
    y[:, :, 2] = x[:, :, 2] * c - x[:, :, 0] * s
    permutations = torch.stack([torch.randperm(x.shape[1], generator=rng) for _ in x])
    y = y.gather(1, permutations[:, :, None].expand_as(y))
    lo, hi, sigma, clip = (0.95, 1.05, 0.005, 0.025) if mild else (0.9, 1.1, 0.01, 0.05)
    y = normalize(y * torch.empty((len(x), 1, 3)).uniform_(lo, hi, generator=rng))
    y += torch.randn(y.shape, generator=rng).mul_(sigma).clamp_(-clip, clip)
    y = y.cuda()
    candidates = fps(y, 1200)
    subset = torch.randperm(1200, generator=rng)[:1024].cuda()
    return y.gather(1, candidates[:, subset, None].expand(-1, -1, 3)).contiguous()
