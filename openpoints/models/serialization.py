from dataclasses import dataclass
import torch
AXIS_ORDERS = ("xyz", "xzy", "yxz", "yzx", "zxy", "zyx")
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

@dataclass(frozen=True)
class Serialization:
    name: str
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor

def _validate_centers(centers: torch.Tensor) -> None:
    if centers.ndim != 3 or centers.shape[-1] != 3:
        raise ValueError(f"centers must have shape [B,T,3], got {tuple(centers.shape)}")
    if not torch.isfinite(centers).all():
        raise ValueError("centers contain non-finite values")

def parse_order(name: str) -> str:
    if not name.startswith("morton_"):
        raise ValueError(f"Expected a Morton order, got {name}")
    axis_order = name.split("_", 1)[1]
    if axis_order not in AXIS_ORDERS:
        raise ValueError(f"Unknown Morton axis order: {axis_order}")
    return axis_order

def morton_permutation(
    centers: torch.Tensor,
    name: str,
    bits: int = 10,
) -> Serialization:
    """Return a deterministic serialization.

    `permutation[b, t]` is the canonical token index placed at serialized
    position `t`. Ties are resolved by the canonical token index.
    """
    _validate_centers(centers)
    if not 1 <= bits <= 20:
        raise ValueError("bits must be in [1, 20]")
    axis_order = parse_order(name)
    batch, tokens, _ = centers.shape

    lower = centers.amin(dim=1, keepdim=True)
    upper = centers.amax(dim=1, keepdim=True)
    span = (upper - lower).clamp_min(torch.finfo(centers.dtype).eps)
    max_value = (1 << bits) - 1
    quantized = torch.round((centers - lower) / span * max_value).to(torch.int64)

    code = torch.zeros((batch, tokens), dtype=torch.int64, device=centers.device)
    for bit in range(bits):
        for priority, axis_name in enumerate(axis_order):
            lane = 2 - priority
            axis = AXIS_INDEX[axis_name]
            bit_value = (quantized[..., axis] >> bit) & 1
            code |= bit_value << (3 * bit + lane)

    canonical_id = torch.arange(tokens, device=centers.device, dtype=torch.int64)
    composite = code * (tokens + 1) + canonical_id.unsqueeze(0)
    permutation = torch.argsort(composite, dim=1, stable=True)
    inverse = torch.empty_like(permutation)
    serial_position = torch.arange(tokens, device=centers.device, dtype=torch.int64)
    inverse.scatter_(1, permutation, serial_position.unsqueeze(0).expand(batch, -1))
    return Serialization(name=name, permutation=permutation, inverse_permutation=inverse)

def gather_tokens(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if values.shape[0] != indices.shape[0]:
        raise ValueError("batch dimensions do not match")
    if indices.ndim != 2:
        raise ValueError("indices must have shape [B,T]")
    view_shape = (*indices.shape, *((1,) * (values.ndim - 2)))
    expand_shape = (*indices.shape, *values.shape[2:])
    return torch.gather(values, 1, indices.view(view_shape).expand(expand_shape))

def serialize(centers,direction):
    code=_pointcept_continuous_encode(centers.float(),direction,.02)
    permutation=torch.argsort(code,dim=1,stable=True)
    return Serialization('grid_snake_'+direction,permutation,permutation.argsort(dim=1))

def _grid_coord(pos, grid_size=0.02, max_depth=16):
    """Quantize batched xyz coordinates for serialization codes."""
    if grid_size is not None and grid_size > 0:
        # Match Pointcept-style serialization: shift each cloud to a
        # non-negative grid before quantization. This avoids truncation
        # artifacts for centered point clouds with negative coordinates.
        shifted = pos - pos.amin(dim=1, keepdim=True)
        grid = torch.div(shifted, grid_size, rounding_mode='trunc').long()
    else:
        pos_min = pos.amin(dim=1, keepdim=True)
        pos_range = (pos.amax(dim=1, keepdim=True) - pos_min).clamp(min=1e-8)
        depth = min(max_depth, 10)
        scale = (1 << depth) - 1
        grid = ((pos - pos_min) / pos_range * scale).long().clamp(0, scale)

    with torch.no_grad():
        max_coord = int(grid.amax().item()) if grid.numel() > 0 else 1
        depth = max(1, min(max_depth, max_coord.bit_length()))
    limit = (1 << depth) - 1
    grid = grid.clamp(0, limit)
    return grid, depth

def _pointcept_continuous_encode(pos, order, grid_size):
    """Serpentine continuous scan for the six axis orders.

    This keeps adjacent grid cells close while producing collision-free codes.
    The earlier PointDGMamba-style sign formula can collide on small grids;
    RWKV is order-sensitive, so deterministic one-cell/one-code mapping matters.
    """
    if order not in AXIS_ORDERS:
        raise ValueError(f'Continuous serialization expects axis order, got {order}')
    dim_map = {'x': 0, 'y': 1, 'z': 2}
    dims = [dim_map[c] for c in order]
    grid, _ = _grid_coord(pos, grid_size=grid_size)
    grid = grid[:, :, dims]

    code = grid[:, :, 0]
    base = (grid[:, :, 0].amax(dim=1) + 1).clamp(min=1)
    for axis in (1, 2):
        coord = grid[:, :, axis]
        reversed_code = base.unsqueeze(1) - 1 - code
        code = coord * base.unsqueeze(1) + torch.where(
            coord % 2 == 0, code, reversed_code)
        coord_base = (coord.amax(dim=1) + 1).clamp(min=1)
        base = base * coord_base
    return code
