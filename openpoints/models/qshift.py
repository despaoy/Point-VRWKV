import torch
from .serialization import Serialization, gather_tokens
HEAD_SIZE = 64
CHANNELS_PER_DIRECTION = 10

@torch.no_grad()
def plan_directional_donors(centers: torch.Tensor) -> torch.Tensor:
    """Return canonical donor indices [B,T,6], with -1 for empty sectors.

    A displacement belongs to its largest-absolute-coordinate axis (ties x, y,
    z), then its sign. Select the nearest nonzero displacement in each sector.
    Equal-distance candidates use geometric lexicographic (x,y,z) order, making
    tie-breaking independent of Morton and canonical token positions. Candidates
    with exactly identical XYZ cannot be geometrically distinguished; if their
    features differ, that degenerate case has no unique equivariant single donor.
    """
    if centers.ndim != 3 or centers.shape[-1] != 3:
        raise ValueError("centers must have shape [B,T,3]")
    if not centers.is_floating_point():
        raise TypeError("centers must be floating point")
    batch, token_count, _ = centers.shape
    candidate_order = torch.arange(token_count, device=centers.device).expand(batch, -1)
    # Stable sorts from the least significant to the most significant coordinate.
    for axis in (2, 1, 0):
        coordinate = centers[..., axis].gather(1, candidate_order)
        positions = torch.argsort(coordinate, dim=1, stable=True)
        candidate_order = candidate_order.gather(1, positions)
    candidates = gather_tokens(centers, candidate_order)
    displacement = candidates[:, None, :, :] - centers[:, :, None, :]
    squared_distance = displacement.square().sum(dim=-1)
    axis = displacement.abs().argmax(dim=-1)
    signed_component = displacement.gather(-1, axis.unsqueeze(-1)).squeeze(-1)
    sector = axis * 2 + (signed_component < 0).long()
    nonzero = squared_distance > 0
    donor_columns = []
    for direction in range(6):
        distance = squared_distance.masked_fill(~(nonzero & (sector == direction)), torch.inf)
        minimum, column = distance.min(dim=-1)
        canonical_donor = candidate_order.gather(1, column)
        donor_columns.append(torch.where(torch.isfinite(minimum), canonical_donor, -1))
    return torch.stack(donor_columns, dim=-1)

def remap_directional_donors(donors: torch.Tensor, serialization: Serialization) -> torch.Tensor:
    """Reorder receivers and donors together while retaining the -1 boundary."""
    reordered = gather_tokens(donors, serialization.permutation)
    batch = torch.arange(donors.shape[0], device=donors.device)[:, None, None]
    remapped = serialization.inverse_permutation[batch, reordered.clamp_min(0)]
    return torch.where(reordered >= 0, remapped, -1)

def directional_qshift(tokens: torch.Tensor, donors: torch.Tensor) -> torch.Tensor:
    """Gather ten channels per direction per head, retaining the last four."""
    if tokens.ndim != 3 or tokens.shape[-1] % HEAD_SIZE:
        raise ValueError("tokens must have shape [B,T,H*64]")
    if donors.shape != (*tokens.shape[:2], 6) or donors.dtype != torch.long:
        raise ValueError("donors must be long [B,T,6]")
    channels = torch.arange(tokens.shape[-1], device=tokens.device)
    group = (channels % HEAD_SIZE) // CHANNELS_PER_DIRECTION
    # One gather for all channels avoids 6 separate neighborhood feature copies.
    indices = donors[..., group.clamp_max(5)]
    self_index = torch.arange(tokens.shape[1], device=tokens.device)[None, :, None]
    indices = torch.where(group[None, None, :] == 6, self_index, indices)
    output = tokens.gather(1, indices.clamp_min(0))
    return output.masked_fill(indices < 0, 0.0)
