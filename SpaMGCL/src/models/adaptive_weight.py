"""MGCMVC / SpaMGCL global fusion and adaptive view weighting."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from scipy import sparse


def wasserstein_distance_1d(x: Tensor, y: Tensor) -> Tensor:
    """Compute the empirical 1-D Wasserstein-1 distance.

    MGCMVC's reference implementation applies SciPy's 1-D Wasserstein
    distance to flattened view/global representations. This torch equivalent
    preserves that empirical definition for equal-sized tensors and keeps the
    operation differentiable with respect to sorted values.
    """

    x = x.reshape(-1)
    y = y.reshape(-1)
    if x.numel() != y.numel():
        raise ValueError(
            "This empirical implementation requires equal flattened sizes; "
            f"received {x.numel()} and {y.numel()}"
        )
    return torch.mean(torch.abs(torch.sort(x).values - torch.sort(y).values))


class GlobalFusion(nn.Module):
    """Fuse view-level ``G_v`` tensors into global representation ``U``."""

    def __init__(
        self,
        num_views: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        self.num_views = num_views
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(num_views * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.normalize_output = normalize_output

    def forward(self, representations: Sequence[Tensor]) -> Tensor:
        _validate_views(representations, self.num_views, self.input_dim)
        fused = self.network(torch.cat(list(representations), dim=1))
        if self.normalize_output:
            fused = torch.nn.functional.normalize(fused, dim=1)
        return fused


def _to_sparse_coo(adjacency: object) -> sparse.coo_matrix:
    if adjacency is None:
        raise ValueError("adjacency cannot be None")
    if sparse.issparse(adjacency):
        return adjacency.tocoo()
    if isinstance(adjacency, torch.Tensor):
        if adjacency.is_sparse:
            tensor = adjacency.coalesce().cpu()
            indices = tensor.indices().numpy()
            values = tensor.values().numpy()
            return sparse.coo_matrix((values, (indices[0], indices[1])), shape=tensor.shape)
        adjacency = adjacency.detach().cpu().numpy()
    return sparse.coo_matrix(np.asarray(adjacency))


def spatial_consistency_score(representation: Tensor, adjacency: object) -> Tensor:
    """Compute graph smoothness over the spatial edge set.

    The score is the mean squared distance over non-zero spatial edges.
    Self-loops are excluded from the edge set.
    """

    if representation.ndim != 2:
        raise ValueError("representation must be 2D")
    coo = _to_sparse_coo(adjacency)
    row = torch.as_tensor(coo.row, device=representation.device, dtype=torch.long)
    col = torch.as_tensor(coo.col, device=representation.device, dtype=torch.long)
    if row.numel() == 0:
        return representation.new_zeros(())
    mask = row != col
    row = row[mask]
    col = col[mask]
    if row.numel() == 0:
        return representation.new_zeros(())
    diff = representation[row] - representation[col]
    return diff.pow(2).sum(dim=1).mean()


class AdaptiveWeightModule(nn.Module):
    """Compute WD-based adaptive weights and optional SC-aware weights."""

    def __init__(
        self,
        num_views: int,
        representation_dim: int,
        global_dim: int,
        fusion_hidden_dim: int = 256,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.global_fusion = GlobalFusion(
            num_views=num_views,
            input_dim=representation_dim,
            output_dim=global_dim,
            hidden_dim=fusion_hidden_dim,
        )
        self.alpha = alpha

    def forward(
        self,
        representations: Sequence[Tensor],
        spatial_adjacency: Optional[object] = None,
        alpha: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return WD weights or SC-aware weights.

        When ``spatial_adjacency`` is omitted, the module preserves the
        Phase 3 signature and returns ``(global_u, distances, weights)``.
        When provided, it returns ``(global_u, wd_distances, sc_scores,
        combined_distances, weights)``.
        """

        global_u = self.global_fusion(representations)
        wd_distances = torch.stack(
            [wasserstein_distance_1d(view, global_u) for view in representations]
        )
        if spatial_adjacency is None:
            weights = torch.softmax(-wd_distances, dim=0)
            return global_u, wd_distances, weights

        sc_scores = torch.stack(
            [spatial_consistency_score(view, spatial_adjacency) for view in representations]
        )
        scale = self.alpha if alpha is None else alpha
        combined_distances = wd_distances + scale * sc_scores
        weights = torch.softmax(-combined_distances, dim=0)
        return global_u, wd_distances, sc_scores, combined_distances, weights


def _validate_views(
    representations: Sequence[Tensor],
    num_views: int,
    representation_dim: int,
) -> None:
    if len(representations) != num_views:
        raise ValueError(f"expected {num_views} views; received {len(representations)}")
    if not representations:
        raise ValueError("at least one view is required")
    n_spots = representations[0].shape[0]
    for index, representation in enumerate(representations):
        if representation.ndim != 2:
            raise ValueError(f"view {index} must be 2D")
        if representation.shape[0] != n_spots:
            raise ValueError("all views must have the same number of spots")
        if representation.shape[1] != representation_dim:
            raise ValueError(
                f"view {index} has dim {representation.shape[1]}, "
                f"expected {representation_dim}"
            )
