"""Spatial smoothness regularization for SpaMGCL."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from scipy import sparse


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


class SpatialRegularizationLoss(nn.Module):
    """Compute ``L_spatial = sum_v sum_(i,j in E_s) ||g_i^v - g_j^v||^2``."""

    def forward(self, representations: Sequence[Tensor], spatial_adjacency: object) -> Tensor:
        if not representations:
            raise ValueError("at least one representation is required")
        coo = _to_sparse_coo(spatial_adjacency)
        row = torch.as_tensor(coo.row, device=representations[0].device, dtype=torch.long)
        col = torch.as_tensor(coo.col, device=representations[0].device, dtype=torch.long)
        if row.numel() == 0:
            return representations[0].new_zeros(())
        mask = row < col
        row = row[mask]
        col = col[mask]
        if row.numel() == 0:
            return representations[0].new_zeros(())

        total = representations[0].new_zeros(())
        for representation in representations:
            if representation.ndim != 2:
                raise ValueError("representations must all be 2D")
            if representation.shape[0] <= row.max().item():
                raise ValueError("spatial adjacency index exceeds representation size")
            diff = representation[row] - representation[col]
            total = total + diff.pow(2).sum(dim=1).mean()
        return total


def spatial_regularization_loss(
    representations: Sequence[Tensor],
    spatial_adjacency: object,
) -> Tensor:
    return SpatialRegularizationLoss()(representations, spatial_adjacency)
