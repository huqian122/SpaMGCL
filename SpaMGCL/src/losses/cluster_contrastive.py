"""MGCMVC-style cluster-level contrastive loss."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ClusterHead(nn.Module):
    """Map each ``G_v`` to soft cluster assignments ``Q_v``."""

    def __init__(self, input_dim: int, num_clusters: int) -> None:
        super().__init__()
        if num_clusters < 2:
            raise ValueError("num_clusters must be at least 2")
        self.projection = nn.Linear(input_dim, num_clusters)

    def forward(self, representation: Tensor) -> Tensor:
        if representation.ndim != 2:
            raise ValueError("representation must be 2D")
        return F.softmax(self.projection(representation), dim=1)


def target_distribution(q: Tensor, eps: float = 1e-12) -> Tensor:
    """Construct MGCMVC's sharpened target distribution ``P``."""

    if q.ndim != 2:
        raise ValueError("q must be a 2D probability matrix")
    weight = q.pow(2) / q.sum(dim=0, keepdim=True).clamp_min(eps)
    return weight / weight.sum(dim=1, keepdim=True).clamp_min(eps)


class ClusterContrastiveLoss(nn.Module):
    """Contrast corresponding cluster distributions across view pairs.

    The regularizer follows the reference implementation's batch cluster
    entropy term: ``log(C) + sum_c p_c log(p_c)`` for each view.
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-12) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.eps = eps

    def pair_loss(self, q_left: Tensor, q_right: Tensor) -> Tensor:
        if q_left.shape != q_right.shape or q_left.ndim != 2:
            raise ValueError("q_left and q_right must have equal 2D shapes")
        p_left = target_distribution(q_left, self.eps)
        p_right = target_distribution(q_right, self.eps)
        # Cluster vectors are columns of P, matching the source implementation.
        left = F.normalize(p_left.T, dim=1)
        right = F.normalize(p_right.T, dim=1)
        logits = left @ right.T / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    def entropy_regularizer(self, q: Tensor) -> Tensor:
        cluster_mass = q.mean(dim=0)
        return torch.log(torch.tensor(q.shape[1], device=q.device, dtype=q.dtype)) + (
            cluster_mass * torch.log(cluster_mass.clamp_min(self.eps))
        ).sum()

    def forward(
        self,
        assignments: Sequence[Tensor],
        regularization_weight: float = 1.0,
    ) -> Tensor:
        if len(assignments) < 2:
            raise ValueError("at least two view assignments are required")
        total = assignments[0].new_zeros(())
        pair_count = 0
        for left_index in range(len(assignments)):
            for right_index in range(left_index + 1, len(assignments)):
                total = total + self.pair_loss(
                    assignments[left_index], assignments[right_index]
                )
                pair_count += 1
        contrastive = total / pair_count
        regularizer = torch.stack(
            [self.entropy_regularizer(assignment) for assignment in assignments]
        ).mean()
        return contrastive + regularization_weight * regularizer


def cluster_contrastive_loss(
    assignments: Sequence[Tensor],
    temperature: float = 1.0,
    regularization_weight: float = 1.0,
) -> Tensor:
    return ClusterContrastiveLoss(
        temperature=temperature,
    )(assignments, regularization_weight=regularization_weight)
