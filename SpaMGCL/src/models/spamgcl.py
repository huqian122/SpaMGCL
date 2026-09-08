"""Core SpaMGCL model composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from src.losses.cluster_contrastive import ClusterContrastiveLoss, ClusterHead
from src.losses.reconstruction import ReconstructionLoss
from src.losses.sample_contrastive import AdaptiveSampleContrastiveLoss
from src.losses.spatial_regularization import SpatialRegularizationLoss
from src.models.adaptive_weight import AdaptiveWeightModule
from src.models.gcn import GraphViewGCN
from src.models.multigranularity import ViewMultiGranularityEncoder


@dataclass
class SpaMGCLForwardOutput:
    total_loss: Tensor
    reconstruction_loss: Tensor
    sample_contrastive_loss: Tensor
    cluster_contrastive_loss: Tensor
    spatial_loss: Tensor
    global_representation: Tensor
    wd_distances: Tensor
    spatial_consistency_scores: Tensor
    combined_distances: Tensor
    weights: Tensor
    cluster_assignments: Sequence[Tensor]
    gcn_views: Dict[str, Tensor]
    multigranularity_views: Dict[str, Dict[str, Tensor]]


class SpaMGCL(nn.Module):
    """End-to-end Phase 4 model with SC and SNF enabled."""

    def __init__(
        self,
        input_dims: Mapping[str, int],
        gcn_hidden_dim: int = 64,
        fine_dim: int = 32,
        coarse_dim: int = 32,
        representation_dim: int = 32,
        num_clusters: int = 7,
        fusion_hidden_dim: int = 128,
        alpha: float = 0.5,
        temperature: float = 0.5,
        cluster_temperature: float = 1.0,
        cluster_regularization_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if len(input_dims) != 2:
            raise ValueError("SpaMGCL expects exactly two modalities")
        self.modality_names = tuple(input_dims)
        self.gcn = GraphViewGCN(input_dims, hidden_dim=gcn_hidden_dim)
        first_modality, second_modality = self.modality_names
        self.view_order = (
            f"{first_modality.lower()}_spatial",
            f"{first_modality.lower()}_feature",
            f"{second_modality.lower()}_spatial",
            f"{second_modality.lower()}_feature",
        )
        self.view_encoders = nn.ModuleDict(
            {
                view_name: ViewMultiGranularityEncoder(
                    input_dim=gcn_hidden_dim,
                    fine_dim=fine_dim,
                    coarse_dim=coarse_dim,
                    output_dim=representation_dim,
                )
                for view_name in self.view_order
            }
        )
        self.weight_module = AdaptiveWeightModule(
            num_views=len(self.view_order),
            representation_dim=representation_dim,
            global_dim=representation_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            alpha=alpha,
        )
        # MGCMVC uses one label projection shared by all views. Sharing the
        # head makes cluster index c mean the same thing in every Q_v.
        self.cluster_head = ClusterHead(representation_dim, num_clusters)
        self.reconstruction_loss = ReconstructionLoss()
        self.sample_contrastive_loss = AdaptiveSampleContrastiveLoss(temperature=temperature)
        self.cluster_contrastive_loss = ClusterContrastiveLoss(temperature=cluster_temperature)
        self.spatial_regularization_loss = SpatialRegularizationLoss()
        self.cluster_regularization_weight = cluster_regularization_weight

    def forward(
        self,
        modality_a_features: object,
        modality_a_spatial_adj: object,
        modality_a_feature_adj: object,
        modality_b_features: object,
        modality_b_spatial_adj: object,
        modality_b_feature_adj: object,
        spatial_adjacency: object,
        alpha: Optional[float] = None,
        modality_a_name: Optional[str] = None,
        modality_b_name: Optional[str] = None,
    ) -> SpaMGCLForwardOutput:
        modality_a_name = modality_a_name or self.modality_names[0]
        modality_b_name = modality_b_name or self.modality_names[1]
        gcn_views = self.gcn(
            modality_a_features,
            modality_a_spatial_adj,
            modality_a_feature_adj,
            modality_b_features,
            modality_b_spatial_adj,
            modality_b_feature_adj,
            modality_a_name=modality_a_name,
            modality_b_name=modality_b_name,
        )

        multigranularity_views: Dict[str, Dict[str, Tensor]] = {
            view_name: self.view_encoders[view_name](gcn_views[view_name])
            for view_name in self.view_order
        }
        representations = [multigranularity_views[view_name]["g"] for view_name in self.view_order]
        reconstructions = [multigranularity_views[view_name]["reconstruction"] for view_name in self.view_order]

        (
            global_representation,
            wd_distances,
            spatial_consistency_scores,
            combined_distances,
            weights,
        ) = self.weight_module(
            representations,
            spatial_adjacency=spatial_adjacency,
            alpha=alpha,
        )

        reconstruction = self.reconstruction_loss(
            [gcn_views[view_name] for view_name in self.view_order],
            reconstructions,
        )
        sample_contrastive = self.sample_contrastive_loss(
            representations,
            weights,
            spatial_adjacency=spatial_adjacency,
        )
        assignments = [
            self.cluster_head(multigranularity_views[view_name]["g"])
            for view_name in self.view_order
        ]
        cluster_contrastive = self.cluster_contrastive_loss(
            assignments,
            regularization_weight=self.cluster_regularization_weight,
        )
        spatial_loss = self.spatial_regularization_loss(representations, spatial_adjacency)

        total = reconstruction + sample_contrastive + cluster_contrastive + spatial_loss
        return SpaMGCLForwardOutput(
            total_loss=total,
            reconstruction_loss=reconstruction,
            sample_contrastive_loss=sample_contrastive,
            cluster_contrastive_loss=cluster_contrastive,
            spatial_loss=spatial_loss,
            global_representation=global_representation,
            wd_distances=wd_distances,
            spatial_consistency_scores=spatial_consistency_scores,
            combined_distances=combined_distances,
            weights=weights,
            cluster_assignments=assignments,
            gcn_views=gcn_views,
            multigranularity_views=multigranularity_views,
        )
