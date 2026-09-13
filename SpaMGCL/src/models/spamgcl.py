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
    mgcl_weight_std: Tensor
    neg_count: int
    snf_masked_positions: int
    weighted_representation: Tensor
    mean_representation: Tensor
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
        # MGCMVC applies the shared cluster head to the fine-grained encoder
        # output Z_v, before multigranularity fusion. This also keeps the
        # implementation correct when fine_dim and representation_dim differ.
        self.cluster_head = ClusterHead(fine_dim, num_clusters)
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
        use_spatial_weighting: bool = True,
        use_spatial_negative_filter: bool = True,
        use_spatial_loss: bool = True,
        lambda_rec: float = 1.0,
        lambda_mgcl: float = 1.0,
        lambda_cluster: float = 1.0,
        lambda_spatial: float = 1.0,
    ) -> SpaMGCLForwardOutput:
        loss_coefficients = (
            lambda_rec,
            lambda_mgcl,
            lambda_cluster,
            lambda_spatial,
        )
        if any(float(coefficient) < 0 for coefficient in loss_coefficients):
            raise ValueError("loss coefficients must be nonnegative")
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

        if use_spatial_weighting:
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
        else:
            global_representation, wd_distances, weights = self.weight_module(
                representations,
                spatial_adjacency=None,
            )
            spatial_consistency_scores = torch.zeros_like(wd_distances)
            combined_distances = wd_distances

        reconstruction = self.reconstruction_loss(
            [gcn_views[view_name] for view_name in self.view_order],
            reconstructions,
        )
        sample_contrastive, sample_debug = self.sample_contrastive_loss(
            representations,
            weights,
            spatial_adjacency=(
                spatial_adjacency if use_spatial_negative_filter else None
            ),
            return_debug=True,
        )
        assignments = [
            self.cluster_head(multigranularity_views[view_name]["z"])
            for view_name in self.view_order
        ]
        cluster_contrastive = self.cluster_contrastive_loss(
            assignments,
            regularization_weight=self.cluster_regularization_weight,
        )
        if use_spatial_loss:
            spatial_loss = self.spatial_regularization_loss(
                representations, spatial_adjacency
            )
        else:
            spatial_loss = representations[0].new_zeros(())
        stacked_representations = torch.stack(representations, dim=0)
        weighted_representation = torch.sum(
            weights[:, None, None] * stacked_representations, dim=0
        )
        mean_representation = stacked_representations.mean(dim=0)

        total = (
            lambda_rec * reconstruction
            + lambda_mgcl * sample_contrastive
            + lambda_cluster * cluster_contrastive
            + lambda_spatial * spatial_loss
        )
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
            mgcl_weight_std=weights.detach().std(unbiased=False),
            neg_count=int(sample_debug["neg_count"]),
            snf_masked_positions=int(sample_debug["masked_positions"]),
            weighted_representation=weighted_representation,
            mean_representation=mean_representation,
            cluster_assignments=assignments,
            gcn_views=gcn_views,
            multigranularity_views=multigranularity_views,
        )
