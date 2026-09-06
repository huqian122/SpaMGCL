"""Phase 3 smoke validation using random tensors only."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.losses.cluster_contrastive import (  # noqa: E402
    ClusterContrastiveLoss,
    ClusterHead,
)
from src.losses.reconstruction import ReconstructionLoss  # noqa: E402
from src.losses.sample_contrastive import AdaptiveSampleContrastiveLoss  # noqa: E402
from src.models.adaptive_weight import AdaptiveWeightModule  # noqa: E402
from src.models.multigranularity import ViewMultiGranularityEncoder  # noqa: E402


def main() -> int:
    torch.manual_seed(0)
    n_spots = 24
    n_views = 4
    input_dims = [12, 9, 15, 11]
    fine_dim = 8
    coarse_dim = 8
    representation_dim = 8
    num_clusters = 5

    inputs = [torch.randn(n_spots, input_dim) for input_dim in input_dims]
    encoders = [
        ViewMultiGranularityEncoder(
            input_dim=input_dim,
            fine_dim=fine_dim,
            coarse_dim=coarse_dim,
            output_dim=representation_dim,
        )
        for input_dim in input_dims
    ]
    encoded = [encoder(x) for encoder, x in zip(encoders, inputs)]
    representations = [item["g"] for item in encoded]
    reconstructions = [item["reconstruction"] for item in encoded]

    adaptive_weight = AdaptiveWeightModule(
        num_views=n_views,
        representation_dim=representation_dim,
        global_dim=representation_dim,
    )
    global_representation, distances, weights = adaptive_weight(representations)

    reconstruction = ReconstructionLoss()(inputs, reconstructions)
    sample_contrastive = AdaptiveSampleContrastiveLoss(temperature=0.5)(
        representations, weights
    )
    cluster_heads = [ClusterHead(representation_dim, num_clusters) for _ in representations]
    assignments = [head(representation) for head, representation in zip(cluster_heads, representations)]
    cluster_contrastive = ClusterContrastiveLoss(temperature=1.0)(
        assignments, regularization_weight=1.0
    )

    assert global_representation.shape == (n_spots, representation_dim)
    assert distances.shape == (n_views,)
    assert weights.shape == (n_views,)
    assert torch.isfinite(reconstruction)
    assert torch.isfinite(sample_contrastive)
    assert torch.isfinite(cluster_contrastive)
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    assert all(assignment.shape == (n_spots, num_clusters) for assignment in assignments)
    assert all(torch.allclose(assignment.sum(dim=1), torch.ones(n_spots), atol=1e-5) for assignment in assignments)

    print(f"L_rec={reconstruction.item():.6f}")
    print(f"L_MGCL={sample_contrastive.item():.6f}")
    print(f"L_cluster={cluster_contrastive.item():.6f}")
    print("✅ Phase 3 验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
