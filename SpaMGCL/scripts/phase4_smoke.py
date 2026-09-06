"""Phase 4 smoke validation for the core SpaMGCL model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graphs.feature_graph import build_feature_graph  # noqa: E402
from src.graphs.spatial_graph import build_spatial_graph  # noqa: E402
from src.losses.sample_contrastive import SpatialNegativeMask  # noqa: E402
from src.models.spamgcl import SpaMGCL  # noqa: E402


def _synthetic_modal_features(n_spots: int, dim: int, reverse: bool = False) -> torch.Tensor:
    base = torch.linspace(0.0, 1.0, n_spots).unsqueeze(1)
    if reverse:
        base = 1.0 - base
    columns = [base, base**2, torch.sin(2.0 * torch.pi * base), torch.cos(2.0 * torch.pi * base)]
    noise = torch.randn(n_spots, max(0, dim - len(columns))) * 0.05
    return torch.cat(columns + ([noise] if noise.numel() else []), dim=1)[:, :dim]


def main() -> int:
    torch.manual_seed(7)
    n_spots = 24
    rna_dim = 8
    adt_dim = 8

    coords = torch.stack(
        [torch.arange(n_spots, dtype=torch.float32), torch.zeros(n_spots, dtype=torch.float32)],
        dim=1,
    ).numpy()
    spatial_adj = build_spatial_graph(coords, k=2, include_self=False)

    rna_features = _synthetic_modal_features(n_spots, rna_dim, reverse=False)
    adt_features = _synthetic_modal_features(n_spots, adt_dim, reverse=True)
    rna_feature_adj = build_feature_graph(rna_features.numpy(), k=3)
    adt_feature_adj = build_feature_graph(adt_features.numpy(), k=3)

    model = SpaMGCL(
        input_dims={"RNA": rna_dim, "ADT": adt_dim},
        gcn_hidden_dim=16,
        fine_dim=8,
        coarse_dim=8,
        representation_dim=8,
        num_clusters=5,
        alpha=0.5,
    )

    output = model(
        rna_features,
        spatial_adj,
        rna_feature_adj,
        adt_features,
        spatial_adj,
        adt_feature_adj,
        spatial_adj,
    )

    view_representations = [output.multigranularity_views[name]["g"] for name in model.view_order]
    _, wd_distances, wd_weights = model.weight_module(view_representations)

    ignore_mask = SpatialNegativeMask()(spatial_adj)
    first_pair = view_representations[0]
    second_pair = view_representations[1]
    logits = torch.nn.functional.normalize(first_pair, dim=1) @ torch.nn.functional.normalize(second_pair, dim=1).T
    masked_logits = logits.masked_fill(ignore_mask, -1e9)
    unmasked_denom = torch.logsumexp(logits, dim=1)
    masked_denom = torch.logsumexp(masked_logits, dim=1)

    assert torch.isfinite(output.total_loss)
    assert torch.isfinite(output.spatial_loss)
    assert torch.isfinite(output.weights).all()
    assert torch.isfinite(wd_weights).all()
    assert output.weights.shape == wd_weights.shape == (4,)
    assert not torch.allclose(output.weights, wd_weights)
    assert torch.all(output.weights >= 0)
    assert torch.allclose(output.weights.sum(), torch.tensor(1.0), atol=1e-5)
    assert output.spatial_loss.item() >= 0
    assert torch.any(masked_denom < unmasked_denom)
    assert ignore_mask.any()

    print(f"L_rec={output.reconstruction_loss.item():.6f}")
    print(f"L_MGCL={output.sample_contrastive_loss.item():.6f}")
    print(f"L_cluster={output.cluster_contrastive_loss.item():.6f}")
    print(f"L_spatial={output.spatial_loss.item():.6f}")
    print(f"WD={wd_distances.tolist()}")
    print(f"SC={output.spatial_consistency_scores.tolist()}")
    print(f"w_WD={wd_weights.tolist()}")
    print(f"w_SC={output.weights.tolist()}")
    print("✅ Phase 4 验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
