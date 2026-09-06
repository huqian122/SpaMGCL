"""Phase 2 smoke validation on Simulation without reading ``adata.X``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graphs.feature_graph import build_feature_graph
from src.graphs.spatial_graph import build_spatial_graph, extract_spatial_coordinates
from src.models.gcn import GraphViewGCN


DATA_ROOT = Path(
    "/Users/qianhu/博士论文/多角度聚类医学论文/SpaMGCL/Expert Systems SMGC/SMGC-data"
)


def _pick_feature_key(adata: anndata.AnnData) -> str:
    for key in ("spfac", "nsfac"):
        if key in adata.obsm:
            return key
    raise KeyError("No candidate feature key found in obsm['spfac'] or obsm['nsfac']")


def main() -> int:
    rna_path = DATA_ROOT / "simulation" / "adata_RNA.h5ad"
    adt_path = DATA_ROOT / "simulation" / "adata_ADT.h5ad"
    rna = anndata.read_h5ad(rna_path, backed="r")
    adt = anndata.read_h5ad(adt_path, backed="r")

    rna_coords = extract_spatial_coordinates(rna).coordinates
    adt_coords = extract_spatial_coordinates(adt).coordinates
    rna_feat_key = _pick_feature_key(rna)
    adt_feat_key = _pick_feature_key(adt)
    rna_features = rna.obsm[rna_feat_key]
    adt_features = adt.obsm[adt_feat_key]

    rna_spatial = build_spatial_graph(rna_coords, k=15)
    rna_feature = build_feature_graph(rna_features, k=15)
    adt_spatial = build_spatial_graph(adt_coords, k=15)
    adt_feature = build_feature_graph(adt_features, k=15)

    model = GraphViewGCN(
        input_dims={"RNA": rna_features.shape[1], "ADT": adt_features.shape[1]},
        hidden_dim=32,
    )
    outputs = model(
        rna_features,
        rna_spatial,
        rna_feature,
        adt_features,
        adt_spatial,
        adt_feature,
        modality_a_name="RNA",
        modality_b_name="ADT",
    )

    summary = {
        "rna_feature_key": rna_feat_key,
        "adt_feature_key": adt_feat_key,
        "rna_spatial_shape": tuple(rna_spatial.shape),
        "rna_spatial_nnz": int(rna_spatial.nnz),
        "rna_feature_shape": tuple(rna_feature.shape),
        "rna_feature_nnz": int(rna_feature.nnz),
        "adt_spatial_shape": tuple(adt_spatial.shape),
        "adt_spatial_nnz": int(adt_spatial.nnz),
        "adt_feature_shape": tuple(adt_feature.shape),
        "adt_feature_nnz": int(adt_feature.nnz),
        "view_shapes": {key: tuple(value.shape) for key, value in outputs.items()},
    }

    assert rna_spatial.shape == (rna.n_obs, rna.n_obs)
    assert rna_feature.shape == (rna.n_obs, rna.n_obs)
    assert adt_spatial.shape == (adt.n_obs, adt.n_obs)
    assert adt_feature.shape == (adt.n_obs, adt.n_obs)
    assert all(value.ndim == 2 for value in outputs.values())
    assert all(int(value.shape[0]) == rna.n_obs for key, value in outputs.items() if key.startswith("rna_"))
    assert all(int(value.shape[0]) == adt.n_obs for key, value in outputs.items() if key.startswith("adt_"))
    assert rna_spatial.nnz > 0 and rna_feature.nnz > 0
    assert adt_spatial.nnz > 0 and adt_feature.nnz > 0

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
