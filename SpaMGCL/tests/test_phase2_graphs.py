from pathlib import Path

import anndata

from src.graphs.feature_graph import build_feature_graph
from src.graphs.spatial_graph import build_spatial_graph, extract_spatial_coordinates
from src.models.gcn import GraphViewGCN


DATA_ROOT = Path(
    "/Users/qianhu/博士论文/多角度聚类医学论文/SpaMGCL/Expert Systems SMGC/SMGC-data"
)


def _pick_feature_key(adata):
    for key in ("spfac", "nsfac"):
        if key in adata.obsm:
            return key
    raise KeyError("No candidate feature key found")


def test_phase2_smoke_simulation_graphs():
    rna = anndata.read_h5ad(DATA_ROOT / "simulation" / "adata_RNA.h5ad", backed="r")
    adt = anndata.read_h5ad(DATA_ROOT / "simulation" / "adata_ADT.h5ad", backed="r")

    rna_coords = extract_spatial_coordinates(rna).coordinates
    adt_coords = extract_spatial_coordinates(adt).coordinates
    rna_features = rna.obsm[_pick_feature_key(rna)]
    adt_features = adt.obsm[_pick_feature_key(adt)]

    rna_spatial = build_spatial_graph(rna_coords, k=15)
    rna_feature = build_feature_graph(rna_features, k=15)
    adt_spatial = build_spatial_graph(adt_coords, k=15)
    adt_feature = build_feature_graph(adt_features, k=15)

    model = GraphViewGCN({"RNA": rna_features.shape[1], "ADT": adt_features.shape[1]}, hidden_dim=32)
    outputs = model(rna_features, rna_spatial, rna_feature, adt_features, adt_spatial, adt_feature)

    assert rna_spatial.shape == (rna.n_obs, rna.n_obs)
    assert rna_feature.shape == (rna.n_obs, rna.n_obs)
    assert adt_spatial.shape == (adt.n_obs, adt.n_obs)
    assert adt_feature.shape == (adt.n_obs, adt.n_obs)
    assert outputs["rna_spatial"].shape[0] == rna.n_obs
    assert outputs["rna_feature"].shape[0] == rna.n_obs
    assert outputs["adt_spatial"].shape[0] == adt.n_obs
    assert outputs["adt_feature"].shape[0] == adt.n_obs
    assert rna_spatial.nnz > 0
    assert rna_feature.nnz > 0
    assert adt_spatial.nnz > 0
    assert adt_feature.nnz > 0

