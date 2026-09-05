# SpaMGCL Project Specification

Status: Phase 0 audit artifact, 2026-09-05

## 1. Scope

SpaMGCL is planned as a spatial multi-omics clustering method for spatial
domain identification. It targets paired measurements from the same tissue
locations, including RNA plus ADT and RNA plus ATAC datasets.

The planned pipeline is:

```text
modality-specific preprocessing
  -> spatial and feature graphs
  -> graph-specific views
  -> FG / CG / MG representations
  -> adaptive view weighting
  -> spatial-neighborhood negative filtering
  -> spatial domain clustering
```

This document records the implementation contract. It does not authorize
model implementation or training during Phase 0.

## 2. Source and modification boundary

| Component | Source | SpaMGCL treatment |
|---|---|---|
| RNA, ADT, ATAC preprocessing | SMGC | Reimplement after data-field and source verification |
| Spatial graph | SMGC | KNN over verified tissue coordinates |
| Feature graph | SMGC | Modality-specific feature graph; no cross-modality adjacency |
| GCN graph views | SMGC | Four views for each paired two-modality dataset |
| FG, CG, MG | MGCMVC | Adapt after exact source mapping |
| Global fusion and WD weighting | MGCMVC | Retain principle; scale handling is unresolved |
| Sample and cluster contrast | MGCMVC | Retain with spatial negative filtering |
| Spatial-aware weighting | SpaMGCL | New required contribution |
| Spatial negative filtering | SpaMGCL | New required contribution; ignored neighbors are not positives |
| Spatial Cluster Refinement | SpaMGCL | Optional Phase II, disabled for the core implementation |
| Final clustering | SMGC experiment style | mclust where available; K-means sensitivity analysis |

## 3. Data contract

The loader must eventually produce a `SpatialMultiOmicsSample` with:

- `modality_features`: modality name to `N x d_m` tensor
- `spatial_coordinates`: `N x 2` tensor
- optional `labels`: length `N`
- `spot_ids`: length `N`, with identical order across modalities
- `graphs`: sparse spatial and modality-specific feature adjacencies
- `metadata`: dataset name, fields, preprocessing state, and source paths

Required invariants:

1. All modalities have the same verified spot order and `N`.
2. Feature graphs are isolated by modality.
3. All graph matrices are `N x N`, finite, nonnegative, and sparse.
4. No NaN or Inf appears after a verified preprocessing path.
5. Labels are optional for training but required for ARI/NMI evaluation.

## 4. Tensor and view contract

For two modalities, the initial graph-view set is:

```text
RNA spatial, RNA feature, second-modality spatial, second-modality feature
```

The feature graph rule is strict:

```text
RNA features -> A_f_RNA only
ADT features -> A_f_ADT only
ATAC features -> A_f_ATAC only
```

Cross-modality interaction starts only after view-specific representations
have been obtained.

Planned shapes:

| Name | Shape |
|---|---|
| `X_m` | `N x d_m` |
| `A_s_m`, `A_f_m` | `N x N` sparse |
| `Z_v` | `N x d_fg` |
| `H_v` | `N x d_cg` |
| `G_v` | `N x d_mg` |
| `U` | `N x d_u` |
| `w` | `V`, nonnegative, sums to approximately one |
| `Q_v`, `P_v` | `N x C` probability matrices |
| `embedding` | `N x d_out` |

## 5. Preprocessing contract

The intended SMGC-style candidates are:

- RNA: filter/HVG selection, normalization, log transform, scaling, PCA.
- ADT: centered log-ratio normalization, scaling, PCA.
- ATAC: top peaks, TF-IDF, LSI.
- Spatial graph: KNN on tissue coordinates.
- Feature graph: Pearson/correlation-distance KNN candidate.

Exact field names, operation order, K values, and whether the supplied files
already contain processed representations are `UNRESOLVED` until Phase 1.

## 6. Loss and training contract

Required first-core terms:

- reconstruction loss
- MGCMVC-style adaptive sample contrastive loss
- MGCMVC-style cluster contrastive loss after warm-up
- spatial smoothness loss and spatial consistency score

Candidate cluster warm-up: optimize reconstruction plus sample contrastive
loss for 10 epochs, then enable cluster loss at epoch 11. This is a candidate
from the specification, not a settled empirical result.

Spatial negative rule:

```text
positive = same spot across different views
ignored = spatial neighbors, excluding self
negative = neither positive nor ignored
```

Ignored neighbors must be masked from the denominator, for example with
`-inf` or a large negative sentinel. They must not be converted into zero
similarity negatives or automatic positives.

## 7. Unresolved items

- AnnData field names for coordinates, labels, layers, and processed features.
- Raw versus processed state of each `.h5ad`.
- Exact preprocessing parameters and graph normalization.
- Exact WD implementation and whether flattening is faithful to the paper.
- Mapping of MGCMVC source variables `z`, `h1`, `h`, and `H` to project terms
  `FG`, `CG`, `MG`, and global representation.
- WD and SC scale handling; standardization plus nonnegative shift is only a
  candidate.
- Cluster count source for each dataset.
- Noise injection location and scale.
- Full-batch memory limit and sampler policy.
- Fine-grained encoder depth.

## 8. Phase boundary

Phase 0 ends after repository audit, skeleton creation, documentation, and
path existence checks. It must not read complete matrices, implement a model,
train, or enter P0 experiments.

