# Source Code Map

Status: Phase 0 audit artifact, 2026-09-05

Reference directories were inspected read-only. No reference file was
modified.

## MGCMVC reference

Root: `91MGCMVC/`

| Reference file | Observed role | Planned destination |
|---|---|---|
| `MGCMVC的代码/network.py` | per-view encoder/decoder, `z`, `h1`, `h`, `q`, global `H` | `src/models/multigranularity.py`, `src/models/spamgcl.py` |
| `MGCMVC的代码/loss.py` | feature contrast, label contrast, target distribution | `src/losses/sample_contrastive.py`, `src/losses/cluster_contrastive.py` |
| `MGCMVC的代码/train.py` | pretraining, pairwise view training, flattened SciPy WD weighting | `experiments/`, `src/models/adaptive_weight.py` |
| `MGCMVC的代码/dataloader.py` | `.mat` multi-view datasets, unrelated to spatial AnnData contract | no direct copy; `src/data/dataset.py` |
| `MGCMVC的代码/metric.py` | reference clustering metrics/evaluation | `src/clustering/` |
| `README.md` | original run instructions and dataset context | docs only |
| `requirements.txt` | historical environment snapshot | dependency review only |

### MGCMVC variable mapping

Confirmed from source:

- `z`: output of the per-view `AutoEncoder.encoder`.
- `h1`: output of `instance_head(z)`, normalized.
- `h`: normalized fusion of concatenated `h1` and `z` through
  `feature_fusion_module`.
- `H`: normalized global fusion of all per-view `h` tensors.
- `q`: softmax cluster output from `z`.

Project-term mapping is not fully confirmed:

- `z` may serve as a fine-grained or pre-fusion representation, but the
  source does not name it `FG`.
- `h1` is an instance-head projection, not demonstrably the specification's
  `CG`.
- `h` is a fused per-view representation and may correspond to a candidate
  `MG`, but this is not proven by source alone.
- `H` is a global fusion, while the specification uses `U`; this is a strong
  structural correspondence but remains `UNRESOLVED` until paper/source
  reconciliation.

No guess is authorized in implementation code.

## SMGC reference

Root: `Expert Systems SMGC/`

| Reference file | Observed role | Planned destination |
|---|---|---|
| `SMGC-代码/preprocess.py` | preprocessing helpers, spatial graph, correlation KNN feature graph, sparse normalization, LSI/TF-IDF | `src/data/preprocessing.py`, `src/graphs/` |
| `SMGC-代码/graph_GCN.py` | modality-specific GCN and four graph-specific representations | `src/models/gcn.py`, `src/graphs/` |
| `SMGC-代码/model.py` | spatial/feature encoders, within- and cross-modality attention, reconstruction path | reference only; do not confuse with planned MGCMVC core |
| `SMGC-代码/model/autoencoder.py` | multiview AE and encoder/decoder shape pattern | `src/models/multigranularity.py` candidate |
| `SMGC-代码/model/loss.py` | reconstruction/contrastive helper losses | reference only |
| `SMGC-代码/granular/` | granular-ball data structures and multiview granular contrast | reference only for provenance |
| `SMGC-代码/train.py` | SMGC training loop and four-view batching | experiment integration reference |
| `SMGC-代码/utils.py` | mclust wrapper, clustering, plotting helpers | `src/clustering/` and later visualization |
| `main.py` | top-level pipeline, currently hard-coded paths and fields | reference entry flow only; must be config-driven |
| `requirements.txt` | historical Python/R dependency snapshot | dependency verification later |

Observed source details requiring verification in Phase 1/2:

- `preprocess.py` directly assumes `obsm['spatial']`, `obsm['feat']`, and
  `obs['ground_truth']` in its current workflow.
- SMGC's current main path uses `obs['Spatial_Label']` to create
  `ground_truth`.
- Spatial KNN defaults to 3, with a datatype-specific override to 6.
- Feature KNN defaults to `k=20`, connectivity mode, correlation metric.
- Spatial edges are symmetrized and normalized with self-loops.
- Feature adjacency is built independently for each modality, then
  symmetrized and normalized.
- The current GCN implementation reuses one modality encoder per modality for
  both spatial and feature adjacency.

These observations are source facts, not a decision to hard-code them. Exact
values and field names remain subject to actual data inspection and config.

## Planned new entry points

| New path | Phase 0 status |
|---|---|
| `src/data/dataset.py` | skeleton only; no loader implementation |
| `src/data/preprocessing.py` | skeleton only |
| `src/graphs/spatial_graph.py` | skeleton only |
| `src/graphs/feature_graph.py` | skeleton only |
| `src/models/gcn.py` | skeleton only |
| `src/models/multigranularity.py` | skeleton only |
| `src/models/adaptive_weight.py` | skeleton only |
| `src/models/spamgcl.py` | skeleton only |
| `src/losses/reconstruction.py` | skeleton only |
| `src/losses/sample_contrastive.py` | skeleton only |
| `src/losses/cluster_contrastive.py` | skeleton only |
| `src/losses/spatial_regularization.py` | skeleton only |
| `experiments/run_exp.py` | not created; implementation is out of scope |

