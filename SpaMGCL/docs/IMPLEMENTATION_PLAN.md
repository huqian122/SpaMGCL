# Implementation Plan

Status: Phase 0 planning artifact, 2026-09-05

## Implementation phases

These are software implementation phases, not paper experiments.

| Phase | Scope | Required artifact | Stop rule |
|---|---|---|---|
| Phase 0 | Repository audit and skeleton | source maps, project spec, status | Stop before model/data inspection |
| Phase 1 | Data inspection and loader contract | `DATA_PROFILE.md`, focused tests | Stop after field/alignment report |
| Phase 2 | Spatial/feature graphs and graph views | graph statistics and validation | Stop after Simulation graph validation |
| Phase 3 | MGCMVC-original migration | baseline smoke artifacts | No SpaMGCL spatial adaptations |
| Phase 4 | Core SpaMGCL | MG, spatial weights, SNF feasibility | SCR remains disabled |
| Phase 5 | Clustering/evaluation integration | metrics and visual outputs | No new method changes |
| Phase 6 | Optional Spatial Cluster Refinement | separate evidence | Only after core stability |

## Experiment phases

These are paper/benchmark phases and must not be conflated with the
implementation phases above.

| Experiment | Scope | Purpose |
|---|---|---|
| P0 | Simulation, seed 0, 2-5 epochs | smoke test only; loader, graph, forward, finite losses |
| P1 | Simulation, HLN-A1, E18.5; seeds 0/1/2 | feasibility: MGCMVC-original versus SpaMGCL |
| P2 | all six datasets, main baselines, 10 seeds | primary ARI/NMI table |
| P3 | Base, +MG, +MG+SW, +MG+SW+SNF | mechanism ablation |
| P4 | diagnosed data scale, modality perturbation | robustness and learned weights |
| P5 | FG/CG/MG/Global visualization and interpretation | figures and biological checks |

No experiment phase is entered by this Phase 0 run.

## Module order

1. Create a read-only-safe, config-driven path and field inspection layer.
2. Verify modality alignment, labels, coordinates, and preprocessing state.
3. Implement sparse spatial and modality-isolated feature graph builders.
4. Implement SMGC graph-view construction.
5. Reimplement and validate the MGCMVC-original core.
6. Add spatial-aware weighting and spatial negative filtering.
7. Add clustering, metrics, outputs, and phase-controlled experiment runners.

## Acceptance gates

### Phase 1

- Every `.h5ad` is inspected by metadata and bounded diagnostics only.
- No complete data matrix is printed or exported.
- Actual `shape`, X type/sparsity/dtype/range summary, fields, and raw/layer
  state are recorded.
- Cross-modality order and coordinate/label fields are verified or reported as
  failures/`UNRESOLVED`.

### Phase 2

- Spatial coordinates are verified before graph construction.
- Feature graphs use only their own modality feature tensor.
- Adjacency shapes, finite values, symmetry policy, self-loop policy, and
  sparsity are recorded.

### Phase 3

- The four SMGC graph views are available.
- MGCMVC-original is isolated from spatial-aware weighting and SNF.
- Any unresolved `z/h1/h/H` mapping remains explicit in the run report.

### Phase 4

- MG, spatial-aware weighting, and SNF can be independently switched.
- Ignored spatial neighbors are removed from denominators, not labeled positive.
- WD/SC scale choice is recorded as a candidate and diagnostics are saved.
- SCR is false by default.

## Configuration rules

All data and output locations are configuration values. No source file may
hard-code the local Mac or Colab data root. The dataset registry should map
short names to relative paths and filenames, then resolve them against
`data.root`.

The first configuration candidate is:

```yaml
experiment:
  name: p0_smoke_simulation
  phase: P0
  dataset: Simulation
  seed: 0
data:
  root: /path/to/SMGC-data
  label_key: UNRESOLVED_AFTER_INSPECTION
  spatial_key: UNRESOLVED_AFTER_INSPECTION
graphs:
  spatial:
    k: UNRESOLVED_AFTER_SOURCE_CHECK
  feature:
    metric: pearson_correlation_distance
    k: UNRESOLVED_AFTER_SOURCE_CHECK
model:
  use_spatial_cluster_refinement: false
loss:
  wd_sc_scale_mode: standardize_across_views_candidate
noise:
  enabled: false
```

This is a design record only; no runnable training configuration is created in
Phase 0.

