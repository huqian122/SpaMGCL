# SpaMGCL

Spatial Multi-Granularity Adaptive Contrastive Learning for spatial multi-omics
domain clustering.

This repository is being implemented in staged phases. Phase 0 is a repository
audit and project skeleton only. Raw reference code and `.h5ad` files remain
outside this writable project directory and are read-only.

## Planned layout

- `src/`: implementation modules
- `configs/`: experiment configurations
- `experiments/`: experiment entry points
- `scripts/`: inspection and utility scripts
- `tests/`: focused tests
- `docs/`: specifications and source maps
- `results/`: generated experiment artifacts

## Colab entry point

```text
python SpaMGCL/experiments/run_exp.py --config configs/xxx.yaml
```

The executable entry point is intentionally not implemented in Phase 0.

