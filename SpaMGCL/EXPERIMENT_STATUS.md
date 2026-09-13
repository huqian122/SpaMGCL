# Experiment Status

## Current change

- Audited the SC and `lambda_spatial` paths in the core runner.
- Kept the existing SNF mask and denominator behavior unchanged; added only a
  `neg_count` diagnostic for the remaining off-diagonal negative candidates.
- SC-aware view weights are still passed directly to the MGCL loss. Each epoch
  now records `mgcl_weight_std` so SC on/off runs can be compared quantitatively.
- `SpaMGCLForwardOutput.total_loss` now uses the four coefficients supplied by
  the runner. The training objective is explicitly
  `rec + mgcl + cluster + lambda_spatial * spatial`.
- Added stable loss-history fields: `lambda_spatial`, `spatial_loss`,
  `sc_enabled`, `snf_enabled`, `mgcl_weight_std`, and `neg_count`.

No training was run in this change, and existing `results/` directories were
not modified.

## Kaggle acceptance commands

Run each command in the repository root, using a new `experiment.name` and
the configured `output.root` so no historical result is overwritten:

```bash
python -m py_compile src/losses/sample_contrastive.py src/models/spamgcl.py experiments/run_exp.py
python experiments/run_exp.py --config configs/e185_ablation_core.yaml
python experiments/run_exp.py --config configs/e185_ablation_sc.yaml
python experiments/run_exp.py --config configs/e185_ablation_sc_snf.yaml
```

For the lambda check, use a new copied config with
`loss.lambda_spatial: 0.1`, `spatial.enabled: true`, and a start epoch that is
reached. Do not reuse a result directory containing `metrics.json`.

## Expected observations

- The core and SC runs should show different `mgcl_weight_std` values when the
  SC scores are non-identical; the SC run's sample MGCL is computed with those
  SC-aware weights.
- With `spatial.enabled: true`, `spatial.consistency_weighting: false`, and
  `spatial.negative_filter: true`, only SNF is active; `neg_count` should be
  smaller than the unfiltered core value while the SNF implementation itself is
  unchanged.
- Once spatial training is active, a nonzero `lambda_spatial` makes total loss
  differ from the non-spatial objective by exactly
  `lambda_spatial * spatial_loss`, up to floating-point rounding.
