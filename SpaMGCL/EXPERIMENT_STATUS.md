# Experiment Status

## Current change

- Audited the SC and `lambda_spatial` paths in the core runner.
- Kept the existing SNF mask and denominator behavior unchanged; added only a
  `neg_count` diagnostic for the remaining off-diagonal negative candidates,
  plus `snf_masked_positions` and `unfiltered_neg_count` diagnostics.
- SC-aware view weights are still passed directly to the MGCL loss. Each epoch
  now records `mgcl_weight_std` so SC on/off runs can be compared quantitatively.
- `SpaMGCLForwardOutput.total_loss` now uses the four coefficients supplied by
  the runner. The training objective is explicitly
  `rec + mgcl + cluster + lambda_spatial * spatial`.
- Added stable loss-history fields: `lambda_spatial`, `spatial_loss`,
  `sc_enabled`, `snf_enabled`, `mgcl_weight_std`, `neg_count`, and
  `snf_masked_positions`.

## SNF audit

- `experiments/run_exp.py:358-365` computes SNF independently from SC:
  `spatial_negative_filter_enabled` depends on `spatial.enabled` and
  `spatial.negative_filter`, not on `consistency_weighting`.
- `src/models/spamgcl.py:170-177` passes `spatial_adjacency` to MGCL only when
  the SNF flag is true.
- `src/losses/sample_contrastive.py:87-121` converts non-diagonal spatial
  edges into the ignore mask and fills those denominator logits with `-1e9`.
  SNF therefore changes the negative set; it does not turn neighbors into
  positives. The new counters expose both the masked and remaining candidate
  counts without changing that behavior.

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

## Reproducibility artifacts

- `experiments/run_exp.py:696-739` writes `metrics_epoch50.json`,
  `metrics_epoch100.json`, and `metrics_epoch200.json` when those milestones
  are reached. Each snapshot contains the milestone ARI/NMI and the loss
  history through that epoch.
- `experiments/run_exp.py:826-854` writes the final prediction, GT labels,
  spot IDs, coordinates, clustering input, `mean_z`, `concat_z`, SC/WD view
  weights, and SNF statistics. The final matrix is named `Q` in the runner,
  and `pred_labels.npy` is explicitly `Q.argmax(dim=-1)` with shape
  `(n_spots,)`.
- The same block writes `checkpoint_epoch50.pt`, `checkpoint_epoch100.pt`,
  and `checkpoint_epoch200.pt` at reached milestones, plus
  `checkpoint_last.pt`. Checkpoints contain model state, optimizer state,
  epoch, and Python/NumPy/PyTorch RNG states.
- All files are written below the reserved `output.root / experiment.name`
  directory. The existing output reservation still rejects a directory that
  already contains `metrics.json`; no timestamp or historical result is
  overwritten.

## Artifact smoke checks

These commands are for Kaggle and were not run here:

```bash
python -m py_compile experiments/run_exp.py
python experiments/run_exp.py --config configs/hlna1_best.yaml
python - <<'PY'
import json
from pathlib import Path
import numpy as np

out = Path("results/hlna1_best")
required = [
    "pred_labels.npy", "gt_labels.npy", "spot_ids.npy", "coords.npy",
    "embeddings.npy", "z_mean.npy", "z_concat.npy", "sc_weights.npy",
    "snf_stats.json", "metrics_epoch50.json", "metrics.json",
]
for name in required:
    assert (out / name).is_file(), name
    print(name, end=" ")
print()
snapshot = json.loads((out / "metrics_epoch50.json").read_text())
final = json.loads((out / "metrics.json").read_text())
assert np.isclose(snapshot["ARI"], final["ARI"])
assert np.isclose(snapshot["NMI"], final["NMI"])
print("artifact_smoke_ok")
PY
```
