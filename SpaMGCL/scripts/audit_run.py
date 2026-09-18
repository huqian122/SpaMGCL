"""Audit one completed SpaMGCL run for formal-readout consistency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering.predict import cluster_embedding, clustering_metrics
from src.clustering.refinement import (
    boundary_aware_spatial_residual_refinement,
)


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, Mapping) else {}


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _check_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
        raise AssertionError(f"{name} differs: {actual} != {expected}")


def audit_run(output_dir: Path) -> None:
    metrics = _read_json(output_dir / "metrics.json")
    config = yaml.safe_load((output_dir / "config.yaml").read_text(encoding="utf-8"))
    manifest = _read_json(output_dir / "manifest.json")
    if not isinstance(config, Mapping):
        raise ValueError("config.yaml must contain a mapping")

    experiment = _section(config, "experiment")
    training = _section(config, "training")
    loss = _section(config, "loss")
    clustering = _section(config, "clustering")
    refinement = _section(config, "refinement")
    evaluation = _section(config, "evaluation")
    nmi_average_method = str(evaluation.get("nmi_average_method", "max")).lower()
    warm_up_epochs = int(
        training.get("warm_up_epochs", experiment.get("warm_up_epochs", 0))
    )
    cluster_loss_start_epoch = warm_up_epochs + 1
    configured_loss = metrics.get("configured_loss")
    if not isinstance(configured_loss, Mapping):
        raise AssertionError("metrics.json is missing configured_loss")
    for key in ("lambda_rec", "lambda_mgcl", "lambda_cluster", "lambda_spatial"):
        if key not in loss or key not in configured_loss:
            raise AssertionError(f"missing configured loss coefficient: {key}")
        _check_close(float(configured_loss[key]), float(loss[key]), key)

    history = metrics.get("loss_history")
    if not isinstance(history, list) or not history:
        raise AssertionError("metrics.json is missing loss_history")
    configured_lambda_cluster = float(loss["lambda_cluster"])
    for record in history:
        epoch = int(record["epoch"])
        expected_lambda_cluster = (
            0.0 if epoch <= warm_up_epochs else configured_lambda_cluster
        )
        if "effective_lambda_cluster" not in record:
            raise AssertionError(
                f"epoch {epoch} is missing effective_lambda_cluster"
            )
        _check_close(
            float(record["effective_lambda_cluster"]),
            expected_lambda_cluster,
            f"epoch {epoch} effective_lambda_cluster",
        )
        _check_close(
            float(record.get("lambda_cluster")),
            expected_lambda_cluster,
            f"epoch {epoch} lambda_cluster",
        )
        _check_close(
            float(record["lambda_rec"]),
            float(loss["lambda_rec"]),
            f"epoch {epoch} lambda_rec",
        )
        _check_close(
            float(record["lambda_mgcl"]),
            float(loss["lambda_mgcl"]),
            f"epoch {epoch} lambda_mgcl",
        )

    pred = np.load(output_dir / "pred_labels.npy")
    gt = np.load(output_dir / "gt_labels.npy")
    z_concat = np.load(output_dir / "z_concat.npy")
    pred_q = np.load(output_dir / "pred_q_argmax.npy")
    pred_concat = np.load(output_dir / "pred_concat_z_kmeans.npy")
    coords = np.load(output_dir / "coords.npy")

    if pred.ndim != 1 or gt.ndim != 1 or pred_q.ndim != 1:
        raise AssertionError("label arrays must be one-dimensional")
    if len(pred) != len(gt) or len(pred_q) != len(gt):
        raise AssertionError("prediction and GT lengths differ")
    if z_concat.ndim != 2 or z_concat.shape[0] != len(gt):
        raise AssertionError("z_concat shape is inconsistent with GT labels")
    if not np.isfinite(z_concat).all():
        raise AssertionError("z_concat contains NaN or Inf")
    if coords.shape != (len(gt), 2) or not np.isfinite(coords).all():
        raise AssertionError("coords shape or values are invalid")
    if (
        clustering.get("method") != "kmeans"
        or clustering.get("embedding") != "concat_z"
    ):
        raise AssertionError("config does not declare concat_z + kmeans")
    if int(_section(config, "model").get("num_clusters")) != int(
        clustering.get("n_clusters")
    ):
        raise AssertionError("model and clustering cluster counts differ")

    n_clusters = int(clustering["n_clusters"])
    n_init = int(clustering["n_init"])
    random_state = int(clustering["random_state"])
    resolved_parameters = metrics.get("resolved_parameters")
    if not isinstance(resolved_parameters, Mapping):
        raise AssertionError("metrics.json is missing resolved_parameters")
    resolved_clustering = resolved_parameters.get("clustering")
    if not isinstance(resolved_clustering, Mapping):
        raise AssertionError("metrics.json is missing resolved_parameters.clustering")
    expected_clustering = {
        "method": str(clustering["method"]),
        "embedding": str(clustering["embedding"]),
        "n_clusters": n_clusters,
        "n_init": n_init,
        "random_state": random_state,
    }
    for key, expected in expected_clustering.items():
        if resolved_clustering.get(key) != expected:
            raise AssertionError(
                f"resolved_parameters.clustering.{key} differs from config"
            )
    recomputed_raw_pred, raw_method = cluster_embedding(
        z_concat,
        n_clusters,
        method="kmeans",
        n_init=n_init,
        random_state=random_state,
    )
    if raw_method != "kmeans" or not np.array_equal(
        recomputed_raw_pred, pred_concat
    ):
        raise AssertionError("pred_concat_z_kmeans.npy is not reproducible")

    refinement_enabled = bool(refinement.get("enabled", False))
    refinement_method = str(refinement.get("method", "bsrr")).lower()
    refinement_spatial_k = int(refinement.get("spatial_k", 3))
    official_readout = metrics.get("official_readout")
    metric_refinement = metrics.get("refinement")
    if not isinstance(official_readout, Mapping):
        raise AssertionError("metrics.json is missing official_readout metadata")
    if not isinstance(metric_refinement, Mapping):
        raise AssertionError("metrics.json is missing refinement diagnostics")
    expected_readout = {
        "embedding": "concat_z",
        "refinement_enabled": refinement_enabled,
        "refinement_method": refinement_method,
        "clustering_method": "kmeans",
    }
    for key, expected in expected_readout.items():
        if official_readout.get(key) != expected:
            raise AssertionError(f"official_readout.{key} differs from config")
    if bool(metric_refinement.get("enabled")) != refinement_enabled:
        raise AssertionError("refinement.enabled differs between config and metrics")
    if str(metric_refinement.get("method", "")).lower() != refinement_method:
        raise AssertionError("refinement.method differs between config and metrics")
    if int(metric_refinement.get("spatial_k")) != refinement_spatial_k:
        raise AssertionError("refinement.spatial_k differs between config and metrics")

    if refinement_enabled:
        if refinement_method != "bsrr":
            raise AssertionError("enabled refinement method must be bsrr")
        z_bsrr = np.load(output_dir / "z_bsrr.npy")
        recomputed_bsrr, recomputed_refinement = (
            boundary_aware_spatial_residual_refinement(
                z_concat, coords, spatial_k=refinement_spatial_k
            )
        )
        if z_bsrr.shape != z_concat.shape or not np.isfinite(z_bsrr).all():
            raise AssertionError("z_bsrr shape or values are invalid")
        if not np.allclose(z_bsrr, recomputed_bsrr):
            raise AssertionError("z_bsrr.npy does not reproduce BSRR")
        for key in (
            "sigma_spatial",
            "sigma_latent",
            "confidence_mean",
            "confidence_std",
            "confidence_min",
            "confidence_max",
        ):
            _check_close(
                float(metric_refinement[key]),
                float(recomputed_refinement[key]),
                f"refinement.{key}",
            )
        recomputed_official_pred, official_method = cluster_embedding(
            z_bsrr,
            n_clusters,
            method="kmeans",
            n_init=n_init,
            random_state=random_state,
        )
    else:
        if (output_dir / "z_bsrr.npy").exists():
            raise AssertionError("disabled refinement unexpectedly saved z_bsrr.npy")
        if any(
            metric_refinement.get(key) is not None
            for key in (
                "sigma_spatial",
                "sigma_latent",
                "confidence_mean",
                "confidence_std",
                "confidence_min",
                "confidence_max",
            )
        ):
            raise AssertionError("disabled refinement has non-null diagnostics")
        recomputed_official_pred = recomputed_raw_pred
        official_method = raw_method
        if not np.array_equal(pred, pred_concat):
            raise AssertionError("disabled refinement did not use raw concat-Z")
    if official_method != "kmeans" or not np.array_equal(
        recomputed_official_pred, pred
    ):
        raise AssertionError("pred_labels.npy is not the configured official readout")

    recomputed = clustering_metrics(
        gt, pred, nmi_average_method=nmi_average_method
    )
    _check_close(float(metrics["ARI"]), recomputed["ARI"], "ARI")
    _check_close(float(metrics["NMI"]), recomputed["NMI"], "NMI")
    q_metrics = clustering_metrics(
        gt, pred_q, nmi_average_method=nmi_average_method
    )
    required_manifest = {
        "project_root",
        "python_version",
        "torch_version",
        "numpy_version",
        "scipy_version",
        "sklearn_version",
        "anndata_version",
        "device",
        "cuda_version",
        "source_config_path",
    }
    missing_manifest = required_manifest - set(manifest)
    if missing_manifest:
        raise AssertionError(f"manifest missing fields: {sorted(missing_manifest)}")

    epochs = training.get("epochs", experiment.get("epochs"))
    print(f"Dataset: {metrics.get('dataset', experiment.get('dataset'))}")
    print(f"Seed: {metrics.get('seed', experiment.get('seed'))}")
    print(f"Cluster warm-up epochs: {warm_up_epochs}")
    print(f"Cluster loss start epoch: {cluster_loss_start_epoch}")
    print(f"Configured lambda_cluster: {configured_lambda_cluster}")
    print(f"Official embedding: {metrics.get('clustering_embedding')}")
    print(f"Official clustering method: {metrics.get('clustering_method_used')}")
    print(f"KMeans n_init: {n_init}")
    print(f"KMeans random_state: {random_state}")
    print(f"Refinement enabled: {refinement_enabled}")
    print(f"Refinement method: {refinement_method}")
    print(f"Refinement spatial_k: {refinement_spatial_k}")
    print(f"Epochs: {epochs}")
    print(f"lambda_rec: {loss.get('lambda_rec')}")
    print(f"lambda_mgcl: {loss.get('lambda_mgcl')}")
    print(f"lambda_cluster: {loss.get('lambda_cluster')}")
    print(f"lambda_spatial: {loss.get('lambda_spatial')}")
    print(f"ARI: {metrics['ARI']}")
    print(f"NMI: {metrics['NMI']}")
    print(f"NMI average method: {nmi_average_method}")
    print(f"recomputed ARI: {recomputed['ARI']}")
    print(f"recomputed NMI: {recomputed['NMI']}")
    print(f"Q argmax diagnostic ARI: {q_metrics['ARI']}")
    print(f"Q argmax diagnostic NMI: {q_metrics['NMI']}")
    print(f"z_concat shape: {tuple(z_concat.shape)}")
    print("PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    try:
        audit_run(args.output_dir.resolve())
    except (
        AssertionError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
