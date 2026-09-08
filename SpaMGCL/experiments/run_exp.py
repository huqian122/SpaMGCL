"""Run one config-driven SpaMGCL experiment.

This entry point is intentionally a full-batch runner for the current P0
smoke experiment. Later phases can add batching after the memory contract for
the largest ATAC matrices has been measured.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from src.clustering.predict import cluster_embedding, clustering_metrics
from src.data.dataset import load_spatial_multiomics, sample_summary
from src.data.preprocessing import prepare_features
from src.graphs.feature_graph import build_feature_graph
from src.graphs.spatial_graph import build_spatial_graph
from src.models.spamgcl import SpaMGCL


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {config_path}")
    return config


def _resolve_config_path(raw_path: Path) -> Path:
    """Accept paths from either the repository root or its parent directory."""

    candidates = [raw_path.expanduser()]
    if not raw_path.is_absolute():
        candidates.append(PROJECT_ROOT / raw_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(f"Config file not found; searched: {searched}")


def _resolve_path(value: Any, *, base: Path, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config must define a non-empty string at {field_name}")
    expanded = Path(os.path.expandvars(value)).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (base / expanded).resolve()


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return value


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _device_from_config(config: Mapping[str, Any]) -> torch.device:
    requested = str(config.get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but CUDA is unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'mps'")
    return torch.device(requested)


def _to_sparse_tensor(adjacency: Any, device: torch.device) -> torch.Tensor:
    """Convert a SciPy sparse adjacency once before the training loop."""

    coo = adjacency.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data)
    return torch.sparse_coo_tensor(
        indices, values, size=coo.shape, device=device
    ).coalesce()


def _preprocessing_options(
    config: Mapping[str, Any], modality: str
) -> Mapping[str, Any]:
    options = _section(config, "preprocessing")
    modality_options = options.get(modality)
    if isinstance(modality_options, Mapping):
        return modality_options
    return options


def _json_float(value: torch.Tensor) -> float:
    result = float(value.detach().cpu().item())
    if not np.isfinite(result):
        raise FloatingPointError("non-finite loss encountered")
    return result


def _gradient_norm(module: torch.nn.Module) -> float:
    """Return the post-backward L2 norm for a module's trainable gradients."""

    squared_norm = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared_norm += float(parameter.grad.detach().pow(2).sum().cpu())
    return float(np.sqrt(squared_norm))


def _build_model_config(config: Mapping[str, Any], num_clusters: int) -> Dict[str, Any]:
    model = dict(_section(config, "model"))
    model.setdefault("gcn_hidden_dim", 64)
    model.setdefault("fine_dim", 32)
    model.setdefault("coarse_dim", 32)
    model.setdefault("representation_dim", 32)
    model.setdefault("fusion_hidden_dim", 128)
    model.setdefault("alpha", 0.5)
    model.setdefault("temperature", 0.5)
    model.setdefault("cluster_temperature", 1.0)
    model.setdefault("cluster_regularization_weight", 1.0)
    configured_model_clusters = model.get("num_clusters")
    if configured_model_clusters is not None and int(configured_model_clusters) != num_clusters:
        raise ValueError(
            "model.num_clusters and clustering.n_clusters must match: "
            f"{configured_model_clusters} != {num_clusters}"
        )
    model["num_clusters"] = num_clusters
    return model


def _build_graphs(
    config: Mapping[str, Any],
    features: Mapping[str, np.ndarray],
    coordinates: np.ndarray,
) -> Tuple[Any, Dict[str, Any]]:
    graphs = _section(config, "graphs")
    spatial_options = graphs.get("spatial", {})
    feature_options = graphs.get("feature", {})
    if not isinstance(spatial_options, Mapping) or not isinstance(feature_options, Mapping):
        raise ValueError("graphs.spatial and graphs.feature must be mappings")

    spatial_adjacency = build_spatial_graph(
        coordinates,
        k=int(spatial_options.get("k", 3)),
        include_self=bool(spatial_options.get("include_self", False)),
        symmetrize=bool(spatial_options.get("symmetrize", True)),
    )
    feature_adjacencies = {
        modality: build_feature_graph(
            matrix,
            k=int(feature_options.get("k", 20)),
            symmetrize=bool(feature_options.get("symmetrize", True)),
            clip_negative=bool(feature_options.get("clip_negative", True)),
        )
        for modality, matrix in features.items()
    }
    return spatial_adjacency, feature_adjacencies


def _effective_loss_coefficients(
    config: Mapping[str, Any], epoch_number: int, warm_up_epochs: int
) -> Tuple[float, float, float, float, bool]:
    loss = _section(config, "loss")
    lambda_rec = float(loss.get("lambda_rec", config.get("lambda_rec", 1.0)))
    lambda_mgcl = float(loss.get("lambda_mgcl", config.get("lambda_mgcl", 1.0)))
    lambda_cluster = float(
        loss.get("lambda_cluster", config.get("lambda_cluster", 1.0))
    )
    lambda_spatial = float(
        loss.get("lambda_spatial", config.get("lambda_spatial", 0.0))
    )
    if min(lambda_rec, lambda_mgcl, lambda_cluster, lambda_spatial) < 0:
        raise ValueError("loss coefficients must be nonnegative")

    warm_up = epoch_number <= warm_up_epochs
    if warm_up:
        return lambda_rec, lambda_mgcl, 0.0, 0.0, False
    return lambda_rec, lambda_mgcl, lambda_cluster, lambda_spatial, True


def run_experiment(config_path: Path) -> Dict[str, Any]:
    config = _load_config(config_path)
    experiment = _section(config, "experiment")
    data_config = _section(config, "data")
    training_config = _section(config, "training")
    clustering_config = _section(config, "clustering")

    dataset = str(experiment.get("dataset", config.get("dataset", "")))
    if not dataset:
        raise ValueError("Config must define experiment.dataset or dataset")
    seed = int(experiment.get("seed", config.get("seed", 0)))
    epochs = int(
        training_config.get("epochs", experiment.get("epochs", config.get("epochs", 0)))
    )
    warm_up_epochs = int(
        training_config.get(
            "warm_up_epochs",
            experiment.get("warm_up_epochs", config.get("warm_up_epochs", 10)),
        )
    )
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if warm_up_epochs < 0:
        raise ValueError("warm_up_epochs cannot be negative")

    data_root = _resolve_path(data_config.get("root"), base=PROJECT_ROOT, field_name="data.root")
    label_key = data_config.get("label_key")
    spatial_key = str(data_config.get("spatial_key", "spatial"))
    spatial_row_key = str(data_config.get("spatial_row_key", "array_row"))
    spatial_col_key = str(data_config.get("spatial_col_key", "array_col"))
    matrix_source = str(data_config.get("matrix_source", "X"))
    n_pca_components = data_config.get("n_pca_components")
    if n_pca_components is not None:
        n_pca_components = int(n_pca_components)

    _seed_everything(seed)
    device = _device_from_config(config)
    sample = load_spatial_multiomics(
        data_root,
        dataset,
        label_key=label_key,
        spatial_key=spatial_key,
        spatial_row_key=spatial_row_key,
        spatial_col_key=spatial_col_key,
        matrix_source=matrix_source,
        n_pca_components=n_pca_components,
    )
    features = {
        modality: prepare_features(
            matrix, modality, _preprocessing_options(config, modality)
        )
        for modality, matrix in sample.modality_features.items()
    }
    modality_names = tuple(features)
    if len(modality_names) != 2:
        raise ValueError("SpaMGCL runner requires exactly two modalities")
    first_modality, second_modality = modality_names

    spatial_adjacency, feature_adjacencies = _build_graphs(
        config, features, sample.spatial_coordinates
    )
    spatial_tensor = _to_sparse_tensor(spatial_adjacency, device)
    feature_tensors = {
        modality: _to_sparse_tensor(adjacency, device)
        for modality, adjacency in feature_adjacencies.items()
    }

    inputs = {
        modality: torch.from_numpy(matrix).to(device=device, dtype=torch.float32)
        for modality, matrix in features.items()
    }
    labels = sample.labels
    if labels is None:
        raise ValueError("ARI/NMI evaluation requires a configured label field")

    default_cluster_count = int(np.unique(labels).size)
    model_cluster_count = _section(config, "model").get("num_clusters")
    clustering_cluster_count = clustering_config.get("n_clusters")
    configured_cluster_counts = [
        int(value)
        for value in (model_cluster_count, clustering_cluster_count)
        if value is not None
    ]
    if len(set(configured_cluster_counts)) > 1:
        raise ValueError(
            "model.num_clusters and clustering.n_clusters must match; "
            f"received {configured_cluster_counts}"
        )
    num_clusters = configured_cluster_counts[0] if configured_cluster_counts else default_cluster_count
    if num_clusters < 2:
        raise ValueError("num_clusters must be at least 2")
    model_config = _build_model_config(config, num_clusters)
    model = SpaMGCL(
        input_dims={modality: int(matrix.shape[1]) for modality, matrix in features.items()},
        **model_config,
    ).to(device)
    learning_rate = float(training_config.get("lr", config.get("lr", 1e-3)))
    weight_decay = float(training_config.get("weight_decay", config.get("weight_decay", 1e-5)))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"Dataset: {dataset} | spots={sample.n_spots} | device={device}")
    print(f"Modalities: {', '.join(modality_names)} | label={sample.label_key}")
    print(f"Spatial graph: shape={spatial_adjacency.shape}, nnz={spatial_adjacency.nnz}")

    history = []
    for epoch_index in range(epochs):
        epoch_number = epoch_index + 1
        model.train()
        output = model(
            inputs[first_modality],
            spatial_tensor,
            feature_tensors[first_modality],
            inputs[second_modality],
            spatial_tensor,
            feature_tensors[second_modality],
            spatial_tensor,
            modality_a_name=first_modality,
            modality_b_name=second_modality,
        )
        coefficients = _effective_loss_coefficients(
            config, epoch_number, warm_up_epochs
        )
        lambda_rec, lambda_mgcl, lambda_cluster, lambda_spatial, cluster_enabled = coefficients
        total_loss = (
            lambda_rec * output.reconstruction_loss
            + lambda_mgcl * output.sample_contrastive_loss
            + lambda_cluster * output.cluster_contrastive_loss
            + lambda_spatial * output.spatial_loss
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"non-finite total loss at epoch {epoch_number}")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        cluster_head_gradient_norm = _gradient_norm(model.cluster_head)
        optimizer.step()

        cluster_diagnostics = model.cluster_contrastive_loss.diagnostics(
            output.cluster_assignments
        )

        record = {
            "epoch": epoch_number,
            "total": _json_float(total_loss),
            "reconstruction": _json_float(output.reconstruction_loss),
            "sample_contrastive": _json_float(output.sample_contrastive_loss),
            "cluster_contrastive": _json_float(output.cluster_contrastive_loss),
            "spatial": _json_float(output.spatial_loss),
            "lambda_rec": lambda_rec,
            "lambda_mgcl": lambda_mgcl,
            "lambda_cluster": lambda_cluster,
            "lambda_spatial": lambda_spatial,
            "cluster_enabled": cluster_enabled,
            "cluster_assignment_entropy": _json_float(
                cluster_diagnostics["assignment_entropy"]
            ),
            "cluster_max_probability": _json_float(
                cluster_diagnostics["max_probability"]
            ),
            "cluster_effective_clusters": _json_float(
                cluster_diagnostics["effective_clusters"]
            ),
            "cluster_min_mass": _json_float(cluster_diagnostics["min_cluster_mass"]),
            "cluster_max_mass": _json_float(cluster_diagnostics["max_cluster_mass"]),
            "cluster_head_gradient_norm": cluster_head_gradient_norm,
        }
        history.append(record)
        print(
            f"epoch {epoch_number:03d}/{epochs:03d} | "
            f"total={record['total']:.6f} | rec={record['reconstruction']:.6f} | "
            f"mgcl={record['sample_contrastive']:.6f} | "
            f"cluster={record['cluster_contrastive']:.6f} | "
            f"spatial={record['spatial']:.6f} | "
            f"effC={record['cluster_effective_clusters']:.3f} | "
            f"gradC={record['cluster_head_gradient_norm']:.3e}"
        )

    model.eval()
    with torch.no_grad():
        final_output = model(
            inputs[first_modality],
            spatial_tensor,
            feature_tensors[first_modality],
            inputs[second_modality],
            spatial_tensor,
            feature_tensors[second_modality],
            spatial_tensor,
            modality_a_name=first_modality,
            modality_b_name=second_modality,
        )
    embedding = final_output.global_representation.detach().cpu().numpy()
    method = str(clustering_config.get("method", "kmeans"))
    predicted, method_used = cluster_embedding(
        embedding, num_clusters, method=method, seed=seed
    )
    metrics = clustering_metrics(labels, predicted)
    final_cluster_diagnostics = model.cluster_contrastive_loss.diagnostics(
        final_output.cluster_assignments
    )

    output_section = _section(config, "output")
    output_value = output_section.get(
        "root",
        output_section.get(
            "dir", config.get("output_dir", config.get("output", "results/p0_smoke"))
        ),
    )
    output_dir = _resolve_path(output_value, base=PROJECT_ROOT, field_name="output.root")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_config_path = output_dir / "config.yaml"
    if config_path.resolve() != output_config_path.resolve():
        shutil.copy2(config_path, output_config_path)

    report: Dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "warm_up_epochs": warm_up_epochs,
        "cluster_enabled_epochs": [
            record["epoch"] for record in history if record["cluster_enabled"]
        ],
        "clustering_method_requested": method,
        "clustering_method_used": method_used,
        "num_clusters": num_clusters,
        "ARI": metrics["ARI"],
        "NMI": metrics["NMI"],
        "metrics": metrics,
        "final_weights": [float(value) for value in final_output.weights.detach().cpu()],
        "final_cluster_diagnostics": {
            key: _json_float(value) for key, value in final_cluster_diagnostics.items()
        },
        "data": sample_summary(sample),
        "graph_stats": {
            "spatial": {"shape": list(spatial_adjacency.shape), "nnz": int(spatial_adjacency.nnz)},
            "feature": {
                modality: {"shape": list(adjacency.shape), "nnz": int(adjacency.nnz)}
                for modality, adjacency in feature_adjacencies.items()
            },
        },
        "loss_history": history,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"ARI={report['ARI']:.6f} | NMI={report['NMI']:.6f}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved config copy: {output_config_path}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config_path = _resolve_config_path(args.config)
    run_experiment(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
