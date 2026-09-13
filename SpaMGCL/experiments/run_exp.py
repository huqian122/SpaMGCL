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


def _prepare_output_dir(
    config: Mapping[str, Any], config_path: Path
) -> Tuple[Path, Path]:
    """Resolve and reserve the immutable output location for one run.

    ``output.root`` is the current format. ``output.dir`` is retained for
    compatibility, but its final path is treated as a legacy hint and its
    parent becomes the root for the new ``root / experiment.name`` layout.
    """

    output = _section(config, "output")
    if "root" in output and output.get("root") is not None:
        root_value = output.get("root")
        root_field = "output.root"
    elif "dir" in output and output.get("dir") is not None:
        legacy_value = output.get("dir")
        if not isinstance(legacy_value, str) or not legacy_value.strip():
            raise ValueError("Config must define a non-empty string at output.dir")
        root_value = str(Path(legacy_value).expanduser().parent)
        root_field = "output.dir parent"
    else:
        raise ValueError(
            "Config must define output.root or output.dir; "
            "the output location is required"
        )

    root = _resolve_path(root_value, base=PROJECT_ROOT, field_name=root_field)
    experiment = _section(config, "experiment")
    experiment_name = experiment.get("name")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("Config must define a non-empty string at experiment.name")
    output_dir = (root / experiment_name.strip()).resolve()
    if output_dir == root:
        raise ValueError("experiment.name must identify a child output directory")
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        raise FileExistsError(
            f"Output directory already contains metrics.json: {output_dir}. "
            "Choose a different experiment.name or output root."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "config.yaml"
    if config_path.resolve() != snapshot_path.resolve():
        shutil.copy2(config_path, snapshot_path)
    return root, output_dir


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


def _embedding_from_output(output: Any, mode: str) -> torch.Tensor:
    """Select the representation that is actually sent to clustering."""

    mode = mode.lower()
    if mode == "global":
        return output.global_representation
    if mode == "weighted_views":
        return output.weighted_representation
    if mode == "mean_views":
        return output.mean_representation
    if mode == "cluster_probabilities":
        return torch.stack(list(output.cluster_assignments), dim=0).mean(dim=0)
    if mode == "concat_views":
        return torch.cat(
            [output.multigranularity_views[name]["g"] for name in output.multigranularity_views],
            dim=1,
        )
    if mode == "mean_z":
        return torch.stack(
            [output.multigranularity_views[name]["z"] for name in output.multigranularity_views],
            dim=0,
        ).mean(dim=0)
    if mode == "concat_z":
        return torch.cat(
            [output.multigranularity_views[name]["z"] for name in output.multigranularity_views],
            dim=1,
        )
    raise ValueError(
        "clustering.embedding must be one of: global, weighted_views, "
        "mean_views, cluster_probabilities, concat_views, mean_z, concat_z"
    )


def _cluster_embedding(
    embedding: np.ndarray,
    n_clusters: int,
    *,
    method: str,
    seed: int,
) -> Tuple[np.ndarray, str]:
    """Cluster a representation, including MGCMVC's direct Q argmax path."""

    if method.lower() == "argmax":
        if embedding.ndim != 2 or embedding.shape[1] != n_clusters:
            raise ValueError(
                "argmax clustering requires an N x n_clusters probability matrix"
            )
        return np.argmax(embedding, axis=1), "argmax"
    return cluster_embedding(embedding, n_clusters, method=method, seed=seed)


def _build_model_config(config: Mapping[str, Any], num_clusters: int) -> Dict[str, Any]:
    model = dict(_section(config, "model"))
    model.setdefault("gcn_hidden_dim", 64)
    model.setdefault("fine_dim", 32)
    model.setdefault("coarse_dim", 32)
    model.setdefault("representation_dim", 32)
    model.setdefault("fusion_hidden_dim", 128)
    model.setdefault("alpha", 0.5)
    # ``tau_s`` is the paper-facing name for the sample-level temperature.
    # Translate it here so the model keeps its stable Python API.
    tau_s = model.pop("tau_s", None)
    if tau_s is not None:
        model["temperature"] = float(tau_s)
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
    config: Mapping[str, Any],
    epoch_number: int,
    warm_up_epochs: int,
    spatial_start_epoch: Optional[int] = None,
    spatial_mechanism_enabled: bool = True,
) -> Tuple[float, float, float, float, bool, bool]:
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

    # Keep cluster activation and spatial regularization independently
    # schedulable. The default preserves the old behavior, while P1 can
    # delay spatial regularization to measure the MGCMVC structure first.
    if spatial_start_epoch is None:
        spatial_start_epoch = warm_up_epochs + 1
    if spatial_start_epoch < 1:
        raise ValueError("spatial_start_epoch must be positive")

    cluster_enabled = epoch_number > warm_up_epochs
    spatial_enabled = (
        spatial_mechanism_enabled and epoch_number >= spatial_start_epoch
    )
    return (
        lambda_rec,
        lambda_mgcl,
        lambda_cluster if cluster_enabled else 0.0,
        lambda_spatial if spatial_enabled else 0.0,
        cluster_enabled,
        spatial_enabled,
    )


def run_experiment(config_path: Path) -> Dict[str, Any]:
    config = _load_config(config_path)
    _, output_dir = _prepare_output_dir(config, config_path)
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
    spatial_start_epoch = int(
        training_config.get("spatial_start_epoch", warm_up_epochs + 1)
    )
    if spatial_start_epoch < 1:
        raise ValueError("spatial_start_epoch must be positive")
    spatial_config = _section(config, "spatial")
    spatial_mechanism_enabled = bool(spatial_config.get("enabled", True))
    spatial_weighting_enabled = spatial_mechanism_enabled and bool(
        spatial_config.get("consistency_weighting", True)
    )
    spatial_negative_filter_enabled = spatial_mechanism_enabled and bool(
        spatial_config.get("negative_filter", True)
    )

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
    cluster_head_multiplier = float(
        training_config.get("cluster_head_lr_multiplier", 8.0)
    )
    if learning_rate <= 0 or cluster_head_multiplier <= 0:
        raise ValueError("lr and cluster_head_lr_multiplier must be positive")
    cluster_head_parameters = list(model.cluster_head.parameters())
    cluster_head_parameter_ids = {id(parameter) for parameter in cluster_head_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in cluster_head_parameter_ids
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_parameters, "lr": learning_rate},
            {
                "params": cluster_head_parameters,
                "lr": learning_rate * cluster_head_multiplier,
            },
        ],
        weight_decay=weight_decay,
    )

    print(f"Dataset: {dataset} | spots={sample.n_spots} | device={device}")
    print(f"Modalities: {', '.join(modality_names)} | label={sample.label_key}")
    print(f"Spatial graph: shape={spatial_adjacency.shape}, nnz={spatial_adjacency.nnz}")
    print(
        f"Learning rates: backbone={learning_rate:g} | "
        f"cluster_head={learning_rate * cluster_head_multiplier:g} "
        f"({cluster_head_multiplier:g}x)"
    )

    history = []
    cluster_head_init = str(training_config.get("cluster_head_init", "none")).lower()
    cluster_head_initialized = False
    for epoch_index in range(epochs):
        epoch_number = epoch_index + 1
        model.train()
        coefficients = _effective_loss_coefficients(
            config,
            epoch_number,
            warm_up_epochs,
            spatial_start_epoch=spatial_start_epoch,
            spatial_mechanism_enabled=spatial_mechanism_enabled,
        )
        (
            lambda_rec,
            lambda_mgcl,
            lambda_cluster,
            lambda_spatial,
            cluster_enabled,
            spatial_enabled,
        ) = coefficients
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
            use_spatial_weighting=spatial_weighting_enabled,
            use_spatial_negative_filter=spatial_negative_filter_enabled,
            use_spatial_loss=spatial_enabled,
            lambda_rec=lambda_rec,
            lambda_mgcl=lambda_mgcl,
            lambda_cluster=lambda_cluster,
            lambda_spatial=lambda_spatial,
        )
        total_loss = output.total_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"non-finite total loss at epoch {epoch_number}")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        cluster_head_gradient_norm = _gradient_norm(model.cluster_head)
        optimizer.step()

        if (
            not cluster_head_initialized
            and cluster_head_init == "kmeans"
            and epoch_number >= warm_up_epochs
        ):
            model.cluster_head.initialize_from_kmeans(
                [
                    output.multigranularity_views[view_name]["z"].detach()
                    for view_name in model.view_order
                ],
                seed=seed,
                temperature=float(model.cluster_contrastive_loss.temperature),
            )
            cluster_head_initialized = True

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
            "spatial_loss": _json_float(output.spatial_loss),
            "lambda_rec": lambda_rec,
            "lambda_mgcl": lambda_mgcl,
            "lambda_cluster": lambda_cluster,
            "lambda_spatial": lambda_spatial,
            "cluster_enabled": cluster_enabled,
            "spatial_enabled": spatial_enabled,
            "spatial_weighting_enabled": spatial_weighting_enabled,
            "spatial_negative_filter_enabled": spatial_negative_filter_enabled,
            "sc_enabled": spatial_weighting_enabled,
            "snf_enabled": spatial_negative_filter_enabled,
            "mgcl_weight_std": _json_float(output.mgcl_weight_std),
            "neg_count": output.neg_count,
            "snf_masked_positions": output.snf_masked_positions,
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
            f"spatial_loss={record['spatial_loss']:.6f} | "
            f"lambda_spatial={record['lambda_spatial']:.3g} | "
            f"sc_enabled={record['sc_enabled']} | "
            f"snf_enabled={record['snf_enabled']} | "
            f"mgcl_weight_std={record['mgcl_weight_std']:.3e} | "
            f"neg_count={record['neg_count']} | "
            f"snf_masked_positions={record['snf_masked_positions']} | "
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
            use_spatial_weighting=spatial_weighting_enabled,
            use_spatial_negative_filter=spatial_negative_filter_enabled,
            use_spatial_loss=spatial_enabled,
            lambda_rec=lambda_rec,
            lambda_mgcl=lambda_mgcl,
            lambda_cluster=lambda_cluster,
            lambda_spatial=lambda_spatial,
        )
    clustering_options = dict(clustering_config)
    method = str(clustering_config.get("method", "kmeans"))
    default_embedding = (
        "cluster_probabilities" if method.lower() == "argmax" else "global"
    )
    embedding_mode = str(clustering_options.get("embedding", default_embedding))
    embedding = _embedding_from_output(final_output, embedding_mode).detach().cpu().numpy()
    predicted, method_used = _cluster_embedding(
        embedding, num_clusters, method=method, seed=seed
    )
    metrics = clustering_metrics(labels, predicted)
    candidate_metrics: Dict[str, Any] = {}
    for candidate_mode in (
        "global",
        "weighted_views",
        "mean_views",
        "cluster_probabilities",
        "concat_views",
    ):
        candidate_embedding = _embedding_from_output(
            final_output, candidate_mode
        ).detach().cpu().numpy()
        candidate_predicted, candidate_method = _cluster_embedding(
            candidate_embedding,
            num_clusters,
            method="kmeans",
            seed=seed,
        )
        candidate_metrics[candidate_mode] = {
            "method": candidate_method,
            **clustering_metrics(labels, candidate_predicted),
        }
    q_probabilities = _embedding_from_output(
        final_output, "cluster_probabilities"
    ).detach().cpu().numpy()
    q_predicted, q_method = _cluster_embedding(
        q_probabilities,
        num_clusters,
        method="argmax",
        seed=seed,
    )
    candidate_metrics["cluster_argmax"] = {
        "method": q_method,
        **clustering_metrics(labels, q_predicted),
    }
    z_view_metrics: Dict[str, Any] = {}
    for view_name in model.view_order:
        z_embedding = (
            final_output.multigranularity_views[view_name]["z"]
            .detach()
            .cpu()
            .numpy()
        )
        z_predicted, z_method = _cluster_embedding(
            z_embedding, num_clusters, method="kmeans", seed=seed
        )
        z_view_metrics[view_name] = {
            "method": z_method,
            "dim": int(z_embedding.shape[1]),
            **clustering_metrics(labels, z_predicted),
        }
    representation_metrics: Dict[str, Any] = {"z_views": z_view_metrics}
    for z_mode in ("mean_z", "concat_z"):
        z_embedding = _embedding_from_output(final_output, z_mode).detach().cpu().numpy()
        z_predicted, z_method = _cluster_embedding(
            z_embedding, num_clusters, method="kmeans", seed=seed
        )
        representation_metrics[z_mode] = {
            "method": z_method,
            "dim": int(z_embedding.shape[1]),
            **clustering_metrics(labels, z_predicted),
        }
    final_cluster_diagnostics = model.cluster_contrastive_loss.diagnostics(
        final_output.cluster_assignments
    )

    output_config_path = output_dir / "config.yaml"

    report: Dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "device": str(device),
        "output_dir": str(output_dir),
        "epochs": epochs,
        "warm_up_epochs": warm_up_epochs,
        "spatial_start_epoch": spatial_start_epoch,
        "spatial_mechanism_enabled": spatial_mechanism_enabled,
        "spatial_weighting_enabled": spatial_weighting_enabled,
        "spatial_negative_filter_enabled": spatial_negative_filter_enabled,
        "cluster_enabled_epochs": [
            record["epoch"] for record in history if record["cluster_enabled"]
        ],
        "spatial_enabled_epochs": [
            record["epoch"] for record in history if record["spatial_enabled"]
        ],
        "clustering_method_requested": method,
        "clustering_method_used": method_used,
        "clustering_embedding": embedding_mode,
        "candidate_metrics": candidate_metrics,
        "representation_metrics": representation_metrics,
        "fine_dim": int(model_config["fine_dim"]),
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
    print(f"Output directory: {output_dir}")
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
