#!/usr/bin/env python3
"""Redraw saved SpaMGCL results; never import project code or run training.

With no arguments, reads figure_source.json alongside this script. To refresh
the data, pass --notebook /path/to/executed.ipynb and optionally --epoch-dir.
Only stored HTML outputs are parsed; notebook cells are never executed.
"""

import argparse
import csv
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parent
DATASETS = ["HLN-A1", "E18.5", "HLN-D1", "S2-E15", "S2-E18"]
PREFIXES = {"HLN-A1": "hlna1", "E18.5": "e185", "HLN-D1": "d1",
            "S2-E15": "s2e15", "S2-E18": "s2e18"}
EPOCH_FILES = {
    "e185_core_200": "", "e185_scsnf_200": "(1)",
    "hlna1_core_200": "(2)", "hlna1_scsnf_200": "(3)",
    "d1_core_200": "(4)", "s2e15_core_200": "(5)",
    "s2e18_core_200": "(6)",
}
COLORS = {"core": "#246783", "no_sample": "#D89149", "no_cluster": "#AAAEB7"}
BLUE, ORANGE = "#246783", "#C17C2F"


class HTMLTable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if tag in ("th", "td") and self.row is not None:
            self.cell = []

    def handle_data(self, value):
        if self.cell is not None:
            self.cell.append(value)

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        if tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def numeric(value):
    if value in ("NaN", "nan", "None", ""):
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return value


def table_from_notebook(notebook, required):
    matches = []
    for index, cell in enumerate(notebook["cells"]):
        for output in cell.get("outputs", []):
            html = output.get("data", {}).get("text/html")
            if not html:
                continue
            parser = HTMLTable()
            parser.feed("".join(html) if isinstance(html, list) else html)
            if not parser.rows or not required.issubset(parser.rows[0]):
                continue
            header = parser.rows[0]
            if any(len(row) != len(header) for row in parser.rows[1:]):
                raise ValueError(f"Irregular table in notebook cell {index}")
            rows = [{key: numeric(value) for key, value in zip(header, row) if key}
                    for row in parser.rows[1:]]
            matches.append((index, rows))
    if not matches:
        raise ValueError(f"Missing saved table with columns {sorted(required)}")
    # Later summary displays repeat smaller subsets; take the largest table.
    return max(matches, key=lambda item: len(item[1]))


def extract_source(notebook_path, epoch_dir):
    raw = notebook_path.read_bytes()
    notebook = json.loads(raw)
    final_cell, final = table_from_notebook(
        notebook, {"epochs", "Q_argmax_ARI", "concat_z_ARI", "run", "dataset"})
    config_cell, configs = table_from_notebook(
        notebook, {"epochs(cfg)", "lambda_mgcl", "lambda_cluster", "run"})
    epoch_cell, epochs = table_from_notebook(
        notebook, {"epoch", "Q_argmax_ARI", "Q_argmax_NMI", "run"})
    manifest = {
        "notebook": notebook_path.name,
        "notebook_sha256": hashlib.sha256(raw).hexdigest(),
        "source_cells_zero_based": {"final": final_cell, "config": config_cell,
                                    "epochs": epoch_cell},
        "final_score_precision": "Notebook rendered table, rounded to 4 decimals",
        "epoch_score_precision": "Notebook rendered table, rounded to 6 decimals",
        "NMI_policy": "Use stored metrics NMI only, never notebook nmi_numpy recomputation",
        "seed": 0,
        "seed_basis": "User-reported protocol; seed is not a column in the saved config table",
        "readouts": {"Q_argmax": "End-to-end Q argmax", "concat_z": "Concatenated representation + KMeans"},
        "limits": ["No independent repetitions or uncertainty estimates",
                   "Saved config table is not a full config or code-version audit",
                   "50-epoch ablation and 200-epoch training cohorts are kept separate"],
        "exact_epoch_files": [],
    }
    if epoch_dir:
        for row in epochs:
            suffix = EPOCH_FILES.get(row["run"])
            if suffix is None:
                continue
            path = epoch_dir / f"metrics_epoch{row['epoch']}{suffix}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            metric = json.loads(path.read_text())
            if metric["epoch"] != row["epoch"]:
                raise ValueError(f"Epoch mismatch: {path}")
            for key in ("ARI", "NMI"):
                if abs(metric[key] - row[f"Q_argmax_{key}"]) > 0.00000051:
                    raise ValueError(f"Notebook / uploaded metric mismatch: {path} {key}")
                row[f"Q_argmax_{key}"] = metric[key]
            manifest["exact_epoch_files"].append({"file": path.name, "run": row["run"],
                                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest["epoch_score_precision"] = "Full precision from uploaded epoch JSONs, cross-checked against notebook"
    return {"manifest": manifest, "final": final, "configs": configs, "epochs": epochs}


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate(source):
    final = {row["run"]: row for row in source["final"]}
    configs = {row["run"]: row for row in source["configs"]}
    shared_fields = ["dataset", "epochs(cfg)", "lambda_spatial", "n_clusters", "fine_dim",
                     "cluster_temperature", "cluster_head_lr_mult", "warm_up_epochs",
                     "clustering_method", "clustering_embedding"]
    for dataset in DATASETS[:2]:
        prefix = "abl_" + PREFIXES[dataset] + "_"
        core = configs[prefix + "core"]
        for variant in ("core", "no_sample", "no_cluster", "sc", "snf", "sc_snf"):
            run = prefix + variant
            if final[run]["batch"] != "ablation" or final[run]["epochs"] != 50:
                raise ValueError(f"Wrong ablation cohort: {run}")
            cfg = configs[run]
            for field in shared_fields:
                if cfg[field] != core[field]:
                    raise ValueError(f"Unmatched ablation field {run}: {field}")
            for field, ablated_variant in (("lambda_mgcl", "no_sample"), ("lambda_cluster", "no_cluster")):
                expected = 0 if variant == ablated_variant else core[field]
                if cfg[field] != expected:
                    raise ValueError(f"Wrong loss switch: {run} {field}")
            expected_sc = variant in ("sc", "sc_snf")
            expected_snf = variant in ("snf", "sc_snf")
            if cfg["consistency_weighting"] != expected_sc or cfg["negative_filter"] != expected_snf:
                raise ValueError(f"Wrong spatial switch: {run}")
        if core["lambda_spatial"] != 0:
            raise ValueError("This presentation expects spatial regularization off")
    for dataset in DATASETS:
        run = PREFIXES[dataset] + "_core_200"
        if final[run]["epochs"] != 200:
            raise ValueError(f"Wrong final epoch: {run}")
    return final


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#B7BFC5", "axes.linewidth": 0.65,
        "grid.color": "#E5E9ED", "grid.linewidth": 0.65,
        "text.color": "#202D36", "axes.labelcolor": "#202D36",
        "xtick.color": "#46545E", "ytick.color": "#46545E",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def save(fig, out, name):
    for extension in ("pdf", "png"):
        path = out / f"{name}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12)
        if path.stat().st_size == 0:
            raise OSError(f"Empty figure export: {path}")
    plt.close(fig)


def loss_ablation(final, metric, out, name):
    fig, axes = plt.subplots(2, 2, figsize=(7.7, 6.2), sharey=True)
    variants = ["core", "no_sample", "no_cluster"]
    labels = ["Core", "w/o sample\ncontrast", "w/o cluster\ncontrast"]
    for row, (readout, title) in enumerate((("Q_argmax", "Q output"), ("concat_z", "Embedding + k-means"))):
        for col, dataset in enumerate(DATASETS[:2]):
            ax = axes[row, col]
            values = [final[f"abl_{PREFIXES[dataset]}_{variant}"][f"{readout}_{metric}"] for variant in variants]
            bars = ax.bar(range(3), values, color=[COLORS[v] for v in variants], width=0.58, zorder=3)
            ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=5, fontsize=9)
            ax.set_xticks(range(3), labels)
            ax.tick_params(axis="x", length=0, pad=7)
            ax.set_ylim(0, 0.61)
            ax.set_yticks([0, .2, .4, .6])
            ax.grid(axis="y", zorder=0)
            ax.set_axisbelow(True)
            ax.set_title(f"{chr(97 + row * 2 + col)}  {dataset} | {title}", loc="left", pad=12)
            if col == 0:
                ax.set_ylabel(metric)
    fig.suptitle("Contrastive-loss ablation at 50 epochs", x=.5, y=.985, fontsize=14, weight="bold")
    fig.subplots_adjust(top=.88, bottom=.09, left=.08, right=.985, hspace=.61, wspace=.24)
    save(fig, out, name)


def readout_200(final, out):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.85), sharey=True)
    y = np.arange(len(DATASETS))
    for ax, metric in zip(axes, ("ARI", "NMI")):
        q = np.array([final[PREFIXES[ds] + "_core_200"][f"Q_argmax_{metric}"] for ds in DATASETS])
        z = np.array([final[PREFIXES[ds] + "_core_200"][f"concat_z_{metric}"] for ds in DATASETS])
        ax.hlines(y, q, z, color="#B9C2C8", lw=2, zorder=2)
        ax.scatter(q, y, color=BLUE, s=43, marker="o", label="Q output", zorder=3)
        ax.scatter(z, y, color=ORANGE, s=45, marker="D", label="Embedding + k-means", zorder=3)
        ax.set_yticks(y, DATASETS)
        ax.set_ylim(4.55, -.55)
        ax.set_xlim(0, .62)
        ax.set_xticks([0, .2, .4, .6])
        ax.set_xlabel(metric)
        ax.grid(axis="x", zorder=0)
        ax.tick_params(axis="y", length=0)
    axes[0].set_title("a  ARI", loc="left", pad=10)
    axes[1].set_title("b  NMI", loc="left", pad=10)
    fig.suptitle("Readout comparison at 200 epochs", y=.98, fontsize=14, weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.54, -.005), fontsize=10)
    fig.subplots_adjust(top=.77, bottom=.2, left=.13, right=.98, wspace=.22)
    save(fig, out, "fig_S2_readouts_200")


def training_curve(source, metric, out, name):
    fig, axes = plt.subplots(2, 3, figsize=(8.7, 5.6), sharey=True)
    for index, dataset in enumerate(DATASETS):
        ax = axes.flat[index]
        for variant, color, marker, linestyle in (("core", BLUE, "o", "-"), ("scsnf", ORANGE, "s", "--")):
            run = PREFIXES[dataset] + "_" + variant + "_200"
            rows = sorted((r for r in source["epochs"] if r["run"] == run), key=lambda r: r["epoch"])
            if not rows:
                continue
            if [r["epoch"] for r in rows] != [50, 100, 200]:
                raise ValueError(f"Incomplete training trajectory: {run}")
            ax.plot([r["epoch"] for r in rows], [r[f"Q_argmax_{metric}"] for r in rows],
                    marker=marker, color=color, linestyle=linestyle, lw=1.7, ms=4.5, alpha=.94)
        ax.set_title(f"{chr(97 + index)}  {dataset}", loc="left", pad=9)
        ax.set_xticks([50, 100, 200])
        ax.set_xlim(35, 215)
        ax.set_ylim(0, .61)
        ax.set_yticks([0, .2, .4, .6])
        ax.grid(axis="y")
        ax.set_xlabel("Epoch")
        if index % 3 == 0:
            ax.set_ylabel(metric)
    ax = axes.flat[-1]
    ax.axis("off")
    handles = [Line2D([0], [0], color=BLUE, marker="o", lw=1.7, label="Core"),
               Line2D([0], [0], color=ORANGE, marker="s", ls="--", lw=1.7, label="SC + SNF")]
    ax.legend(handles=handles, loc="center left", frameon=False, bbox_to_anchor=(.10, .52))
    fig.suptitle(f"Training duration | Q output {metric}", y=.987, fontsize=14, weight="bold")
    fig.subplots_adjust(top=.86, bottom=.10, left=.07, right=.98, hspace=.61, wspace=.28)
    save(fig, out, name)


def export_tables(source, final, out):
    rows = []
    for dataset in DATASETS[:2]:
        for variant in ("core", "no_sample", "no_cluster", "sc", "snf", "sc_snf"):
            run = f"abl_{PREFIXES[dataset]}_{variant}"
            rows.append({"dataset": dataset, "variant": variant, "epochs": 50, "run": run,
                         **{f"{readout}_{metric}": final[run][f"{readout}_{metric}"]
                            for readout in ("Q_argmax", "concat_z") for metric in ("ARI", "NMI")}})
    write_csv(out / "table_ablation_50.csv", rows)
    spatial = []
    for dataset in DATASETS[:2]:
        core = final[f"abl_{PREFIXES[dataset]}_core"]
        for variant in ("core", "sc", "snf", "sc_snf"):
            row = final[f"abl_{PREFIXES[dataset]}_{variant}"]
            spatial.append({"dataset": dataset, "variant": variant, "epochs": 50, "lambda_spatial": 0,
                            "Q_ARI": row["Q_argmax_ARI"], "Q_NMI": row["Q_argmax_NMI"],
                            "delta_ARI_vs_core": round(row["Q_argmax_ARI"] - core["Q_argmax_ARI"], 4),
                            "delta_NMI_vs_core": round(row["Q_argmax_NMI"] - core["Q_argmax_NMI"], 4)})
    write_csv(out / "table_spatial_50.csv", spatial)
    main = []
    for dataset in DATASETS:
        run = PREFIXES[dataset] + "_core_200"
        main.append({"dataset": dataset, "variant": "core", "epochs": 200, "run": run,
                     **{f"{readout}_{metric}": final[run][f"{readout}_{metric}"]
                        for readout in ("Q_argmax", "concat_z") for metric in ("ARI", "NMI")}})
    write_csv(out / "table_core_200.csv", main)
    write_csv(out / "table_all_saved_results.csv", source["final"])
    write_csv(out / "table_saved_config_fields.csv", source["configs"])
    write_csv(out / "table_training_50_100_200.csv", source["epochs"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--epoch-dir", type=Path, help="Optional directory of uploaded metrics_epoch*.json")
    parser.add_argument("--source-json", type=Path, default=ROOT / "figure_source.json")
    parser.add_argument("--out", type=Path, default=ROOT / "redrawn")
    parser.add_argument("--figures-only", action="store_true", help="Redraw figures without rewriting saved source or tables")
    args = parser.parse_args()
    source = extract_source(args.notebook, args.epoch_dir) if args.notebook else json.loads(args.source_json.read_text())
    final = validate(source)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.figures_only:
        (args.out / "figure_source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        export_tables(source, final, args.out)
    style()
    loss_ablation(final, "ARI", args.out, "fig_1_loss_ablation_ARI")
    loss_ablation(final, "NMI", args.out, "fig_S1_loss_ablation_NMI")
    readout_200(final, args.out)
    training_curve(source, "ARI", args.out, "fig_S3_training_ARI")
    training_curve(source, "NMI", args.out, "fig_S4_training_NMI")
    saved = "five PDF/PNG figures" if args.figures_only else "five PDF/PNG figures and six CSV tables"
    print(f"Saved {saved} to {args.out.resolve()}")
    print("No notebook cells executed; no training or project model/config edits performed.")


if __name__ == "__main__":
    main()
