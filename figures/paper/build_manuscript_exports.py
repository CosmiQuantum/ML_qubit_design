#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "figures" / "paper"
EXPORT_DIR = PAPER_DIR / "manuscript_exports"
HYPER_DIR = EXPORT_DIR / "hypertuner_search_metrics"
SIM_RESULTS_DIR = EXPORT_DIR / "simulated_results"

TRANSMON_DIR = REPO_ROOT / "experiments" / "model_predict_qubit_TransmonCross_Hamiltonian_params"
TRANSMON_ARTIFACT_DIR = REPO_ROOT / "experiments" / "model_predict_qubit-TransmonCross-Hamiltonian_params"
NCAP_DIR = REPO_ROOT / "experiments" / "model_predict_coupler_NCap_cap_matrix"
RESONATOR_DIR = REPO_ROOT / "experiments" / "model_predict_cavity_claw_RouteMeander_eigenmode"

COMPILED_PDF = REPO_ROOT / "Component_Level_Inverse_Design_of_Transmon_Qubits_Using_Neural_Networks.pdf"

FULL_WIDTH_IN = 7.10
COLUMN_WIDTH_IN = 3.35

ORANGE = "#E87A00"
ORANGE_LIGHT = "#FFF4E6"
PURPLE = "#7B68AE"
PURPLE_LIGHT = "#E8E4F0"
GREEN = "#3D8B3D"
GREEN_LIGHT = "#E8F5E8"
TEXT = "#222222"
TEXT_DIM = "#555555"
GRID = "#D7D7D7"
SPINE = "#888888"


def ensure_dirs() -> None:
    EXPORT_DIR.mkdir(exist_ok=True)
    HYPER_DIR.mkdir(exist_ok=True)
    SIM_RESULTS_DIR.mkdir(exist_ok=True)


def use_paper_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "normal",
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": SPINE,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"copied {src.relative_to(REPO_ROOT)} -> {dst.relative_to(REPO_ROOT)}")


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    plt.close(fig)


def crop_pdf_page(page_number: int, clip_rect: tuple[float, float, float, float], out_path: Path) -> None:
    if not COMPILED_PDF.exists() and out_path.exists():
        print(f"reused existing {out_path.relative_to(REPO_ROOT)}")
        return
    src_doc = fitz.open(COMPILED_PDF)
    src_page = src_doc.load_page(page_number - 1)
    clip = fitz.Rect(*clip_rect)
    out_doc = fitz.open()
    out_page = out_doc.new_page(width=clip.width, height=clip.height)
    out_page.show_pdf_page(out_page.rect, src_doc, page_number - 1, clip=clip)
    out_doc.save(out_path)
    out_doc.close()
    src_doc.close()
    print(f"cropped pdf -> {out_path.relative_to(REPO_ROOT)}")


def crop_png_page(page_number: int, clip_rect: tuple[float, float, float, float], out_path: Path, zoom: float = 2.5) -> None:
    if not COMPILED_PDF.exists() and out_path.exists():
        print(f"reused existing {out_path.relative_to(REPO_ROOT)}")
        return
    src_doc = fitz.open(COMPILED_PDF)
    src_page = src_doc.load_page(page_number - 1)
    clip = fitz.Rect(*clip_rect)
    pix = src_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    pix.save(out_path)
    src_doc.close()
    print(f"cropped png -> {out_path.relative_to(REPO_ROOT)}")


def load_transmon_trials() -> pd.DataFrame:
    trial_dir = TRANSMON_ARTIFACT_DIR / "kt_dir2" / "transmon_cross_surrogate_loss2"
    rows: list[dict[str, float | int | bool | str]] = []
    for trial_path in sorted(trial_dir.glob("trial_*/trial.json")):
        data = json.loads(trial_path.read_text())
        values = dict(data.get("hyperparameters", {}).get("values", {}))
        score = data.get("score")
        if score is None:
            continue
        rows.append(values | {"val_loss": float(score), "trial_id": trial_path.parent.name})

    df = pd.DataFrame(rows)
    neuron_cols = sorted(col for col in df.columns if col.startswith("neurons_"))
    for col in neuron_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def estimate_trainable_params(row: pd.Series, input_dim: int = 2, output_dim: int = 3) -> int:
        n_layers = int(row["n_layers"])
        widths = [int(row[f"neurons_{idx}"]) for idx in range(n_layers)]
        use_batchnorm = bool(row["use_batchnorm"])
        prev = input_dim
        total = 0
        for width in widths:
            total += (prev + 1) * width
            if use_batchnorm:
                total += 2 * width
            prev = width
        total += (prev + 1) * output_dim
        return total

    df["total_hidden_units"] = df[neuron_cols].fillna(0).sum(axis=1)
    df["max_width"] = df[neuron_cols].fillna(0).max(axis=1)
    df["total_trainable_params"] = df.apply(estimate_trainable_params, axis=1)
    return df


def draw_binned_median(ax: plt.Axes, x: np.ndarray, y: np.ndarray, *, bins: int = 8, log_scale: bool = False, color: str = TEXT_DIM) -> None:
    if log_scale:
        x_work = np.log10(x)
    else:
        x_work = x

    finite = np.isfinite(x_work) & np.isfinite(y)
    x_work = x_work[finite]
    x_raw = x[finite]
    y = y[finite]

    if len(x_work) < bins:
        return

    edges = np.quantile(x_work, np.linspace(0, 1, bins + 1))
    edges[0] -= 1e-12
    mids = []
    meds = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (x_work > left) & (x_work <= right)
        if mask.sum() < 4:
            continue
        mids.append(np.median(x_raw[mask]))
        meds.append(np.median(y[mask]))

    if mids:
        ax.plot(mids, meds, color=color, linewidth=1.7, linestyle="--", zorder=4)


def plot_dataset_distributions() -> None:
    use_paper_style()

    data_dir = TRANSMON_ARTIFACT_DIR / "data" / "npy"
    y_train = np.load(data_dir / "y_train_linear_encoding.npy")
    y_val = np.load(data_dir / "y_val_linear_encoding.npy")
    y_test = np.load(data_dir / "y_test_linear_encoding.npy")
    Y = np.vstack([y_train, y_val, y_test]) * 1e6

    labels = np.load(TRANSMON_DIR / "metadata" / "y_columns.npy", allow_pickle=True).tolist()
    short_labels = [
        "claw length (um)",
        "ground spacing (um)",
        "cross length (um)",
    ]
    if len(labels) == len(short_labels):
        labels = short_labels

    fig, axes = plt.subplots(3, 1, figsize=(COLUMN_WIDTH_IN, 5.2), sharey=False)
    bins = [24, np.arange(3.5, 10.6, 1.0), 24]

    color = GREEN
    for ax, values, label, binspec in zip(
        axes,
        Y.T,
        labels,
        bins,
    ):
        ax.hist(values, bins=binspec, color=color, alpha=0.18, edgecolor=color, linewidth=1.3)
        ax.axvline(np.median(values), color=color, linewidth=1.2, linestyle="--")
        ax.set_xlabel(label)
        ax.set_ylabel("count")
        ax.grid(axis="y", linestyle=":", color=GRID)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_title("Quantum Metal parameter distributions")
    fig.tight_layout()
    save_figure(fig, EXPORT_DIR / "dataset_distributions.pdf")


def plot_architecture_sweep() -> None:
    use_paper_style()

    df = pd.read_csv(TRANSMON_DIR / "results" / "validation" / "sweep_results.csv")
    widths_to_keep = [16, 32, 64, 128, 256, 512]
    depths_to_keep = [1, 2, 3, 4, 5]
    df = df[df["width"].isin(widths_to_keep) & df["depth"].isin(depths_to_keep)].copy()
    heatmap_df = (
        df.pivot_table(index="depth", columns="width", values="best_val_loss", aggfunc="min")
        .sort_index()
        .sort_index(axis=1)
    )

    cmap = LinearSegmentedColormap.from_list("paper_sweep", [GREEN, "#F7F7F7", ORANGE])
    best_row = df.loc[df["best_val_loss"].idxmin()]

    fig, ax0 = plt.subplots(figsize=(FULL_WIDTH_IN, 2.75))
    im = ax0.imshow(heatmap_df.values, cmap=cmap, aspect="auto", origin="lower")
    ax0.set_xticks(np.arange(len(heatmap_df.columns)))
    ax0.set_xticklabels([str(int(v)) for v in heatmap_df.columns])
    ax0.set_yticks(np.arange(len(heatmap_df.index)))
    ax0.set_yticklabels([str(int(v)) for v in heatmap_df.index])
    ax0.set_xlabel("width (neurons per layer)")
    ax0.set_ylabel("depth (number of layers)")
    ax0.set_title("Validation loss")

    for row_idx, depth in enumerate(heatmap_df.index):
        for col_idx, width in enumerate(heatmap_df.columns):
            value = heatmap_df.loc[depth, width]
            if np.isnan(value):
                continue
            ax0.text(
                col_idx,
                row_idx,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=7.6,
                color=TEXT if value > np.nanmin(heatmap_df.values) + 0.004 else "#173717",
                fontweight="bold",
            )

    best_depth = int(best_row["depth"])
    best_width = int(best_row["width"])
    best_col = list(heatmap_df.columns).index(best_width)
    best_row_idx = list(heatmap_df.index).index(best_depth)
    ax0.scatter(best_col, best_row_idx, marker="*", s=130, color="#F5F5F5", edgecolor="#173717", linewidth=1.0, zorder=5)

    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.03)
    cbar.set_label("best val loss")

    fig.tight_layout()
    save_figure(fig, EXPORT_DIR / "architecture_sweep-v2.pdf")


def plot_tuner_correlations() -> None:
    use_paper_style()

    df = load_transmon_trials()
    best_idx = df["val_loss"].idxmin()
    best = df.loc[best_idx]

    feature_specs: list[tuple[str, str, str]] = [
        ("learning_rate", "Learning rate", "log"),
        ("l2_reg", "L2 regularization", "log"),
        ("dropout_rate", "Dropout rate", "linear"),
        ("penalty_weight", "Penalty weight", "log"),
        ("n_layers", "Hidden layers", "discrete"),
        ("total_trainable_params", "Total trainable params", "log"),
        ("use_batchnorm", "Batch normalization", "bool"),
    ]

    correlations = []
    for col, label, kind in feature_specs:
        if kind == "bool":
            series = df[col].astype(int)
        else:
            series = pd.to_numeric(df[col], errors="coerce")
        rho = series.corr(df["val_loss"], method="spearman")
        if pd.isna(rho):
            continue
        correlations.append({"column": col, "label": label, "kind": kind, "rho": float(rho)})

    corr_df = pd.DataFrame(correlations).sort_values("rho", key=lambda s: s.abs(), ascending=False)
    top_features = corr_df.head(4).to_dict("records")

    fig = plt.figure(figsize=(FULL_WIDTH_IN, 4.6))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.05, 1.0, 1.0], hspace=0.38, wspace=0.35, figure=fig)

    ax_bar = fig.add_subplot(gs[:, 0])
    bar_df = corr_df.iloc[::-1]
    bar_colors = [ORANGE if rho > 0 else PURPLE for rho in bar_df["rho"]]
    ax_bar.barh(bar_df["label"], bar_df["rho"], color=bar_colors, alpha=0.85)
    ax_bar.axvline(0, color=SPINE, linewidth=0.9)
    ax_bar.set_xlabel("Spearman correlation with val loss")
    ax_bar.set_title("Which hyperparameters move with val loss")
    ax_bar.grid(axis="x", linestyle=":", color=GRID)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    panel_axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=5, markerfacecolor=ORANGE, markeredgecolor="none", alpha=0.5, label="Trial"),
        Line2D([0], [0], marker="*", linestyle="none", markersize=10, markerfacecolor="#C14F00", markeredgecolor="white", markeredgewidth=0.7, label=f"Best trial ({best['val_loss']:.4f})"),
        Line2D([0], [0], color=TEXT_DIM, linewidth=1.7, linestyle="--", label="Binned median"),
    ]

    for ax, feature in zip(panel_axes, top_features):
        col = feature["column"]
        label = feature["label"]
        kind = feature["kind"]

        if kind == "bool":
            x_vals = df[col].astype(int).to_numpy()
            groups = [df.loc[x_vals == 0, "val_loss"].to_numpy(), df.loc[x_vals == 1, "val_loss"].to_numpy()]
            ax.boxplot(
                groups,
                tick_labels=["False", "True"],
                patch_artist=True,
                boxprops=dict(facecolor=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=1.1),
                medianprops=dict(color=ORANGE, linewidth=1.2),
                whiskerprops=dict(color=ORANGE, linewidth=1.0),
                capprops=dict(color=ORANGE, linewidth=1.0),
                flierprops=dict(marker="o", markerfacecolor=ORANGE, markeredgecolor="none", alpha=0.3, markersize=3),
            )
            jitter = np.random.default_rng(0).normal(0, 0.035, size=len(df))
            ax.scatter(x_vals + 1 + jitter, df["val_loss"], s=10, color=ORANGE, alpha=0.35, linewidths=0)
            ax.scatter(int(best[col]) + 1, best["val_loss"], marker="*", s=120, color="#C14F00", edgecolors="white", linewidths=0.7, zorder=5)
        elif kind == "discrete":
            x_vals = df[col].to_numpy()
            jitter = np.random.default_rng(0).normal(0, 0.04, size=len(df))
            ax.scatter(x_vals + jitter, df["val_loss"], s=10, color=ORANGE, alpha=0.45, linewidths=0)
            medians = df.groupby(col)["val_loss"].median().sort_index()
            ax.plot(medians.index, medians.values, color=TEXT_DIM, linewidth=1.7, linestyle="--")
            ax.scatter(best[col], best["val_loss"], marker="*", s=120, color="#C14F00", edgecolors="white", linewidths=0.7, zorder=5)
            ax.set_xticks(sorted(df[col].dropna().unique()))
        else:
            x_vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
            ax.scatter(x_vals, df["val_loss"], s=9, color=ORANGE, alpha=0.42, linewidths=0)
            draw_binned_median(ax, x_vals, df["val_loss"].to_numpy(), bins=8, log_scale=(kind == "log"))
            ax.scatter(best[col], best["val_loss"], marker="*", s=120, color="#C14F00", edgecolors="white", linewidths=0.7, zorder=5)
            if kind == "log":
                ax.set_xscale("log")
            if col == "total_trainable_params":
                ax.ticklabel_format(axis="y", style="plain")

        ax.set_title(f"{label}  (rho = {feature['rho']:+.2f})")
        ax.set_ylabel("Val loss")
        ax.grid(axis="y", linestyle=":", color=GRID)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if col == "total_trainable_params":
            ax.set_xlabel("trainable params")
        else:
            ax.set_xlabel(label)

    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=True, edgecolor="#CCCCCC", bbox_to_anchor=(0.68, 1.02))
    fig.suptitle(
        f"Keras Tuner hyperparameter search  ({len(df)} trials)",
        y=1.03,
        fontsize=11,
        fontweight="normal",
    )
    fig.tight_layout()
    save_figure(fig, EXPORT_DIR / "keras_tuner_hyperparameter_search-v2.pdf")


def plot_inverse_surrogate_boxplot() -> None:
    use_paper_style()

    df = pd.read_csv(TRANSMON_DIR / "results" / "validation" / "inverse+surrogate_percentErrors.csv")
    data = [df["frequency"].to_numpy(), df["anharmonicity"].to_numpy()]
    edge_colors = [ORANGE, PURPLE]
    fill_colors = [ORANGE_LIGHT, PURPLE_LIGHT]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.6))
    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.54,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor=TEXT, markersize=5),
        medianprops=dict(color=TEXT_DIM, linewidth=1.5),
        whiskerprops=dict(color=TEXT_DIM, linewidth=1.0),
        capprops=dict(color=TEXT_DIM, linewidth=1.0),
        flierprops=dict(marker="", markersize=0),
    )
    for patch, face, edge in zip(box["boxes"], fill_colors, edge_colors):
        patch.set_facecolor(face)
        patch.set_edgecolor(edge)
        patch.set_linewidth(1.2)

    rng = np.random.default_rng(0)
    for idx, (values, color) in enumerate(zip(data, edge_colors), start=1):
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(np.full_like(values, idx) + jitter, values, s=7, color=color, alpha=0.34, linewidths=0)

    means = [np.mean(values) for values in data]
    medians = [np.median(values) for values in data]
    legend_handles = [
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor=ORANGE, markeredgewidth=1.2, markersize=5.5, label=fr"$\omega_q$ mean: {means[0]:.2f}%"),
        Line2D([0], [0], color=ORANGE, linewidth=1.8, label=fr"$\omega_q$ median: {medians[0]:.2f}%"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor=PURPLE, markeredgewidth=1.2, markersize=5.5, label=fr"$\alpha$ mean: {means[1]:.2f}%"),
        Line2D([0], [0], color=PURPLE, linewidth=1.8, label=fr"$\alpha$ median: {medians[1]:.2f}%"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=True, edgecolor="#CCCCCC", facecolor="white")
    ax.set_xticklabels([r"$\omega_q$", r"$\alpha$"])
    ax.set_ylabel("Percent error [%]")
    ax.set_title("Inverse + surrogate reconstruction error")
    ax.grid(axis="y", linestyle=":", color=GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, EXPORT_DIR / "inverse_surrogate_percent_error_boxplot.pdf")


def plot_ansys_validation_vs_nn_distance() -> None:
    use_paper_style()

    data = json.loads((TRANSMON_DIR / "results" / "validation" / "random_candidates_tested_with_ansys_nn_bins.json").read_text())
    bins = sorted({int(row["nn_bin"]) for row in data})
    eps = 1e-15

    fq_pts = []
    ah_pts = []
    nn_pts = []
    bin_labels = []

    for bin_id in bins:
        rows = [row for row in data if int(row["nn_bin"]) == bin_id]
        fq = np.array(
            [
                100 * abs(row["pred_H_params"]["qubit_frequency_GHz"] - row["surrogate_H_params"]["qubit_frequency_GHz"])
                / (abs(row["pred_H_params"]["qubit_frequency_GHz"]) + eps)
                for row in rows
            ]
        )
        ah = np.array(
            [
                100 * abs(row["pred_H_params"]["anharmonicity_MHz"] - row["surrogate_H_params"]["anharmonicity_MHz"])
                / (abs(row["pred_H_params"]["anharmonicity_MHz"]) + eps)
                for row in rows
            ]
        )
        nn = np.array([row["nn_distance_scaled"] * 100 for row in rows])

        fq_pts.append(fq)
        ah_pts.append(ah)
        nn_pts.append(nn)
        bin_labels.append(f"{np.mean([row['nn_distance_scaled'] for row in rows]):.3f}")

    def qstats(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        med = np.array([np.median(v) for v in values])
        q1 = np.array([np.percentile(v, 25) for v in values])
        q3 = np.array([np.percentile(v, 75) for v in values])
        return med, q1, q3

    fq_med, fq_q1, fq_q3 = qstats(fq_pts)
    ah_med, ah_q1, ah_q3 = qstats(ah_pts)
    nn_med, nn_q1, nn_q3 = qstats(nn_pts)

    x = np.arange(len(bins))
    width = 0.35
    jitter_amp = 0.055

    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 3.3))
    rng = np.random.default_rng(42)
    for idx in range(len(bins)):
        ax.scatter(
            x[idx] - width / 2 + rng.uniform(-jitter_amp, jitter_amp, size=len(fq_pts[idx])),
            fq_pts[idx],
            color=ORANGE,
            alpha=0.26,
            s=9,
            linewidths=0,
            zorder=2,
        )
        ax.scatter(
            x[idx] + width / 2 + rng.uniform(-jitter_amp, jitter_amp, size=len(ah_pts[idx])),
            ah_pts[idx],
            color=PURPLE,
            alpha=0.26,
            s=9,
            linewidths=0,
            zorder=2,
        )

    ax.bar(x - width / 2, fq_med, width=width, color=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=1.2, zorder=3)
    ax.bar(x + width / 2, ah_med, width=width, color=PURPLE_LIGHT, edgecolor=PURPLE, linewidth=1.2, zorder=3)

    ax.errorbar(x - width / 2, fq_med, yerr=[fq_med - fq_q1, fq_q3 - fq_med], fmt="none", ecolor=ORANGE, elinewidth=1.3, capsize=3, zorder=5)
    ax.errorbar(x + width / 2, ah_med, yerr=[ah_med - ah_q1, ah_q3 - ah_med], fmt="none", ecolor=PURPLE, elinewidth=1.3, capsize=3, zorder=5)
    ax.fill_between(x, nn_q1, nn_q3, color=GREEN, alpha=0.16, zorder=1)
    ax.plot(x, nn_med, color=GREEN, linewidth=1.5, zorder=4)
    ax.plot(x, nn_med, "D", color=GREEN, markersize=4.5, markeredgecolor="white", markeredgewidth=0.7, zorder=6)

    legend_handles = [
        Patch(facecolor=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=1.0, label=r"$f_q$ error IQR"),
        Line2D([0], [0], marker="o", color=ORANGE, linestyle="none", markersize=5, label=r"$f_q$ error median"),
        Patch(facecolor=PURPLE_LIGHT, edgecolor=PURPLE, linewidth=1.0, label=r"$\alpha$ error IQR"),
        Line2D([0], [0], marker="s", color=PURPLE, linestyle="none", markersize=5, label=r"$\alpha$ error median"),
        Patch(facecolor=GREEN_LIGHT, edgecolor=GREEN, linewidth=1.0, label="NN distance IQR"),
        Line2D([0], [0], marker="D", color=GREEN, linestyle="-", markersize=4.5, label="NN distance median"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=True, edgecolor="#CCCCCC", facecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Scaled [0,1] Euclidean distance to nearest neighbor")
    ax.set_ylabel("Percent error / NN distance (%)")
    ax.set_title("Ansys vs surrogate Hamiltonian error")
    y_max = max(
        np.nanmax(fq_q3),
        np.nanmax(ah_q3),
        np.nanmax(nn_q3),
        np.nanmax([np.nanmax(v) for v in fq_pts + ah_pts]),
    )
    ax.set_ylim(0, y_max * 1.18)
    ax.grid(axis="y", linestyle=":", color=GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, EXPORT_DIR / "ansys_validation_error_vs_nn_distance-v2.pdf")


def plot_model_architecture_combined() -> None:
    use_paper_style()

    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 2.75))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    ax.axis("off")

    def block(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        lines: list[str],
        *,
        edge: str,
        face: str,
        hatch: str | None = None,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.1",
            linewidth=1.35,
            edgecolor=edge,
            facecolor=face,
            hatch=hatch,
            zorder=3,
        )
        ax.add_patch(patch)
        ax.text(x + 1.2, y + h - 2.2, title, ha="left", va="top", fontsize=8.8, fontweight="bold", color=edge, zorder=5)
        for idx, line in enumerate(lines):
            ax.text(x + 1.2, y + h - 5.6 - idx * 2.2, line, ha="left", va="top", fontsize=7.7, color=TEXT, zorder=5)

    def arrow(x0: float, y0: float, x1: float, y1: float, label: str | None = None) -> None:
        arr = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.45,
            color=TEXT_DIM,
            shrinkA=2.5,
            shrinkB=2.5,
            zorder=4,
        )
        ax.add_patch(arr)
        if label:
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2 + 2.2,
                label,
                ha="center",
                va="center",
                fontsize=7.2,
                color=TEXT_DIM,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.5),
                zorder=6,
            )

    y = 11.0
    h = 12.2
    boxes = {
        "input": (2.0, y, 15.0, h),
        "inverse": (22.5, y, 18.0, h),
        "geom": (46.0, y, 15.5, h),
        "surrogate": (67.0, y, 18.0, h),
        "output": (90.0, y, 8.0, h),
    }

    training_region = FancyBboxPatch(
        (20.4, y - 1.2),
        78.0,
        h + 3.0,
        boxstyle="round,pad=0,rounding_size=1.3",
        linewidth=1.0,
        edgecolor=GREEN,
        facecolor=GREEN_LIGHT,
        alpha=0.26,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax.add_patch(training_region)
    ax.text(
        21.5,
        y + h + 0.9,
        "Training",
        ha="left",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        fontstyle="italic",
        color=GREEN,
        zorder=1,
    )

    block(*boxes["input"], "Targets", [r"$\omega_q,\ \alpha$", "scaled inputs"], edge=ORANGE, face=ORANGE_LIGHT)
    block(*boxes["inverse"], "Inverse MLP", ["2 hidden layers", "16 neurons each", "387 trainable"], edge=GREEN, face=GREEN_LIGHT)
    block(*boxes["geom"], "Design", ["3 Quantum Metal", "geometry params"], edge=PURPLE, face=PURPLE_LIGHT)
    block(*boxes["surrogate"], "Ansys surrogate", ["736 hidden units", "4,418 non-trainable", r"$\hat{\omega}_q,\ \hat{\alpha}$ check"], edge=GREEN, face="#F0F7F0")
    block(*boxes["output"], "Loss", ["MAE in", "target", "space"], edge=TEXT_DIM, face="#F7F7F7")

    arrow(17.0, y + h / 2, 22.5, y + h / 2)
    arrow(40.5, y + h / 2, 46.0, y + h / 2)
    arrow(61.5, y + h / 2, 67.0, y + h / 2)
    arrow(85.0, y + h / 2, 90.0, y + h / 2)

    update_arrow = FancyArrowPatch(
        (94.0, y - 0.1),
        (31.5, y - 0.1),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.25,
        color=GREEN,
        linestyle=(0, (5, 3)),
        connectionstyle="arc3,rad=-0.18",
        shrinkA=3.0,
        shrinkB=3.0,
        zorder=2,
    )
    ax.add_patch(update_arrow)
    ax.text(
        62.0,
        y - 2.8,
        "update inverse weights",
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        fontstyle="italic",
        color=GREEN,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.5),
        zorder=6,
    )

    ax.text(
        50,
        31.7,
        "Surrogate-defined inverse training",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="normal",
        color=TEXT,
    )
    ax.text(
        50,
        29.2,
        "Trainable params: 387   |   Non-trainable params: 4,418   |   Total params: 4,805",
        ha="center",
        va="center",
        fontsize=8.1,
        color=TEXT_DIM,
    )
    ax.text(
        50,
        2.7,
        "The surrogate is fixed during inverse-model training, so geometry predictions are graded by recovered Hamiltonian parameters.",
        ha="center",
        va="center",
        fontsize=7.7,
        color=TEXT_DIM,
    )

    fig.tight_layout(pad=0.15)
    out_paths = [
        EXPORT_DIR / "model_architecture_paper_theme_combined.pdf",
        TRANSMON_DIR / "plots" / "model_architecture_paper_theme_combined.pdf",
        TRANSMON_DIR / "plots" / "model_architecture_paper_theme_combined.png",
    ]
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    plt.close(fig)


def export_static_sources() -> None:
    copy_file(PAPER_DIR / "outputs" / "pipeline_updated.pdf", EXPORT_DIR / "design_example.pdf")
    copy_file(PAPER_DIR / "outputs" / "inverse_pipeline.pdf", EXPORT_DIR / "inverse_pipeline.pdf")
    copy_file(PAPER_DIR / "outputs" / "workflow.pdf", EXPORT_DIR / "workflow.pdf")
    copy_file(PAPER_DIR / "outputs" / "testing_pipeline.pdf", EXPORT_DIR / "testing_pipeline.pdf")
    copy_file(PAPER_DIR / "outputs" / "stress_test_methodology.pdf", EXPORT_DIR / "stress_test_methodology.pdf")
    copy_file(
        TRANSMON_DIR / "plots" / "surrogate_stress_random_points_pairs.pdf",
        EXPORT_DIR / "surrogate_stress_random_points_pairs.pdf",
    )


def export_pdf_fallbacks() -> None:
    crop_pdf_page(18, (105, 70, 495, 334), EXPORT_DIR / "data_amount_sweep-v2.pdf")
    crop_pdf_page(19, (98, 48, 502, 346), EXPORT_DIR / "predicted_vs_reference_design_comparsion.pdf")


def export_png_fallbacks() -> None:
    crop_png_page(19, (42, 392, 272, 520), EXPORT_DIR / "sample_data_distribution.png")

    crop_png_page(21, (120, 48, 510, 166), EXPORT_DIR / "testing_pipeline.png")
    crop_png_page(21, (22, 232, 280, 350), SIM_RESULTS_DIR / "transmon2.png")
    crop_png_page(21, (18, 462, 280, 552), SIM_RESULTS_DIR / "transmon3.png")
    crop_png_page(21, (312, 236, 562, 390), SIM_RESULTS_DIR / "predicted_vs_ref_ccapacitance.png")

    crop_png_page(24, (86, 63, 248, 246), EXPORT_DIR / "param_sweep_ncap.png")
    crop_png_page(24, (319, 62, 560, 247), HYPER_DIR / "coupling_cap.png")
    crop_png_page(24, (88, 296, 250, 482), EXPORT_DIR / "param_sweep_res.png")
    crop_png_page(24, (319, 294, 560, 484), HYPER_DIR / "3D_Val_Loss_vs_Learning_Rate_and_Total_Params_with_Lowest.png")


def main() -> None:
    ensure_dirs()
    export_static_sources()
    plot_dataset_distributions()
    plot_architecture_sweep()
    plot_tuner_correlations()
    plot_inverse_surrogate_boxplot()
    plot_ansys_validation_vs_nn_distance()
    plot_model_architecture_combined()
    export_pdf_fallbacks()
    export_png_fallbacks()


if __name__ == "__main__":
    main()
