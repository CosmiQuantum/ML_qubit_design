#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import os
import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.ticker import LogFormatterMathtext, LogLocator


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

# Flowchart color scheme. Use "blue" for the current requested palette, or
# "legacy" for the previous orange/green/purple palette.
FLOWCHART_COLOR_SCHEME = "blue"

FLOWCHART_PALETTES = {
    "blue": {
        "physics_fill": "#D6E5EE",
        "physics_edge": "#567A90",
        "ml_fill": "#B0CCDE",
        "ml_edge": "#3F6F8B",
        "validation_fill": "#8AABC8",
        "validation_edge": "#17384F",
    },
    "legacy": {
        "physics_fill": "#FFF4E6",
        "physics_edge": "#E87A00",
        "ml_fill": "#E8F5E8",
        "ml_edge": "#3D8B3D",
        "validation_fill": "#E8E4F0",
        "validation_edge": "#7B68AE",
    },
}

ORANGE = "#E87A00"
ORANGE_LIGHT = "#FFF4E6"
PURPLE = "#7B68AE"
PURPLE_LIGHT = "#E8E4F0"
GREEN = "#3D8B3D"
GREEN_LIGHT = "#E8F5E8"

# Data-split categorical palette (Train/Validation/Test). Colorblind-safe
# (Okabe-Ito blue + bluish-green + a neutral dark grey) and deliberately
# distinct from the orange/purple reserved for f_q and alpha, from the
# capacitance green, and from the process-flow blues used in the flowcharts.
SPLIT_COLORS = {"Train": "#0072B2", "Validation": "#009E73", "Test": "#444444"}
SPLIT_FILLS = {"Train": "#D6E6F2", "Validation": "#D5EFE7", "Test": "#DBDBDB"}

TEXT = "#222222"
TEXT_DIM = "#555555"
GRID = "#D7D7D7"
SPINE = "#888888"


def flowchart_palette() -> dict[str, str]:
    key = FLOWCHART_COLOR_SCHEME.strip().lower()
    aliases = {
        "current": "blue",
        "new": "blue",
        "old": "legacy",
        "classic": "legacy",
    }
    key = aliases.get(key, key)
    if key not in FLOWCHART_PALETTES:
        options = ", ".join(sorted(FLOWCHART_PALETTES))
        raise ValueError(f"Unknown FLOWCHART_COLOR_SCHEME={FLOWCHART_COLOR_SCHEME!r}. Use one of: {options}.")
    return FLOWCHART_PALETTES[key]


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
            "axes.spines.top": True,
            "axes.spines.right": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def close_plot_box(ax: plt.Axes, linewidth: float = 0.8) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(SPINE)
        spine.set_linewidth(linewidth)


def regenerate_generated_sources() -> None:
    scripts = [
        PAPER_DIR / "generate_transmon_resonator_system_figure.py",
        PAPER_DIR / "generate_inverse_training_pipeline_figure.py",
        PAPER_DIR / "generate_inverse_design_workflow_figure.py",
        PAPER_DIR / "generate_compact_overview_figure.py",
        PAPER_DIR / "generate_forward_testing_pipeline_figure.py",
        PAPER_DIR / "generate_gaussian_stress_test_methodology_figure.py",
    ]
    old_env = os.environ.get("FLOWCHART_COLOR_SCHEME")
    old_path = list(sys.path)
    os.environ["FLOWCHART_COLOR_SCHEME"] = FLOWCHART_COLOR_SCHEME
    sys.path.insert(0, str(PAPER_DIR))
    try:
        for script in scripts:
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.path[:] = old_path
        if old_env is None:
            os.environ.pop("FLOWCHART_COLOR_SCHEME", None)
        else:
            os.environ["FLOWCHART_COLOR_SCHEME"] = old_env


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    plt.close(fig)


def get_pymupdf():
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "PDF cropping requires PyMuPDF, imported as 'fitz'. "
            "Install it with `python -m pip install PyMuPDF`. "
            "If you installed the unrelated package named `fitz`, remove it with "
            "`python -m pip uninstall fitz`."
        ) from exc

    if not hasattr(fitz, "open") or not hasattr(fitz, "Rect"):
        raise RuntimeError(
            "The imported 'fitz' module is not PyMuPDF. "
            "Run `python -m pip uninstall fitz` and then `python -m pip install PyMuPDF`."
        )
    return fitz


def crop_pdf_page(page_number: int, clip_rect: tuple[float, float, float, float], out_path: Path) -> None:
    if not COMPILED_PDF.exists() and out_path.exists():
        print(f"reused existing {out_path.relative_to(REPO_ROOT)}")
        return
    fitz = get_pymupdf()
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
    fitz = get_pymupdf()
    src_doc = fitz.open(COMPILED_PDF)
    src_page = src_doc.load_page(page_number - 1)
    clip = fitz.Rect(*clip_rect)
    pix = src_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    pix.save(out_path)
    src_doc.close()
    print(f"cropped png -> {out_path.relative_to(REPO_ROOT)}")


def load_transmon_trials() -> pd.DataFrame:
    trial_dir_candidates = [
        TRANSMON_DIR / "kt_dir2 3",
        TRANSMON_DIR / "kt_dir2",
    ]
    trial_files: list[Path] = []
    for trial_dir in trial_dir_candidates:
        if not trial_dir.exists():
            continue
        trial_files = sorted(set(trial_dir.glob("trial_*/trial.json")) | set(trial_dir.glob("**/trial_*/trial.json")))
        if trial_files:
            break
    if not trial_files:
        raise FileNotFoundError("No TransmonCross Keras Tuner trial.json files found")

    rows: list[dict[str, float | int | bool | str]] = []
    for trial_path in trial_files:
        data = json.loads(trial_path.read_text())
        values = dict(data.get("hyperparameters", {}).get("values", {}))
        score = data.get("score")
        if score is None:
            continue
        rows.append(values | {"val_loss": float(score), "trial_id": trial_path.parent.name})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No completed TransmonCross Keras Tuner trials found")

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
    df["total_trainable_params"] = [estimate_trainable_params(row) for _, row in df.iterrows()]
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


def parse_um(value: object) -> float:
    text = str(value).strip()
    for suffix in ("um", "\u00b5m", "\u03bcm"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return float(text)


def split_indices_like_ml00(n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_rng = np.random.RandomState(42)
    first_perm = first_rng.permutation(n_rows)
    n_val_test = int(np.ceil(0.3 * n_rows))
    val_test_idx = first_perm[:n_val_test]
    train_idx = first_perm[n_val_test:]

    second_rng = np.random.RandomState(42)
    second_perm = second_rng.permutation(len(val_test_idx))
    n_test = int(np.ceil(0.5 * len(val_test_idx)))
    test_idx = val_test_idx[second_perm[:n_test]]
    val_idx = val_test_idx[second_perm[n_test:]]
    return train_idx, val_idx, test_idx


def load_transmon_design_splits_um() -> dict[str, np.ndarray]:
    npy_dirs = [
        TRANSMON_ARTIFACT_DIR / "data" / "npy",
        TRANSMON_DIR / "data" / "npy",
    ]
    for data_dir in npy_dirs:
        train_path = data_dir / "y_train_linear_encoding.npy"
        val_path = data_dir / "y_val_linear_encoding.npy"
        test_path = data_dir / "y_test_linear_encoding.npy"
        if train_path.exists() and val_path.exists() and test_path.exists():
            splits = {
                "Train": np.load(train_path),
                "Validation": np.load(val_path),
                "Test": np.load(test_path),
            }
            for key, values in splits.items():
                values = np.asarray(values, dtype=float)
                if np.nanmax(np.abs(values)) < 1.0:
                    values = values * 1e6
                splits[key] = values
            return splits

    rows = json.loads((TRANSMON_DIR / "metadata" / "qubit-TransmonCross-Hamiltonian_params.json").read_text())
    values = []
    for row in rows:
        opts = row["design"]["design_options"]
        readout = opts["connection_pads"]["readout"]
        values.append(
            [
                parse_um(readout["claw_length"]),
                parse_um(readout["ground_spacing"]),
                parse_um(opts["cross_length"]),
            ]
        )
    all_values = np.asarray(values, dtype=float)
    train_idx, val_idx, test_idx = split_indices_like_ml00(len(all_values))
    return {
        "Train": all_values[train_idx],
        "Validation": all_values[val_idx],
        "Test": all_values[test_idx],
    }


def load_transmon_hamiltonian_splits() -> dict[str, np.ndarray]:
    npy_dirs = [
        TRANSMON_ARTIFACT_DIR / "data" / "npy",
        TRANSMON_DIR / "data" / "npy",
    ]
    for data_dir in npy_dirs:
        train_path = data_dir / "x_train_linear_encoding.npy"
        val_path = data_dir / "x_val_linear_encoding.npy"
        test_path = data_dir / "x_test_linear_encoding.npy"
        if train_path.exists() and val_path.exists() and test_path.exists():
            return {
                "Train": np.asarray(np.load(train_path), dtype=float),
                "Validation": np.asarray(np.load(val_path), dtype=float),
                "Test": np.asarray(np.load(test_path), dtype=float),
            }

    rows = json.loads((TRANSMON_DIR / "metadata" / "qubit-TransmonCross-Hamiltonian_params.json").read_text())
    values = []
    for row in rows:
        h_params = row["Hamiltonian_params"]
        values.append(
            [
                float(h_params["qubit_frequency_GHz"]),
                float(h_params["anharmonicity_MHz"]),
            ]
        )
    all_values = np.asarray(values, dtype=float)
    train_idx, val_idx, test_idx = split_indices_like_ml00(len(all_values))
    return {
        "Train": all_values[train_idx],
        "Validation": all_values[val_idx],
        "Test": all_values[test_idx],
    }


def nearest_training_hamiltonian_for_candidates(candidate_values_um: np.ndarray) -> np.ndarray:
    train_design_um = load_transmon_design_splits_um()["Train"]
    train_hamiltonian = load_transmon_hamiltonian_splits()["Train"]

    scale_min = train_design_um.min(axis=0)
    scale_range = np.maximum(train_design_um.max(axis=0) - scale_min, 1e-15)
    train_scaled = (train_design_um - scale_min) / scale_range
    candidate_scaled = (candidate_values_um - scale_min) / scale_range

    distances = np.linalg.norm(candidate_scaled[:, None, :] - train_scaled[None, :, :], axis=2)
    nearest_idx = np.argmin(distances, axis=1)
    return train_hamiltonian[nearest_idx]


def draw_split_histograms(fig: plt.Figure, axes: np.ndarray, splits: dict[str, np.ndarray], *, legend_y: float = 0.98) -> None:
    labels = [
        r"Claw length ($\mu$m)",
        r"Ground spacing ($\mu$m)",
        r"Cross length ($\mu$m)",
    ]
    colors = SPLIT_COLORS
    fills = SPLIT_FILLS

    all_values = np.vstack(list(splits.values()))
    bin_specs = [
        np.linspace(all_values[:, 0].min() - 5, all_values[:, 0].max() + 5, 18),
        np.arange(3.5, 10.6, 1.0),
        np.linspace(all_values[:, 2].min() - 5, all_values[:, 2].max() + 5, 18),
    ]

    for param_idx, (ax, label, bins) in enumerate(zip(axes, labels, bin_specs)):
        for split_name in ("Train", "Validation", "Test"):
            values = splits[split_name][:, param_idx]
            ax.hist(
                values,
                bins=bins,
                histtype="stepfilled",
                facecolor=fills[split_name],
                edgecolor=colors[split_name],
                alpha=0.36,
                linewidth=1.15,
                label=f"{split_name} (n={len(values)})",
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Counts")
        ax.grid(axis="y", linestyle=":", color=GRID)
        close_plot_box(ax)

    legend_handles = [
        Patch(facecolor=SPLIT_FILLS["Train"], edgecolor=SPLIT_COLORS["Train"], linewidth=1.0, alpha=0.8, label="Train"),
        Patch(facecolor=SPLIT_FILLS["Validation"], edgecolor=SPLIT_COLORS["Validation"], linewidth=1.0, alpha=0.8, label="Validation"),
        Patch(facecolor=SPLIT_FILLS["Test"], edgecolor=SPLIT_COLORS["Test"], linewidth=1.0, alpha=0.8, label="Test"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=True,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, legend_y),
        handlelength=1.4,
        columnspacing=0.9,
        borderpad=0.35,
    )


def plot_dataset_distributions() -> None:
    use_paper_style()

    splits = load_transmon_design_splits_um()
    fig, axes = plt.subplots(3, 1, figsize=(COLUMN_WIDTH_IN, 4.35), sharey=False)
    draw_split_histograms(fig, axes, splits)
    fig.suptitle("Quantum Metal parameter distributions", y=0.995, fontsize=10.5, fontweight="normal")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, EXPORT_DIR / "dataset_distributions.pdf")


def plot_sample_data_distribution() -> None:
    use_paper_style()

    splits = load_transmon_design_splits_um()
    fig, axes = plt.subplots(3, 1, figsize=(COLUMN_WIDTH_IN, 4.2), sharey=False)
    draw_split_histograms(fig, axes, splits, legend_y=0.94)
    fig.suptitle("Train, validation, and test distributions", y=0.995, fontsize=10.5, fontweight="normal")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out_png = EXPORT_DIR / "sample_data_distribution.png"
    out_pdf = EXPORT_DIR / "sample_data_distribution.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.04, dpi=300)
    print(f"wrote {out_pdf.relative_to(REPO_ROOT)}")
    print(f"wrote {out_png.relative_to(REPO_ROOT)}")
    plt.close(fig)


def plot_data_amount_sweep() -> None:
    use_paper_style()

    far_to_near_path = TRANSMON_DIR / "results/data_amount_sweep_corner/data_amount_sweep_far_to_near.csv"
    if far_to_near_path.exists():
        df = pd.read_csv(far_to_near_path)
        summary = (
            df.groupby(["training_percent", "n_samples"], as_index=False)
            .agg(
                train_mean=("train_mean_hamiltonian_pct", "mean"),
                train_std=("train_mean_hamiltonian_pct", "std"),
                val_mean=("val_mean_hamiltonian_pct", "mean"),
                val_std=("val_mean_hamiltonian_pct", "std"),
                test_mean=("test_mean_hamiltonian_pct", "mean"),
                test_std=("test_mean_hamiltonian_pct", "std"),
            )
            .sort_values("training_percent")
        )
        x = summary["training_percent"].to_numpy()
        x_tick_labels = [f"{value:.0f}" for value in x]
        x_label = "Training set used [%]"
        y_label = "Mean Hamiltonian percent error [%]"
        title = "Training fraction sweep"
    else:
        df = pd.read_csv(TRANSMON_DIR / "results/data_amount_sweep_uniform/data_amount_sweep.csv")
        summary = (
            df.groupby(["fraction", "n_samples"], as_index=False)
            .agg(
                train_mean=("train_mae", "mean"),
                train_std=("train_mae", "std"),
                val_mean=("val_mae", "mean"),
                val_std=("val_mae", "std"),
                test_mean=("test_mae", "mean"),
                test_std=("test_mae", "std"),
            )
            .sort_values("n_samples")
        )
        x = (summary["fraction"] * 100).to_numpy()
        x_tick_labels = [f"{frac * 100:.0f}" for frac in summary["fraction"]]
        x_label = "Training set used [%]"
        y_label = "MAE loss"
        title = "Learning curve for the inverse model"

    series = [
        ("Train", "train_mean", "train_std", SPLIT_COLORS["Train"], SPLIT_FILLS["Train"]),
        ("Val.", "val_mean", "val_std", SPLIT_COLORS["Validation"], SPLIT_FILLS["Validation"]),
        ("Test", "test_mean", "test_std", SPLIT_COLORS["Test"], SPLIT_FILLS["Test"]),
    ]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.45))
    for label, mean_col, std_col, color, fill in series:
        mean = summary[mean_col].to_numpy()
        std = summary[std_col].fillna(0).to_numpy()
        ax.plot(x, mean, marker="o", markersize=3.6, linewidth=1.35, color=color, label=label)
        ax.fill_between(x, mean - std, mean + std, color=fill, alpha=0.62, linewidth=0)

    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", color=GRID)
    close_plot_box(ax)
    ax.legend(
        loc="upper center",
        ncol=3,
        frameon=True,
        edgecolor="#CCCCCC",
        facecolor="white",
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.2,
        columnspacing=0.75,
        borderpad=0.25,
    )
    ax.set_ylim(bottom=0)
    ax.margins(x=0.06)

    fig.tight_layout()
    for out_path in [
        EXPORT_DIR / "data_amount_sweep-v2.pdf",
        EXPORT_DIR / "data_amount_sweep-v2.png",
    ]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".png":
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04, dpi=300)
        else:
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    plt.close(fig)


def plot_surrogate_data_amount_sweep() -> None:
    use_paper_style()

    diag_dir = TRANSMON_DIR / "results" / "data_amount_sweep_uniform_diag"
    tuned_df = pd.read_csv(diag_dir / "data_amount_sweep_uniform_diag_surrogate_only_tuned_hp.csv")

    metrics = [
        (r"$f_q$", "surrogate_test_omega_q_mean_pct", ORANGE, ORANGE_LIGHT, "o"),
        (r"$\alpha$", "surrogate_test_alpha_mean_pct", PURPLE, PURPLE_LIGHT, "s"),
    ]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.45))
    for label, col, color, fill, marker in metrics:
        tuned = tuned_df.groupby("training_percent")[col].agg(["mean", "std"]).reset_index()
        ax.plot(
            tuned["training_percent"],
            tuned["mean"],
            marker=marker,
            markersize=3.6,
            linewidth=1.35,
            color=color,
            label=label,
            zorder=3,
        )
        ax.fill_between(
            tuned["training_percent"],
            tuned["mean"] - tuned["std"].fillna(0),
            tuned["mean"] + tuned["std"].fillna(0),
            color=fill,
            alpha=0.62,
            linewidth=0,
            zorder=1,
        )

    x_ticks = np.sort(tuned_df["training_percent"].unique())
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{v:.0f}" for v in x_ticks])
    ax.set_xlabel("Training set used [%]")
    ax.set_ylabel("Mean test error [%]")
    ax.set_title("Surrogate training fraction sweep")
    ax.grid(axis="y", linestyle=":", color=GRID)
    close_plot_box(ax)
    ax.legend(
        loc="upper right",
        ncol=2,
        frameon=True,
        edgecolor="#CCCCCC",
        facecolor="white",
        handlelength=1.6,
        columnspacing=0.75,
        borderpad=0.3,
        labelspacing=0.3,
    )
    ax.set_ylim(bottom=0)
    ax.margins(x=0.05)

    fig.tight_layout()
    for out_path in [
        EXPORT_DIR / "data_amount_sweep_surrogate-v2.pdf",
        EXPORT_DIR / "data_amount_sweep_surrogate-v2.png",
    ]:
        if out_path.suffix == ".png":
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04, dpi=300)
        else:
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    plt.close(fig)


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
    if "source" in df.columns:
        reference_rows = df[df["source"].astype(str).eq("saved_best_inverse_model")]
    else:
        reference_rows = pd.DataFrame()
    reference_row = reference_rows.iloc[0] if not reference_rows.empty else df.loc[df["best_val_loss"].idxmin()]

    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, 2.9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.065], hspace=0.5, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[1, 0])
    heatmap_values = heatmap_df.values.astype(float)
    positive_values = heatmap_values[np.isfinite(heatmap_values) & (heatmap_values > 0)]
    if positive_values.size == 0:
        raise ValueError("Architecture sweep heatmap needs positive values for log color scaling.")
    norm = LogNorm(vmin=positive_values.min(), vmax=positive_values.max())
    im = ax0.imshow(
        heatmap_values,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    ax0.set_xticks(np.arange(len(heatmap_df.columns)))
    ax0.set_xticklabels([str(int(v)) for v in heatmap_df.columns])
    ax0.set_yticks(np.arange(len(heatmap_df.index)))
    ax0.set_yticklabels([str(int(v)) for v in heatmap_df.index])
    ax0.set_xlabel("Width")
    ax0.set_ylabel("Depth")
    ax0.set_title("Inverse MLP architecture sweep")
    for spine in ax0.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color(SPINE)

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
                fontsize=7.2,
                color="white" if norm(value) < 0.12 else TEXT,
                fontweight="bold",
            )

    reference_depth = int(reference_row["depth"])
    reference_width = int(reference_row["width"])
    reference_col = list(heatmap_df.columns).index(reference_width)
    reference_row_idx = list(heatmap_df.index).index(reference_depth)
    ax0.scatter(
        reference_col,
        reference_row_idx - 0.20,
        marker="*",
        s=130,
        color="#F5F5F5",
        edgecolor="#173717",
        linewidth=1.0,
        zorder=5,
    )

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Best Validation Loss (log scale)")
    cbar.locator = LogLocator(base=10)
    cbar.formatter = LogFormatterMathtext(base=10)
    cbar.update_ticks()
    cbar.outline.set_edgecolor(SPINE)
    cbar.outline.set_linewidth(0.8)

    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.24, top=0.88)
    png_path = EXPORT_DIR / "architecture_sweep-v2.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    print(f"wrote {png_path.relative_to(REPO_ROOT)}")
    save_figure(fig, EXPORT_DIR / "architecture_sweep-v2.pdf")


def plot_tuner_correlations() -> None:
    use_paper_style()

    df = load_transmon_trials()
    best_idx = df["val_loss"].idxmin()
    best = df.loc[best_idx]

    feature_specs: list[tuple[str, str, str]] = [
        ("learning_rate", "Learning rate", "log"),
        ("l2_reg", "L2 reg.", "log"),
        ("dropout_rate", "Dropout rate", "linear"),
        ("penalty_weight", "Penalty weight", "log"),
        ("n_layers", "Hidden layers", "discrete"),
        ("total_trainable_params", "Trainable params", "log"),
        ("use_batchnorm", "Batch norm.", "bool"),
    ]

    correlations = []
    for col, label, kind in feature_specs:
        if col not in df.columns:
            continue
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

    fig = plt.figure(figsize=(FULL_WIDTH_IN, 5.15))
    gs = gridspec.GridSpec(
        2,
        3,
        width_ratios=[1.12, 1.0, 1.0],
        hspace=0.58,
        wspace=0.48,
        figure=fig,
    )

    ax_bar = fig.add_subplot(gs[:, 0])
    bar_df = corr_df.iloc[::-1]
    bar_colors = [ORANGE if rho > 0 else PURPLE for rho in bar_df["rho"]]
    ax_bar.barh(bar_df["label"], bar_df["rho"], color=bar_colors, alpha=0.85)
    ax_bar.axvline(0, color=SPINE, linewidth=0.9)
    ax_bar.set_xlabel("Spearman correlation\nwith val loss")
    ax_bar.set_title("Hyperparameter sensitivity")
    ax_bar.grid(axis="x", linestyle=":", color=GRID)
    close_plot_box(ax_bar)

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
                ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
                ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
                ax.xaxis.set_minor_locator(LogLocator(base=10, subs=[]))
            if col == "total_trainable_params":
                ax.ticklabel_format(axis="y", style="plain")

        ax.set_title(f"{label}\nrho = {feature['rho']:+.2f}")
        ax.set_ylabel("Val loss")
        ax.grid(axis="y", linestyle=":", color=GRID)
        close_plot_box(ax)

        if col == "total_trainable_params":
            ax.set_xlabel("trainable params")
        else:
            ax.set_xlabel(label)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        edgecolor="#CCCCCC",
        facecolor="white",
        bbox_to_anchor=(0.62, 0.018),
        handlelength=1.6,
        columnspacing=1.0,
        borderpad=0.45,
    )
    fig.suptitle(
        f"Keras Tuner hyperparameter search  ({len(df)} trials)",
        y=0.985,
        fontsize=11,
        fontweight="normal",
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.88, bottom=0.18)
    save_figure(fig, EXPORT_DIR / "keras_tuner_hyperparameter_search-v2.pdf")


def plot_inverse_surrogate_boxplot() -> None:
    use_paper_style()

    # The 97 usable EM simulator-validated designs quoted in the manuscript
    # (median/mean f_q 0.532%/0.734%, alpha 1.138%/1.575%).
    records = json.loads(
        (TRANSMON_DIR / "results" / "validation" / "final_inverse+surrogate_em_sim_results.json").read_text()
    )
    # Source JSON stores percent errors; the plot reports fractional error.
    inv_fq = np.array([row["percent_error_frequency"] for row in records]) / 100.0
    inv_ah = np.array([row["percent_error_anharmonicity"] for row in records]) / 100.0

    # Surrogate-only test-set errors for the same frozen forward surrogate used
    # in the tandem (best_keras_model_model2_surrogate), exported by ml_15's
    # E_C -> (f_q, alpha) conversion on the 291 held-out test samples.
    surr_df = pd.read_csv(TRANSMON_DIR / "results" / "predictions" / "model2_fq_alpha_from_EC.csv")
    surr_fq = np.abs((surr_df["fq_pred_GHz"] - surr_df["fq_ref_GHz"]) / surr_df["fq_ref_GHz"]).to_numpy()
    surr_ah = np.abs((surr_df["alpha_pred_MHz"] - surr_df["alpha_ref_MHz"]) / surr_df["alpha_ref_MHz"]).to_numpy()

    data = [inv_fq, surr_fq, inv_ah, surr_ah]
    positions = [1.0, 1.72, 2.86, 3.58]
    edge_colors = [ORANGE, ORANGE, PURPLE, PURPLE]
    fill_colors = [ORANGE_LIGHT, "white", PURPLE_LIGHT, "white"]
    hatches = [None, "/////", None, "/////"]

    for label, values in zip(
        ("inv+surr f_q", "surrogate f_q", "inv+surr alpha", "surrogate alpha"), data
    ):
        print(
            f"  {label}: median {100 * np.median(values):.3f}%  mean {100 * np.mean(values):.3f}%  n={len(values)}"
        )

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.6))
    box = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        widths=0.42,
        showfliers=False,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor=TEXT, markersize=4.5),
        medianprops=dict(color=TEXT_DIM, linewidth=1.5),
        whiskerprops=dict(color=TEXT_DIM, linewidth=1.0),
        capprops=dict(color=TEXT_DIM, linewidth=1.0),
    )
    for patch, face, edge, hatch in zip(box["boxes"], fill_colors, edge_colors, hatches):
        patch.set_facecolor(face)
        patch.set_edgecolor(edge)
        patch.set_linewidth(1.2)
        patch.set_alpha(0.46)
        if hatch is not None:
            patch.set_hatch(hatch)
        patch.set_zorder(2)
    for key in ("whiskers", "caps", "medians", "means"):
        for artist in box[key]:
            artist.set_zorder(6)

    rng = np.random.default_rng(0)
    for pos, values, color in zip(positions, data, edge_colors):
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        visible_values = values[(values >= lower) & (values <= upper)]
        jitter = rng.normal(0, 0.035, size=len(visible_values))
        ax.scatter(
            np.full_like(visible_values, pos) + jitter,
            visible_values,
            s=9,
            color=color,
            alpha=0.36,
            edgecolors="none",
            linewidths=0,
            zorder=5,
        )

    ax.set_xticks([np.mean(positions[:2]), np.mean(positions[2:])])
    ax.set_xticklabels([r"$f_q$", r"$\alpha$"])
    ax.set_xlim(positions[0] - 0.55, positions[-1] + 0.55)
    ax.set_ylabel("Error")
    ax.set_title("Hamiltonian reconstruction error")
    legend_handles = [
        Patch(facecolor="#E4E4E4", edgecolor=TEXT_DIM, linewidth=1.2, label="inverse + surrogate"),
        Patch(facecolor="white", edgecolor=TEXT_DIM, linewidth=1.2, hatch="/////", label="surrogate only"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        edgecolor="#CCCCCC",
        facecolor="white",
        handlelength=1.6,
        borderpad=0.35,
        labelspacing=0.32,
    )
    ax.grid(axis="y", linestyle=":", color=GRID)
    close_plot_box(ax)
    fig.tight_layout()
    out_paths = [
        EXPORT_DIR / "inverse_surrogate_percent_error_boxplot.pdf",
        TRANSMON_DIR / "plots" / "inverse_surrogate_percent_error_boxplot.pdf",
        TRANSMON_DIR / "plots" / "inverse_surrogate_percent_error_boxplot.png",
    ]
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".png":
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04, dpi=300)
        else:
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    plt.close(fig)


def _draw_error_boxplot(
    ax: plt.Axes,
    data: list[np.ndarray],
    positions: list[float],
    edge_colors: list[str],
    fill_colors: list[str],
    *,
    width: float = 0.42,
) -> None:
    """Shared boxplot styling used by the inverse-only appendix figures.

    Matches the main-text Hamiltonian-reconstruction boxplot
    (plot_inverse_surrogate_boxplot): light colored boxes at alpha 0.46,
    grey medians/whiskers, white diamond means, and jittered sample points.
    """
    box = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        widths=width,
        showfliers=False,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor=TEXT, markersize=4.5),
        medianprops=dict(color=TEXT_DIM, linewidth=1.5),
        whiskerprops=dict(color=TEXT_DIM, linewidth=1.0),
        capprops=dict(color=TEXT_DIM, linewidth=1.0),
    )
    for patch, face, edge in zip(box["boxes"], fill_colors, edge_colors):
        patch.set_facecolor(face)
        patch.set_edgecolor(edge)
        patch.set_linewidth(1.2)
        patch.set_alpha(0.46)
        patch.set_zorder(2)
    for key in ("whiskers", "caps", "medians", "means"):
        for artist in box[key]:
            artist.set_zorder(6)

    rng = np.random.default_rng(0)
    for pos, values, color in zip(positions, data, edge_colors):
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        visible = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
        jitter = rng.normal(0, 0.035, size=len(visible))
        ax.scatter(
            np.full_like(visible, pos) + jitter,
            visible,
            s=9,
            color=color,
            alpha=0.36,
            edgecolors="none",
            linewidths=0,
            zorder=5,
        )


def plot_inverse_only_em_sim_boxplot() -> None:
    use_paper_style()

    # EM-validated results for the 50 inverse-only designs (cap-matrix inverse
    # model trained with a geometry-matching loss, no surrogate in the loop).
    # Each record pairs the SQuADDS reference (ref_*) with the EM simulator
    # capacitance matrix and scqubits Hamiltonian of the predicted design
    # (pred_*). This replaces the legacy transmon2/transmon3 appendix panels
    # with figures in the main-text boxplot style.
    records = json.loads(
        (TRANSMON_DIR / "results" / "validation" / "inverse_only_cap_matrix_em_sim_results.json").read_text()
    )
    frac_err = lambda a, r: np.abs((a - r) / r)

    # --- Hamiltonian-level errors: f_q, alpha ----------------------------
    # f_q and alpha come straight from the recorded scqubits Hamiltonians.
    fq_ref = np.array([r["ref_H_params"]["qubit_frequency_GHz"] for r in records])
    fq_prd = np.array([r["pred_H_params"]["qubit_frequency_GHz"] for r in records])
    al_ref = np.array([r["ref_H_params"]["anharmonicity_MHz"] for r in records])
    al_prd = np.array([r["pred_H_params"]["anharmonicity_MHz"] for r in records])

    ham_data = [frac_err(fq_prd, fq_ref), frac_err(al_prd, al_ref)]
    ham_labels = (r"$f_q$", r"$\alpha$")
    ham_colors = [ORANGE, PURPLE]
    ham_fills = [ORANGE_LIGHT, PURPLE_LIGHT]

    for name, values in zip(("f_q", "alpha"), ham_data):
        print(f"  inverse-only {name}: median {100 * np.median(values):.3f}%  mean {100 * np.mean(values):.3f}%  n={len(values)}")

    positions = [1.0, 2.0]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.6))
    _draw_error_boxplot(ax, ham_data, positions, ham_colors, ham_fills)
    ax.set_xticks(positions)
    ax.set_xticklabels(ham_labels)
    ax.set_xlim(positions[0] - 0.55, positions[-1] + 0.55)
    ax.set_ylabel("Error")
    ax.set_title("Inverse-only Hamiltonian error, EM validated")
    ax.grid(axis="y", linestyle=":", color=GRID)
    close_plot_box(ax)
    fig.tight_layout()
    for out_path in [
        EXPORT_DIR / "inverse_only_hamiltonian_error_boxplot.pdf",
        EXPORT_DIR / "inverse_only_hamiltonian_error_boxplot.png",
    ]:
        if out_path.suffix == ".png":
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04, dpi=300)
        else:
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    plt.close(fig)

    # --- Capacitance-matrix element errors -------------------------------
    # All capacitances are one homogeneous quantity, so they share the green
    # capacitance accent. Compact subscripts: c=cross, l=claw, g=ground.
    cap_elems = [
        ("cross_to_cross", r"$C_{cc}$"),
        ("claw_to_claw", r"$C_{ll}$"),
        ("ground_to_ground", r"$C_{gg}$"),
        ("cross_to_ground", r"$C_{cg}$"),
        ("claw_to_ground", r"$C_{lg}$"),
        ("cross_to_claw", r"$C_{cl}$"),
    ]
    cap_data, cap_labels = [], []
    for key, label in cap_elems:
        ref = np.array([r["ref_cap_matrix"][key] for r in records])
        prd = np.array([r["pred_cap_matrix"][key] for r in records])
        errs = frac_err(prd, ref)
        cap_data.append(errs)
        cap_labels.append(label)
        print(f"  inverse-only {key}: median {100 * np.median(errs):.3f}%  mean {100 * np.mean(errs):.3f}%  n={len(errs)}")

    cap_positions = [float(i + 1) for i in range(len(cap_data))]
    cap_colors = [GREEN] * len(cap_data)
    cap_fills = [GREEN_LIGHT] * len(cap_data)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.6))
    _draw_error_boxplot(ax, cap_data, cap_positions, cap_colors, cap_fills, width=0.5)
    ax.set_xticks(cap_positions)
    ax.set_xticklabels(cap_labels)
    ax.set_xlim(cap_positions[0] - 0.6, cap_positions[-1] + 0.6)
    ax.set_ylabel("Error")
    ax.set_title("Inverse-only capacitance error, EM validated")
    ax.grid(axis="y", linestyle=":", color=GRID)
    close_plot_box(ax)
    fig.tight_layout()
    for out_path in [
        EXPORT_DIR / "inverse_only_capacitance_error_boxplot.pdf",
        EXPORT_DIR / "inverse_only_capacitance_error_boxplot.png",
    ]:
        if out_path.suffix == ".png":
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04, dpi=300)
        else:
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    plt.close(fig)


def plot_combined_capacitance_error_vs_em_sim_boxplot() -> None:
    """Capacitance-matrix error of the EM-simulated designs vs. SQuADDS.

    Replaces the old predicted-vs-reference capacitance comparison figure with
    a boxplot in the shared main-text style. The CSV stores one row per
    (sample, capacitance element) with the SQuADDS target (ref_unscaled) and
    the EM simulation of the predicted design (ansys_unscaled); we plot
    the fractional error |simulated - ref| / ref for each element, i.e. how well
    the realized (EM-simulated) capacitances match the SQuADDS targets, using
    the same green capacitance accent and _draw_error_boxplot styling as the
    inverse-only figure.
    """
    use_paper_style()

    csv_path = (
        TRANSMON_DIR / "results" / "validation" / "combined_models_em_sim_capacitance_results.csv"
    )
    df = pd.read_csv(csv_path)
    frac_err = lambda a, r: np.abs((a - r) / r)

    # Same subscript convention as the inverse-only figure: c=cross, l=claw,
    # g=ground. Ordered diagonal-first (self-capacitances) then off-diagonal.
    cap_elems = [
        ("cross_to_cross", r"$C_{cc}$"),
        ("claw_to_claw", r"$C_{ll}$"),
        ("ground_to_ground", r"$C_{gg}$"),
        ("cross_to_ground", r"$C_{cg}$"),
        ("claw_to_ground", r"$C_{lg}$"),
        ("cross_to_claw", r"$C_{cl}$"),
    ]
    cap_data, cap_labels = [], []
    for key, label in cap_elems:
        rows = df[df["param"] == key]
        errs = frac_err(rows["ansys_unscaled"].to_numpy(), rows["ref_unscaled"].to_numpy())
        cap_data.append(errs)
        cap_labels.append(label)
        print(
            f"  EM simulator vs squadds {key}: median {100 * np.median(errs):.3f}%  "
            f"mean {100 * np.mean(errs):.3f}%  n={len(errs)}"
        )

    cap_positions = [float(i + 1) for i in range(len(cap_data))]
    cap_colors = [GREEN] * len(cap_data)
    cap_fills = [GREEN_LIGHT] * len(cap_data)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.6))
    _draw_error_boxplot(ax, cap_data, cap_positions, cap_colors, cap_fills, width=0.5)
    ax.set_xticks(cap_positions)
    ax.set_xticklabels(cap_labels)
    ax.set_xlim(cap_positions[0] - 0.6, cap_positions[-1] + 0.6)
    ax.set_ylabel("Error")
    ax.set_title("Inverse + surrogate capacitance error, EM validated")
    ax.grid(axis="y", linestyle=":", color=GRID)
    close_plot_box(ax)
    fig.tight_layout()
    for out_path in [
        EXPORT_DIR / "combined_capacitance_error_vs_em_sim_boxplot.pdf",
        EXPORT_DIR / "combined_capacitance_error_vs_em_sim_boxplot.png",
    ]:
        if out_path.suffix == ".png":
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04, dpi=300)
        else:
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    plt.close(fig)


def plot_inverse_surrogate_error_histograms() -> None:
    use_paper_style()

    df = pd.read_csv(TRANSMON_DIR / "results" / "validation" / "inverse+surrogate_percentErrors.csv")
    series = [
        (df["frequency"].to_numpy(), r"$f_q$", ORANGE, ORANGE_LIGHT),
        (df["anharmonicity"].to_numpy(), r"$\alpha$", PURPLE, PURPLE_LIGHT),
    ]

    max_error = max(float(np.nanmax(values)) for values, _, _, _ in series)
    bins = np.linspace(0, max_error * 1.04, 18)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.45), sharey=True)
    for ax, (values, label, edge, fill) in zip(axes, series):
        ax.hist(values, bins=bins, color=fill, edgecolor=edge, linewidth=1.2, alpha=0.9)
        ax.axvline(np.median(values), color=edge, linewidth=1.5, linestyle="-", label=f"median {np.median(values):.2f}%")
        ax.axvline(np.mean(values), color=TEXT_DIM, linewidth=1.3, linestyle="--", label=f"mean {np.mean(values):.2f}%")
        ax.set_title(label)
        ax.set_xlabel("Percent error [%]")
        ax.grid(axis="y", linestyle=":", color=GRID)
        close_plot_box(ax)
        ax.legend(loc="upper right", frameon=True, edgecolor="#CCCCCC", facecolor="white")

    axes[0].set_ylabel("Counts")
    fig.suptitle("Inverse + surrogate error distribution", y=0.99, fontsize=10.5, fontweight="normal")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_paths = [
        EXPORT_DIR / "inverse_surrogate_percent_error_histograms.pdf",
        TRANSMON_DIR / "plots" / "inverse_surrogate_percent_error_histograms.pdf",
        TRANSMON_DIR / "plots" / "inverse_surrogate_percent_error_histograms.png",
    ]
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".png":
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04, dpi=300)
        else:
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    plt.close(fig)


def plot_em_sim_validation_vs_nn_distance() -> None:
    use_paper_style()

    validation_dir = TRANSMON_DIR / "results" / "validation"
    data_path = validation_dir / "random_candidates_tested_with_ansys_nn_bins.json"
    if not data_path.exists():
        data_path = validation_dir / "surrogate_stress_test_em_sim_results.json"
    data = json.loads(data_path.read_text())
    candidate_path = validation_dir / "random_candidates_for_em_sim_validation_nn_bins.csv"
    candidate_df = pd.read_csv(candidate_path)
    param_cols = [
        "design_options.connection_pads.readout.claw_length",
        "design_options.connection_pads.readout.ground_spacing",
        "design_options.cross_length",
    ]
    candidate_values_um = candidate_df[param_cols].to_numpy(dtype=float)
    if np.nanmax(np.abs(candidate_values_um)) < 1.0:
        candidate_values_um = candidate_values_um * 1e6
    nearest_hamiltonian = nearest_training_hamiltonian_for_candidates(candidate_values_um)

    raw_sample_ids = [
        int(row.get("Sample", row.get("sample_number", row_idx)))
        for row_idx, row in enumerate(data)
    ]
    sample_offset = 1 if min(raw_sample_ids) == 1 and max(raw_sample_ids) == len(candidate_df) else 0
    for row, sample_id in zip(data, raw_sample_ids):
        row["_candidate_idx"] = sample_id - sample_offset

    bins = sorted({int(row["nn_bin"]) for row in data})
    eps = 1e-15

    fq_pts = []
    ah_pts = []
    nn_fq_pts = []
    nn_ah_pts = []
    nn_pts = []

    for bin_id in bins:
        rows = [row for row in data if int(row["nn_bin"]) == bin_id]
        sample_idx = np.array([int(row["_candidate_idx"]) for row in rows])
        fq = np.array(
            [
                abs(row["pred_H_params"]["qubit_frequency_GHz"] - row["surrogate_H_params"]["qubit_frequency_GHz"])
                / (abs(row["pred_H_params"]["qubit_frequency_GHz"]) + eps)
                for row in rows
            ]
        )
        ah = np.array(
            [
                abs(row["pred_H_params"]["anharmonicity_MHz"] - row["surrogate_H_params"]["anharmonicity_MHz"])
                / (abs(row["pred_H_params"]["anharmonicity_MHz"]) + eps)
                for row in rows
            ]
        )
        nn_fq = np.array(
            [
                abs(row["pred_H_params"]["qubit_frequency_GHz"] - nearest_hamiltonian[sample_idx[i], 0])
                / (abs(row["pred_H_params"]["qubit_frequency_GHz"]) + eps)
                for i, row in enumerate(rows)
            ]
        )
        nn_ah = np.array(
            [
                abs(row["pred_H_params"]["anharmonicity_MHz"] - nearest_hamiltonian[sample_idx[i], 1])
                / (abs(row["pred_H_params"]["anharmonicity_MHz"]) + eps)
                for i, row in enumerate(rows)
            ]
        )
        nn = np.array([row["nn_distance_scaled"] for row in rows])

        fq_pts.append(fq)
        ah_pts.append(ah)
        nn_fq_pts.append(nn_fq)
        nn_ah_pts.append(nn_ah)
        nn_pts.append(nn)

    def qstats(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        med = np.array([np.median(v) for v in values])
        q1 = np.array([np.percentile(v, 25) for v in values])
        q3 = np.array([np.percentile(v, 75) for v in values])
        return med, q1, q3

    fq_med, fq_q1, fq_q3 = qstats(fq_pts)
    ah_med, ah_q1, ah_q3 = qstats(ah_pts)
    nn_fq_med, nn_fq_q1, nn_fq_q3 = qstats(nn_fq_pts)
    nn_ah_med, nn_ah_q1, nn_ah_q3 = qstats(nn_ah_pts)
    nn_med, nn_q1, nn_q3 = qstats(nn_pts)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 3.0))
    for idx in range(len(bins)):
        ax.scatter(nn_pts[idx], fq_pts[idx], color=ORANGE, alpha=0.16, s=8, linewidths=0, zorder=2)
        ax.scatter(nn_pts[idx], ah_pts[idx], color=PURPLE, alpha=0.16, s=8, linewidths=0, zorder=2)
        ax.scatter(nn_pts[idx], nn_fq_pts[idx], color=ORANGE, alpha=0.16, s=8, linewidths=0, zorder=2)
        ax.scatter(nn_pts[idx], nn_ah_pts[idx], color=PURPLE, alpha=0.16, s=8, linewidths=0, zorder=2)

    series = [
        (fq_med, fq_q1, fq_q3, ORANGE, "o", "-", ORANGE, r"$f_q$ surrogate"),
        (ah_med, ah_q1, ah_q3, PURPLE, "s", "-", PURPLE, r"$\alpha$ surrogate"),
        (nn_fq_med, nn_fq_q1, nn_fq_q3, ORANGE, "o", "--", "white", r"$f_q$ nearest neighbor"),
        (nn_ah_med, nn_ah_q1, nn_ah_q3, PURPLE, "s", "--", "white", r"$\alpha$ nearest neighbor"),
    ]
    for med, q1, q3, color, marker, linestyle, face, label in series:
        ax.errorbar(
            nn_med,
            med,
            xerr=[nn_med - nn_q1, nn_q3 - nn_med],
            yerr=[med - q1, q3 - med],
            color=color,
            marker=marker,
            markersize=5,
            markerfacecolor=face,
            markeredgecolor=color,
            linestyle=linestyle,
            linewidth=1.6,
            elinewidth=1.1,
            capsize=3.0,
            capthick=1.0,
            label=label,
            zorder=5,
        )

    ax.legend(
        loc="upper left",
        frameon=True,
        edgecolor="#CCCCCC",
        facecolor="white",
        handlelength=1.8,
        borderpad=0.35,
        labelspacing=0.32,
        title="median + IQR",
        title_fontsize=8.5,
    )
    ax.set_xlabel("Scaled NN distance")
    ax.set_ylabel("Error")
    ax.set_title("Validation error vs NN distance")
    y_max = max(
        np.nanmax(fq_q3),
        np.nanmax(ah_q3),
        np.nanmax(nn_fq_q3),
        np.nanmax(nn_ah_q3),
        np.nanmax([np.nanmax(v) for v in fq_pts + ah_pts + nn_fq_pts + nn_ah_pts]),
    )
    ax.set_xlim(0, max(np.nanmax(v) for v in nn_pts) * 1.06)
    ax.set_ylim(0, y_max * 1.18)
    ax.grid(axis="y", linestyle=":", color=GRID)
    close_plot_box(ax)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.91, bottom=0.15)
    out_paths = [
        EXPORT_DIR / "em_sim_validation_error_vs_nn_distance-v2.pdf",
        EXPORT_DIR / "em_sim_validation_error_vs_nn_distance-v2.png",
        TRANSMON_DIR / "plots" / "em_sim_validation_error_vs_nn_distance-v2.pdf",
        TRANSMON_DIR / "plots" / "em_sim_validation_error_vs_nn_distance-v2.png",
    ]
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".png":
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04, dpi=300)
        else:
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    plt.close(fig)


def plot_corner_sweep_distance_and_em_sim_error() -> None:
    """Combined far-corner sweep figure: training-to-corner distance (top) and
    EM-simulator-validated fractional errors for f_q and alpha (middle/bottom), all
    sharing the training-pool-percent x axis."""
    use_paper_style()

    distance_path = (
        TRANSMON_DIR / "plots" / "corner_far_to_near_sweep_retrained_surrogate" / "corner_distance_summary.csv"
    )
    error_path = (
        TRANSMON_DIR
        / "results"
        / "validation"
        / "data_amount_sweep_heldout_corner_em_sim"
        / "heldout_corner_test_em_sim_error_summary_by_training_percent.csv"
    )
    dist = pd.read_csv(distance_path).sort_values("training_percent")
    errs = pd.read_csv(error_path).sort_values("training_percent")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(COLUMN_WIDTH_IN, 5.4),
        sharex=True,
        gridspec_kw={"hspace": 0.34},
    )
    ax_dist, ax_fq, ax_ah = axes

    x_dist = dist["training_percent"].to_numpy(dtype=float)
    mean_d = dist["scaled_mean"].to_numpy(dtype=float)
    std_d = dist["scaled_std"].to_numpy(dtype=float)
    lower_d = np.minimum(std_d, mean_d)
    ax_dist.fill_between(
        x_dist,
        dist["scaled_p25"].to_numpy(dtype=float),
        dist["scaled_p75"].to_numpy(dtype=float),
        color=GREEN_LIGHT,
        alpha=0.7,
        linewidth=0,
        zorder=1,
        label="Interquartile range",
    )
    ax_dist.errorbar(
        x_dist,
        mean_d,
        yerr=np.vstack([lower_d, std_d]),
        fmt="o-",
        color=GREEN,
        markersize=3.6,
        linewidth=1.5,
        elinewidth=0.9,
        capsize=2.2,
        capthick=0.9,
        zorder=3,
        label=r"Mean $\pm$ 1 SD",
    )
    ax_dist.set_ylabel("Scaled distance\nto corner")
    ax_dist.set_title("Training data distance to held-out corner", loc="left", pad=2)
    ax_dist.set_ylim(bottom=0)
    ax_dist.legend(
        loc="lower left",
        frameon=True,
        edgecolor="#CCCCCC",
        facecolor="white",
        handlelength=1.4,
        borderpad=0.35,
        labelspacing=0.3,
    )

    def draw_error_axis(ax: plt.Axes, metric: str, ylabel: str, panel_title: str, show_legend: bool) -> None:
        x = errs["training_percent"].to_numpy(dtype=float)
        mean = errs[f"ansys_{metric}_mean"].to_numpy(dtype=float) / 100.0
        std = errs[f"ansys_{metric}_std"].fillna(0).to_numpy(dtype=float) / 100.0
        median = errs[f"ansys_{metric}_median"].to_numpy(dtype=float) / 100.0
        lower = np.maximum(mean - std, 1e-5)
        upper = mean + std

        ax.plot(x, mean, marker="o", markersize=3.7, linewidth=1.45, color=GREEN, label="Mean", zorder=3)
        ax.fill_between(x, lower, upper, where=np.isfinite(mean), color=GREEN_LIGHT, alpha=0.65, linewidth=0, zorder=1)
        ax.plot(x, median, marker="s", markersize=3.1, linewidth=1.0, color=PURPLE, linestyle="--", label="Median", zorder=4)

        partial = errs["n_ansys"].lt(errs["n_requested"]) & errs[f"ansys_{metric}_mean"].notna()
        ax.scatter(
            errs.loc[partial, "training_percent"],
            errs.loc[partial, f"ansys_{metric}_mean"] / 100.0,
            s=30,
            marker="o",
            facecolors="white",
            edgecolors=GREEN,
            linewidths=1.0,
            zorder=5,
        )

        finite_y = np.concatenate([values[np.isfinite(values) & (values > 0)] for values in [mean, median]])
        if finite_y.size:
            ax.set_yscale("log")
            ax.set_ylim(max(finite_y.min() / 2.5, 5e-4), finite_y.max() * 2.0)
        for _, row in errs.iterrows():
            n = int(row["n_ansys"])
            requested_n = int(row["n_requested"])
            y_value = row[f"ansys_{metric}_mean"] / 100.0
            if n == 0:
                ax.text(row["training_percent"], ax.get_ylim()[0] * 1.18, "n=0", ha="center", va="bottom", fontsize=6.8, color=TEXT_DIM)
            elif n < requested_n and pd.notna(y_value):
                ax.text(row["training_percent"], y_value * 1.16, f"n={n}", ha="center", va="bottom", fontsize=6.8, color=TEXT_DIM)

        ax.set_ylabel(ylabel)
        ax.set_title(panel_title, loc="left", pad=2)
        if show_legend:
            ax.legend(
                loc="upper right",
                frameon=True,
                edgecolor="#CCCCCC",
                facecolor="white",
                handlelength=1.2,
                borderpad=0.32,
                labelspacing=0.32,
            )

    draw_error_axis(ax_fq, "frequency", r"$f_q$ error", r"EM simulator validation, qubit frequency ($f_q$)", True)
    draw_error_axis(ax_ah, "anharmonicity", r"$\alpha$ error", r"EM simulator validation, anharmonicity ($\alpha$)", False)

    x_ticks = errs["training_percent"].to_numpy(dtype=float)
    ax_ah.set_xticks(x_ticks)
    ax_ah.set_xticklabels([f"{v:.0f}" for v in x_ticks])
    ax_ah.set_xlabel("Far-corner training pool used [%]")
    for ax in axes:
        ax.grid(axis="y", linestyle=":", color=GRID)
        close_plot_box(ax)
        ax.margins(x=0.04)

    fig.subplots_adjust(left=0.17, right=0.98, top=0.96, bottom=0.08)
    out_paths = [
        EXPORT_DIR / "corner_sweep_distance_and_em_sim_error.pdf",
        EXPORT_DIR / "corner_sweep_distance_and_em_sim_error.png",
        TRANSMON_DIR / "plots" / "corner_far_to_near_sweep_retrained_surrogate" / "corner_sweep_distance_and_em_sim_error.pdf",
        TRANSMON_DIR / "plots" / "corner_far_to_near_sweep_retrained_surrogate" / "corner_sweep_distance_and_em_sim_error.png",
    ]
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".png":
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04, dpi=300)
        else:
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    plt.close(fig)


def _plot_stress_pairs_panels(
    pair_indices: list[tuple[int, int]],
    out_names: list[str],
    fig_height: float,
) -> None:
    use_paper_style()

    train_values_um = load_transmon_design_splits_um()["Train"]
    candidate_path = TRANSMON_DIR / "results" / "validation" / "random_candidates_for_em_sim_validation_nn_bins.csv"
    df = pd.read_csv(candidate_path)

    param_cols = [
        "design_options.connection_pads.readout.claw_length",
        "design_options.connection_pads.readout.ground_spacing",
        "design_options.cross_length",
    ]
    labels = ["claw_length", "ground_spacing", "cross_length"]
    candidate_values_um = df[param_cols].to_numpy(dtype=float) * 1e6
    nn_distance = df["nn_distance_scaled"].to_numpy(dtype=float)

    n_panels = len(pair_indices)
    # NN distance is a continuous scale, so it gets a perceptually-uniform,
    # colorblind-safe sequential colormap (distinct from the categorical
    # orange/purple and split palettes used elsewhere).
    paper_cmap = plt.get_cmap("cividis")
    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, fig_height))
    # Keep the colorbar at roughly the same absolute height (~0.11 in)
    # regardless of how many scatter panels sit above it.
    cbar_ratio = 0.11 * n_panels / max(fig_height - 0.11, 1e-6)
    gs = gridspec.GridSpec(
        n_panels + 1,
        1,
        height_ratios=[1.0] * n_panels + [cbar_ratio],
        hspace=0.58 if n_panels > 1 else 0.42,
        figure=fig,
    )
    axes = np.array([fig.add_subplot(gs[idx, 0]) for idx in range(n_panels)])
    cax = fig.add_subplot(gs[n_panels, 0])

    sc = None
    for ax, (i, j) in zip(axes, pair_indices):
        ax.set_axisbelow(True)
        ax.scatter(
            train_values_um[:, i],
            train_values_um[:, j],
            c="#D9D9D9",
            s=6,
            alpha=0.52,
            label="Training data",
            linewidths=0,
            zorder=1,
            rasterized=True,
        )
        sc = ax.scatter(
            candidate_values_um[:, i],
            candidate_values_um[:, j],
            c=nn_distance,
            cmap=paper_cmap,
            vmin=float(nn_distance.min()),
            vmax=float(nn_distance.max()),
            s=28,
            marker="D",
            edgecolors=TEXT,
            linewidths=0.35,
            label="Validation points",
            zorder=3,
        )
        ax.set_xlabel(fr"{labels[i]} ($\mu$m)")
        ax.set_ylabel(fr"{labels[j]} ($\mu$m)")
        ax.grid(axis="both", linestyle=":", color=GRID)
        close_plot_box(ax)
        ax.ticklabel_format(axis="both", style="plain")
        ax.tick_params(axis="both", pad=2)

    axes[0].legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        facecolor="white",
        framealpha=0.95,
        handletextpad=0.45,
        borderpad=0.4,
    )

    assert sc is not None
    cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_label("NN distance")
    cbar.outline.set_edgecolor(SPINE)
    cbar.outline.set_linewidth(0.8)

    fig.subplots_adjust(left=0.2, right=0.965, bottom=0.09 * 3 / (n_panels + 1), top=0.985)
    out_paths = []
    for out_name in out_names:
        out_paths.append(TRANSMON_DIR / "plots" / out_name)
        out_paths.append(EXPORT_DIR / out_name)
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    plt.close(fig)


def plot_surrogate_stress_random_points_pairs() -> None:
    # Main-text panel: claw_length vs cross_length, combined in the manuscript
    # with the error-vs-NN-distance plot as one two-panel figure.
    _plot_stress_pairs_panels(
        [(0, 2)],
        ["surrogate_stress_random_points_middle.pdf"],
        fig_height=2.85,
    )
    # Appendix panels: the remaining two parameter pairs.
    _plot_stress_pairs_panels(
        [(0, 1), (1, 2)],
        ["surrogate_stress_random_points_pairs_appendix.pdf"],
        fig_height=4.55,
    )


def plot_model_architecture_combined() -> None:
    use_paper_style()

    palette = flowchart_palette()
    physics_fill = palette["physics_fill"]
    ml_fill = palette["ml_fill"]
    validation_fill = palette["validation_fill"]
    physics_edge = palette["physics_edge"]
    ml_edge = palette["ml_edge"]
    validation_edge = palette["validation_edge"]

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
        title_color: str | None = None,
        body_color: str = TEXT,
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
        ax.text(
            x + 1.2,
            y + h - 2.2,
            title,
            ha="left",
            va="top",
            fontsize=8.8,
            fontweight="bold",
            color=title_color or edge,
            zorder=5,
        )
        for idx, line in enumerate(lines):
            ax.text(x + 1.2, y + h - 5.6 - idx * 2.2, line, ha="left", va="top", fontsize=7.7, color=body_color, zorder=5)

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
        edgecolor=ml_edge,
        facecolor=ml_fill,
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
        color=ml_edge,
        zorder=1,
    )

    block(*boxes["input"], "Targets", [r"$f_q,\ \alpha$", "Scaled inputs"], edge=physics_edge, face=physics_fill)
    block(*boxes["inverse"], "Inverse MLP", ["1 hidden layer", "64 neurons", "387 trainable"], edge=ml_edge, face=ml_fill)
    block(*boxes["geom"], "Design", ["3 Quantum Metal", "geometry params"], edge=ml_edge, face=ml_fill)
    block(*boxes["surrogate"], "EM simulator surrogate", ["736 hidden units", "4,418 non-trainable", r"$\hat{f}_q,\ \hat{\alpha}$ check"], edge=ml_edge, face=ml_fill)
    block(
        *boxes["output"],
        "Loss",
        ["MAE in", "target", "space"],
        edge=validation_edge,
        face=validation_edge,
        title_color="#FFFFFF",
        body_color="#FFFFFF",
    )

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
        color=ml_edge,
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
        "Update inverse weights",
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        fontstyle="italic",
        color=ml_edge,
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


def plot_inverse_architecture_standalone() -> None:
    use_paper_style()
    palette = flowchart_palette()
    physics_fill = palette["physics_fill"]
    physics_edge = palette["physics_edge"]
    ml_fill = palette["ml_fill"]
    ml_edge = palette["ml_edge"]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 2.25))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    def block(x, y, w, h, title, lines, *, edge, face):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.35,rounding_size=1.2",
            linewidth=1.3,
            edgecolor=edge,
            facecolor=face,
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(
            x + 1.0,
            y + h - 2.6,
            title,
            ha="left",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color=edge,
        )
        for idx, line in enumerate(lines):
            ax.text(
                x + 1.0,
                y + h - 6.2 - 2.95 * idx,
                line,
                ha="left",
                va="top",
                fontsize=7.3,
                color=TEXT,
            )

    def arrow(start, end):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.5,
                color=TEXT_DIM,
                shrinkA=2,
                shrinkB=2,
                zorder=3,
            )
        )

    input_box = (7, 6.4, 22, 15.0)
    hidden_box = (39, 6.4, 25, 15.0)
    output_box = (74, 6.4, 19, 15.0)

    block(
        *input_box,
        "Targets",
        [r"$f_q,\ \alpha$", "Scaled inputs"],
        edge=physics_edge,
        face=physics_fill,
    )
    block(
        *hidden_box,
        "Hidden layer",
        ["Dense 64", "LeakyReLU", "387 trainable params"],
        edge=ml_edge,
        face=ml_fill,
    )
    block(
        *output_box,
        "Output",
        ["3 Quantum Metal", "geometry params"],
        edge=ml_edge,
        face=ml_fill,
    )

    arrow((input_box[0] + input_box[2], input_box[1] + input_box[3] / 2), (hidden_box[0], hidden_box[1] + hidden_box[3] / 2))
    arrow((hidden_box[0] + hidden_box[2], hidden_box[1] + hidden_box[3] / 2), (output_box[0], output_box[1] + output_box[3] / 2))

    ax.text(
        50,
        27.0,
        "Inverse model architecture",
        ha="center",
        va="center",
        fontsize=10.2,
        color=TEXT,
    )
    ax.text(
        50,
        24.5,
        "1 hidden layer with width 64   |   Trainable params: 387   |   Output params: 3",
        ha="center",
        va="center",
        fontsize=8.1,
        color=TEXT_DIM,
    )

    fig.tight_layout(pad=0.12)
    out_paths = [
        TRANSMON_DIR / "plots" / "model_architecture_paper_theme_inverse_model.pdf",
        TRANSMON_DIR / "plots" / "model_architecture_inverse_model.pdf",
        TRANSMON_DIR / "plots" / "model_architecture_inverse_model.png",
    ]
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    plt.close(fig)


def export_pdf_fallbacks() -> None:
    crop_pdf_page(19, (98, 48, 502, 346), EXPORT_DIR / "predicted_vs_reference_design_comparsion.pdf")


def export_png_fallbacks() -> None:
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
    regenerate_generated_sources()
    plot_dataset_distributions()
    plot_sample_data_distribution()
    plot_data_amount_sweep()
    plot_surrogate_data_amount_sweep()
    plot_architecture_sweep()
    plot_tuner_correlations()
    plot_inverse_surrogate_boxplot()
    plot_inverse_only_em_sim_boxplot()
    plot_combined_capacitance_error_vs_em_sim_boxplot()
    plot_inverse_surrogate_error_histograms()
    plot_em_sim_validation_vs_nn_distance()
    plot_corner_sweep_distance_and_em_sim_error()
    plot_surrogate_stress_random_points_pairs()
    plot_model_architecture_combined()
    plot_inverse_architecture_standalone()
    export_pdf_fallbacks()
    export_png_fallbacks()


if __name__ == "__main__":
    main()
