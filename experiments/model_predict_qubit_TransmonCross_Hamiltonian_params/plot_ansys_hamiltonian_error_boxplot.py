from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
_MPL_CONFIG_DIR = EXPERIMENT_DIR / "results" / "cache" / "matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VALIDATION_DIR = EXPERIMENT_DIR / "results" / "validation" / "data_amount_100pct_on_ansys_targets"
DEFAULT_INPUT_JSON = VALIDATION_DIR / "seed0_ansys_input_geometry_predictions_RESULTS.json"
DEFAULT_OUTPUT_STEM = VALIDATION_DIR / "boxplot_ansys_vs_100pct_sweep_surrogate_same_targets"
DEFAULT_BACKUP_STEM = VALIDATION_DIR / "boxplot_ansys_vs_100pct_sweep_surrogate_same_targets_legacy"
DEFAULT_DATA_CSV = VALIDATION_DIR / "seed0_ansys_input_geometry_predictions_RESULTS_boxplot_data.csv"
EPS = 1e-12


def as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percent_error(predicted: float | None, target: float | None) -> float | None:
    if predicted is None or target is None:
        return None
    return 100.0 * abs(predicted - target) / max(abs(target), EPS)


def load_boxplot_rows(path: Path) -> list[dict[str, float | int]]:
    with path.open() as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Expected {path} to contain a JSON list, got {type(raw).__name__}")

    rows: list[dict[str, float | int]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Expected item {index} in {path} to be an object")

        pred_h = item.get("pred_H_params") or {}
        ref_h = item.get("ref_H_params") or {}
        if not isinstance(pred_h, dict) or not isinstance(ref_h, dict):
            raise ValueError(f"Expected pred_H_params/ref_H_params objects for item {index}")

        ansys_freq = as_float(pred_h.get("qubit_frequency_GHz"))
        ansys_alpha = as_float(pred_h.get("anharmonicity_MHz"))
        target_freq = as_float(ref_h.get("qubit_frequency_GHz"))
        target_alpha = as_float(ref_h.get("anharmonicity_MHz"))

        freq_error = as_float(item.get("percent_error_frequency"))
        alpha_error = as_float(
            item.get("percent_error_anharmonicity", item.get("percent_error_alpha"))
        )
        if freq_error is None:
            freq_error = percent_error(ansys_freq, target_freq)
        if alpha_error is None:
            alpha_error = percent_error(ansys_alpha, target_alpha)
        if freq_error is None or alpha_error is None:
            raise ValueError(f"Could not determine frequency/anharmonicity percent errors for item {index}")

        rows.append(
            {
                "sample": int(item.get("Sample", item.get("sample", index))),
                "target_qubit_frequency_GHz": target_freq,
                "target_anharmonicity_MHz": target_alpha,
                "ansys_qubit_frequency_GHz": ansys_freq,
                "ansys_anharmonicity_MHz": ansys_alpha,
                "ansys_frequency_pct_error": freq_error,
                "ansys_anharmonicity_pct_error": alpha_error,
                "ansys_mean_hamiltonian_pct_error": (freq_error + alpha_error) / 2.0,
            }
        )

    return rows


def write_plot_data_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def backup_existing_outputs(output_stem: Path, backup_stem: Path) -> list[Path]:
    backups: list[Path] = []
    for suffix in (".pdf", ".png"):
        output_path = output_stem.with_suffix(suffix)
        backup_path = backup_stem.with_suffix(suffix)
        if output_path.exists() and not backup_path.exists():
            shutil.copy2(output_path, backup_path)
            backups.append(backup_path)
    return backups


def make_boxplot(rows: list[dict[str, float | int]], output_stem: Path) -> None:
    freq_errors = [float(row["ansys_frequency_pct_error"]) for row in rows]
    alpha_errors = [float(row["ansys_anharmonicity_pct_error"]) for row in rows]

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    boxplot_kwargs = dict(
        patch_artist=True,
        showmeans=True,
        widths=0.55,
        flierprops=dict(marker="o", markersize=3, alpha=0.35, markerfacecolor="0.45"),
        medianprops=dict(color="black", linewidth=1.4),
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=4),
    )
    labels = [r"$\omega_F$", r"$\alpha$"]
    try:
        bp = ax.boxplot([freq_errors, alpha_errors], tick_labels=labels, **boxplot_kwargs)
    except TypeError:
        bp = ax.boxplot([freq_errors, alpha_errors], labels=labels, **boxplot_kwargs)

    for patch, color in zip(bp["boxes"], ["#4c78a8", "#f58518"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    rng = random.Random(0)
    for x_pos, values in enumerate((freq_errors, alpha_errors), start=1):
        jittered_x = [x_pos + rng.uniform(-0.055, 0.055) for _ in values]
        ax.scatter(jittered_x, values, s=13, color="0.2", alpha=0.38, linewidths=0)

    ax.set_ylabel("Absolute percent error [%]")
    ax.set_title("Ansys output vs target Hamiltonian")
    ax.text(0.02, 0.96, f"n = {len(rows)}", transform=ax.transAxes, va="top", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def print_summary(rows: list[dict[str, float | int]]) -> None:
    metrics = {
        "w_F frequency": [float(row["ansys_frequency_pct_error"]) for row in rows],
        "alpha anharmonicity": [float(row["ansys_anharmonicity_pct_error"]) for row in rows],
        "mean Hamiltonian": [float(row["ansys_mean_hamiltonian_pct_error"]) for row in rows],
    }
    for label, values in metrics.items():
        arr = np.asarray(values, dtype=float)
        print(
            f"{label}: n={arr.size}, mean={arr.mean():.4f}%, "
            f"median={np.median(arr):.4f}%, min={arr.min():.4f}%, max={arr.max():.4f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Ansys-vs-target Hamiltonian percent errors for w_F and alpha."
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--backup-stem", type=Path, default=DEFAULT_BACKUP_STEM)
    parser.add_argument("--data-csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--no-backup", action="store_true", help="Do not preserve existing output files first.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_boxplot_rows(args.input_json)
    if not rows:
        raise ValueError(f"No rows found in {args.input_json}")

    if not args.no_backup:
        backups = backup_existing_outputs(args.output_stem, args.backup_stem)
        for path in backups:
            print(f"Backed up existing plot -> {path}")

    write_plot_data_csv(rows, args.data_csv)
    make_boxplot(rows, args.output_stem)
    print_summary(rows)
    print(f"Wrote plot -> {args.output_stem.with_suffix('.pdf')}")
    print(f"Wrote plot -> {args.output_stem.with_suffix('.png')}")
    print(f"Wrote data -> {args.data_csv}")


if __name__ == "__main__":
    main()
