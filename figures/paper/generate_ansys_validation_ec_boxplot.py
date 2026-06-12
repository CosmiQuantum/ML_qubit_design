"""Ansys-validation version of the inverse+surrogate E_C percent-error boxplot.

Same style and layout as plot_inverse_surrogate_boxplot() in
build_manuscript_exports.py, but the percent errors are computed from a
ref/pred CSV (target E_C vs Ansys-simulated E_C of the predicted designs)
instead of the surrogate-reconstruction CSV.

Usage:
    python generate_ansys_validation_ec_boxplot.py [path/to/ref_vs_pred_EC.csv]

The CSV must have columns: ref_EC, pred_EC (GHz).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_manuscript_exports import (
    COLUMN_WIDTH_IN,
    EXPORT_DIR,
    GRID,
    ORANGE,
    ORANGE_LIGHT,
    TEXT,
    TEXT_DIM,
    TRANSMON_DIR,
    close_plot_box,
    plt,
    use_paper_style,
)

DEFAULT_CSV = Path.home() / "Downloads" / "ref_vs_pred_EC.csv"


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    df = pd.read_csv(csv_path)
    pct_errors = 100.0 * np.abs(df["pred_EC"] - df["ref_EC"]) / np.abs(df["ref_EC"])
    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"  median {np.median(pct_errors):.3f}%  mean {np.mean(pct_errors):.3f}%  "
          f"90th {np.percentile(pct_errors, 90):.3f}%  max {np.max(pct_errors):.3f}%")

    use_paper_style()

    data = [pct_errors.to_numpy()]
    edge_colors = [ORANGE]
    fill_colors = [ORANGE_LIGHT]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.6))
    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.54,
        showfliers=False,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor=TEXT, markersize=5),
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
    for idx, (values, color) in enumerate(zip(data, edge_colors), start=1):
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        visible_values = values[(values >= lower) & (values <= upper)]
        jitter = rng.normal(0, 0.045, size=len(visible_values))
        ax.scatter(
            np.full_like(visible_values, idx) + jitter,
            visible_values,
            s=11,
            color=color,
            alpha=0.42,
            edgecolors="none",
            linewidths=0,
            zorder=5,
        )

    ax.set_xticklabels([r"$E_C$"])
    ax.set_ylabel("Percent error [%]")
    ax.set_title("Inverse + surrogate reconstruction error")
    ax.grid(axis="y", linestyle=":", color=GRID)
    close_plot_box(ax)
    fig.tight_layout()

    out_paths = [
        EXPORT_DIR / "ansys_validation_percent_error_boxplot.pdf",
        TRANSMON_DIR / "plots" / "ansys_validation_percent_error_boxplot.pdf",
        TRANSMON_DIR / "plots" / "ansys_validation_percent_error_boxplot.png",
    ]
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".png":
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04, dpi=300)
        else:
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
        print(f"Written {out_path}")


if __name__ == "__main__":
    main()
