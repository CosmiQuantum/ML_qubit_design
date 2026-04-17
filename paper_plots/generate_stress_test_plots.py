#!/usr/bin/env python3
"""
Generate publication-quality stress test plots for the Hamiltonian-params
surrogate model, matching the paper_plots color palette and typography.

This script reproduces (and restyles) the saved figures from
ml_14_surrogate_stress_test.ipynb so they are consistent with the rest
of the manuscript figures.

Changes vs. the original notebook plots:
  • All colours use the paper palette (GREEN, ORANGE, PURPLE, etc.)
  • Font family ← Helvetica / sans-serif; mathtext.fontset ← cm
  • The "Nearest training H (oracle)" baseline (previously in purple) is
    REMOVED from the final density plot per author request.

Usage:
    cd <repo_root>/model_predict_qubit-TransmonCross-Hamiltonian_params
    python ../paper_plots/generate_stress_test_plots.py

All output PDFs are written to  plots/  (relative to cwd).
"""

import os, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# ──────────────────────────────────────────────────────────────────────
# Paper colour palette & typography  (shared with other paper_plots)
# ──────────────────────────────────────────────────────────────────────
GREEN         = "#3D8B3D"
GREEN_LIGHT   = "#E8F5E8"
GREEN_DARK    = "#2E6B2E"
ORANGE        = "#E87A00"
ORANGE_LIGHT  = "#FFF4E6"
ORANGE_DARK   = "#A85600"
PURPLE        = "#7B68AE"
PURPLE_LIGHT  = "#E8E4F0"
PURPLE_DARK   = "#4A3D78"
TEXT_MAIN     = "#222222"
TEXT_DIM      = "#555555"

# Sigma-scatter colours (6 bins — one per sigma fraction)
SIGMA_COLORS = [GREEN, ORANGE, PURPLE, GREEN_DARK, ORANGE_DARK, PURPLE_DARK]


def _apply_paper_rcparams():
    """Apply global rcParams that match the paper_plots style."""
    plt.rcParams.update({
        "text.usetex":         False,
        "mathtext.fontset":    "cm",
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.labelcolor":     TEXT_MAIN,
        "xtick.color":         TEXT_MAIN,
        "ytick.color":         TEXT_MAIN,
        "axes.edgecolor":      TEXT_DIM,
        "axes.titlesize":      13,
        "axes.labelsize":      11,
        "xtick.labelsize":     10,
        "ytick.labelsize":     10,
        "legend.fontsize":     9,
        "figure.dpi":          150,
    })


# ──────────────────────────────────────────────────────────────────────
# Data loading  (must run from the Hamiltonian_params directory)
# ──────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
SCALER_DIR = "scalers"


def _load_training_data():
    """Load scaled training X/y arrays (same convention as the notebook)."""
    X_train = np.load(f"{DATA_DIR}/npy/y_train_linear_encoding_scaled.npy",
                      allow_pickle=True)
    y_train = np.load(f"{DATA_DIR}/npy/x_train_linear_encoding_scaled.npy",
                      allow_pickle=True)
    return X_train.astype("float64"), y_train.astype("float64")


def _load_scalers():
    """Load joblib scalers for Qiskit params and Hamiltonian outputs."""
    with open("X_names", "r") as f:
        ham_names = f.read().splitlines()
    qiskit_names = np.load("y_columns.npy", allow_pickle=True).astype(str).tolist()

    qiskit_scalers = {n: joblib.load(f"{SCALER_DIR}/scaler_y_linear_{n}.save")
                      for n in qiskit_names}
    ham_scalers    = {n: joblib.load(f"{SCALER_DIR}/scaler_X_linear_{n}.save")
                      for n in ham_names}
    return qiskit_names, ham_names, qiskit_scalers, ham_scalers


def _make_scale_funcs(qiskit_names, ham_names, qiskit_scalers, ham_scalers):
    """Return (scale_qiskit, unscale_Hamiltonian) closures."""
    def scale_qiskit(X_real):
        out = X_real.copy()
        for j, n in enumerate(qiskit_names):
            out[:, j] = qiskit_scalers[n].transform(out[:, j:j+1]).ravel()
        return out

    def unscale_Hamiltonian(y_scaled):
        out = y_scaled.copy()
        for j, n in enumerate(ham_names):
            out[:, j] = ham_scalers[n].inverse_transform(out[:, j:j+1]).ravel()
        return out

    return scale_qiskit, unscale_Hamiltonian


def _nearest_train_distance(points_scaled, X_train_scaled):
    """Euclidean NN distance in scaled [0,1] param space."""
    diff = points_scaled[:, None, :] - X_train_scaled[None, :, :]
    d    = np.sqrt((diff ** 2).sum(axis=-1))
    return d.min(axis=1)


# ──────────────────────────────────────────────────────────────────────
# Plot 1: Surrogate vs baselines with density panel
#         (the "last saved plot" — purple oracle REMOVED)
# ──────────────────────────────────────────────────────────────────────
def plot_surrogate_vs_baselines(X_train, y_train, scale_qiskit,
                                unscale_Hamiltonian, out_dir="plots"):
    """
    Reproduce the two-panel figure from the notebook
    (ansys_stress_test_surr_vs_nn_with_density.pdf) using the paper
    palette and WITHOUT the "Nearest training H (oracle)" series.
    """
    CSV = "MLP_Ansys_results_random_stress_test.csv"
    if not os.path.isfile(CSV):
        print(f"  ⚠ Skipping surrogate-vs-baselines: {CSV} not found")
        return

    df_H = pd.read_csv(CSV)
    print(f"  Loaded {len(df_H)} Ansys-validated points from {CSV}")

    # --- NN-distance of each noisy point to the training set ----------
    param_cols = [
        "design_options.connection_pads.readout.claw_length",
        "design_options.connection_pads.readout.ground_spacing",
        "design_options.cross_length",
    ]
    df_params_unscaled = df_H[param_cols].to_numpy(dtype="float64")
    df_params_scaled   = scale_qiskit(df_params_unscaled)
    X_train_np         = np.asarray(X_train).astype("float64")
    df_H["nn_dist"]    = _nearest_train_distance(df_params_scaled, X_train_np)

    # --- Reconstruct Ansys Hamiltonian values -------------------------
    def _recover_ansys(target, pd_target, surr, pd_surr_reported):
        pd_t       = pd_target / 100.0
        cand_plus  = target / (1.0 + pd_t)
        cand_minus = target / (1.0 - pd_t)
        pds_plus   = 100.0 * np.abs(surr - cand_plus)  / np.abs(cand_plus)
        pds_minus  = 100.0 * np.abs(surr - cand_minus) / np.abs(cand_minus)
        use_plus   = (np.abs(pds_plus - pd_surr_reported)
                      < np.abs(pds_minus - pd_surr_reported))
        return np.where(use_plus, cand_plus, cand_minus)

    ansys_fq = _recover_ansys(
        df_H["target_qubit_frequency_GHz"].to_numpy(),
        df_H["pd_fq_target_vs_ansys"].to_numpy(),
        df_H["surrogate_pred_qubit_frequency_GHz"].to_numpy(),
        df_H["pd_fq_surrogate_vs_ansys"].to_numpy(),
    )
    ansys_alpha = _recover_ansys(
        df_H["target_anharmonicity_MHz"].to_numpy(),
        df_H["pd_alpha_target_vs_ansys"].to_numpy(),
        df_H["surrogate_pred_anharmonicity_MHz"].to_numpy(),
        df_H["pd_alpha_surrogate_vs_ansys"].to_numpy(),
    )
    ansys_vec = np.column_stack([ansys_fq, ansys_alpha])

    # --- Classical baselines ------------------------------------------
    y_train_np = np.asarray(y_train).astype("float64")
    nn1 = NearestNeighbors(n_neighbors=1).fit(X_train_np)
    nnK = NearestNeighbors(n_neighbors=5).fit(X_train_np)

    _, idx_1nn = nn1.kneighbors(df_params_scaled)
    pred_1nn_scaled = y_train_np[idx_1nn[:, 0]]

    dK, idxK = nnK.kneighbors(df_params_scaled)
    wK  = 1.0 / (dK + 1e-12)
    wK /= wK.sum(axis=1, keepdims=True)
    pred_knn_scaled = np.zeros((len(df_H), 2))
    for c in range(2):
        pred_knn_scaled[:, c] = (wK * y_train_np[idxK, c]).sum(axis=1)

    pred_1nn = unscale_Hamiltonian(pred_1nn_scaled)
    pred_knn = unscale_Hamiltonian(pred_knn_scaled)

    # --- Score --------------------------------------------------------
    def _mean_pct_vs_ansys(pred):
        pe = 100.0 * np.abs(pred - ansys_vec) / (np.abs(ansys_vec) + 1e-15)
        return pe.mean(axis=1)

    err_surr = 0.5 * (
        df_H["pd_fq_surrogate_vs_ansys"].abs().to_numpy()
      + df_H["pd_alpha_surrogate_vs_ansys"].abs().to_numpy()
    )
    err_1nn = _mean_pct_vs_ansys(pred_1nn)
    err_knn = _mean_pct_vs_ansys(pred_knn)

    # --- Plot ---------------------------------------------------------
    fig, (ax, ax_d) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    nn       = df_H["nn_dist"].to_numpy()
    order    = np.argsort(nn)
    nn_sorted = nn[order]

    ROLL_W = 15

    def _rolling_median(y_sorted, w):
        s = pd.Series(y_sorted)
        return s.rolling(window=w, center=True,
                         min_periods=max(3, w // 2)).median().to_numpy()

    # "Nearest training H (oracle)" is intentionally REMOVED here.
    series = [
        ("Surrogate",                GREEN,  "o", 3.0),
        ("1-NN (nearest in params)", ORANGE, "s", 2.0),
        (r"$k$-NN ($k$=5)",         PURPLE, "D", 2.0),
    ]

    # Scatter
    for label, color, marker, _ in series:
        y = {"Surrogate": err_surr,
             "1-NN (nearest in params)": err_1nn,
             r"$k$-NN ($k$=5)": err_knn}[label]
        ax.scatter(nn, y, s=42, alpha=0.45, color=color, marker=marker,
                   edgecolors="none", label=label)

    # Rolling-median trend line
    for label, color, marker, lw in series:
        y = {"Surrogate": err_surr,
             "1-NN (nearest in params)": err_1nn,
             r"$k$-NN ($k$=5)": err_knn}[label]
        y_sorted = y[order]
        y_roll   = _rolling_median(y_sorted, ROLL_W)
        ax.plot(nn_sorted, y_roll, "-", color=color, linewidth=lw, alpha=0.9)

    ax.set_ylabel("Mean % Error vs Ansys  (avg over $f_q$, anharmonicity)")
    ax.set_title("Surrogate vs Classical Interpolation, scored against Ansys HFSS\n"
                 f"(lines = rolling median, window = {ROLL_W} points)")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    # Bottom: NN-distance density
    ax_d.hist(nn, bins=20, color=GREEN_LIGHT, alpha=0.85,
              edgecolor=GREEN_DARK, linewidth=0.8)
    for xv in nn:
        ax_d.plot([xv, xv], [-0.4, -0.1], color=TEXT_DIM,
                  linewidth=0.8, alpha=0.6, clip_on=False)
    ax_d.set_xlabel("Euclidean distance to nearest training point "
                    "(scaled [0,1] param space)")
    ax_d.set_ylabel("# Ansys points")
    ax_d.grid(True, axis="y", alpha=0.3)
    ax_d.set_ylim(bottom=0)

    plt.tight_layout()
    outpath = os.path.join(out_dir,
                           "ansys_stress_test_surr_vs_nn_with_density.pdf")
    plt.savefig(outpath)
    print(f"  ✓ Saved {outpath}")
    plt.close(fig)

    # --- Numeric summary (without oracle) -----------------------------
    print()
    n_bins = 5
    q_edges = np.quantile(nn, np.linspace(0, 1, n_bins + 1))
    print(f'{"NN dist bin":>20s} {"N":>5s} {"Surr":>8s} '
          f'{"1-NN":>8s} {"k-NN":>8s}')
    print("-" * 52)
    for k in range(n_bins):
        lo, hi = q_edges[k], q_edges[k + 1]
        in_bin = (nn >= lo) & (nn <= hi if k == n_bins - 1 else nn < hi)
        if not in_bin.sum():
            continue
        label = f"[{lo:.3f}, {hi:.3f}]"
        print(f'{label:>20s} {in_bin.sum():5d} '
              f'{np.median(err_surr[in_bin]):8.2f} '
              f'{np.median(err_1nn[in_bin]):8.2f} '
              f'{np.median(err_knn[in_bin]):8.2f}')
    print()
    print(f"Overall median vs Ansys:  surrogate={np.median(err_surr):.2f}%  "
          f"1-NN={np.median(err_1nn):.2f}%  k-NN={np.median(err_knn):.2f}%")


# ──────────────────────────────────────────────────────────────────────
# Plot 2: Error vs NN distance (all sigma fractions)
# ──────────────────────────────────────────────────────────────────────
def plot_error_vs_nn_distance(all_results, out_dir="plots"):
    """
    Reproduce surrogate_stress_test_error_vs_nn_distance.pdf using the
    paper palette.
    """
    sigmas = sorted(all_results.keys())
    all_nn  = np.concatenate([all_results[s]["nn_dist"] for s in sigmas])
    all_err = np.concatenate([all_results[s]["mean_pct_per_sample"]
                              for s in sigmas])
    all_sig = np.concatenate([np.full(len(all_results[s]["nn_dist"]), s)
                              for s in sigmas])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter coloured by sigma
    n_sig = len(sigmas)
    colors = SIGMA_COLORS[:n_sig] if n_sig <= len(SIGMA_COLORS) else \
             [SIGMA_COLORS[i % len(SIGMA_COLORS)] for i in range(n_sig)]

    for i, s in enumerate(sigmas):
        mask = all_sig == s
        ax.scatter(all_nn[mask], all_err[mask], s=8, alpha=0.25,
                   color=colors[i], label=f"σ = {s*100:.0f}%")

    # Binned trend lines
    n_bins = 12
    bin_edges = np.linspace(all_nn.min(), all_nn.max(), n_bins + 1)
    centers, medians, means = [], [], []
    for k in range(n_bins):
        in_bin = (all_nn >= bin_edges[k]) & (all_nn < bin_edges[k + 1])
        if in_bin.sum() < 3:
            continue
        centers.append(0.5 * (bin_edges[k] + bin_edges[k + 1]))
        medians.append(np.median(all_err[in_bin]))
        means.append(all_err[in_bin].mean())

    ax.plot(centers, medians, "o-", color=TEXT_MAIN,  linewidth=2,
            markersize=6, label="Binned median")
    ax.plot(centers, means,   "s--", color=ORANGE_DARK, linewidth=2,
            markersize=6, label="Binned mean")

    ax.set_xlabel("Euclidean distance to nearest training point "
                  "(scaled [0,1] space)")
    ax.set_ylabel("Mean % Error in Predicted Hamiltonian Parameters")
    ax.set_title("Surrogate Stress Test: Error vs Distance to "
                 "Nearest Training Point")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    outpath = os.path.join(out_dir,
                           "surrogate_stress_test_error_vs_nn_distance.pdf")
    plt.savefig(outpath)
    print(f"  ✓ Saved {outpath}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Plot 3: Per-sigma histogram of % error
# ──────────────────────────────────────────────────────────────────────
def plot_pct_error_histograms(all_results, out_dir="plots"):
    """
    Reproduce pctErr{sigma}.png with paper palette colours.
    frequency  → ORANGE
    anharmonicity → GREEN
    """
    for sigma_frac, r in all_results.items():
        pct_errors = r["pct_errors"]
        fig, ax = plt.subplots()
        bins = np.linspace(pct_errors[:, 1].min(),
                           pct_errors[:, 1].max(), 50)

        ax.hist(pct_errors[:, 0], color=ORANGE, histtype="step",
                bins=bins, label="frequency", linewidth=1.5)
        ax.hist(pct_errors[:, 1], color=GREEN, histtype="step",
                bins=bins, label="anharmonicity", linewidth=1.5)

        ax.set_title(r"% error (pred vs target), $\sigma$ = "
                     f"{np.round(sigma_frac*100):.0f}%")
        ax.set_ylabel("Counts")
        ax.set_xlabel("% error")
        ax.legend()
        plt.tight_layout()

        outpath = os.path.join(
            out_dir, f"pctErr{np.round(sigma_frac*100):.0f}.png")
        plt.savefig(outpath, dpi=200)
        print(f"  ✓ Saved {outpath}")
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────
def main():
    _apply_paper_rcparams()
    os.makedirs("plots", exist_ok=True)

    # --- Load data & scalers -----------------------------------------
    print("Loading training data & scalers …")
    X_train, y_train = _load_training_data()
    (qiskit_names, ham_names,
     qiskit_scalers, ham_scalers) = _load_scalers()
    scale_qiskit, unscale_Hamiltonian = _make_scale_funcs(
        qiskit_names, ham_names, qiskit_scalers, ham_scalers)
    print(f"  Training set: {X_train.shape[0]} samples "
          f"({X_train.shape[1]} param dims → {y_train.shape[1]} H dims)")

    # --- Plot 1: Surrogate vs baselines (FINAL plot) ------------------
    print("\n[1/3] Surrogate vs baselines (density) …")
    plot_surrogate_vs_baselines(X_train, y_train, scale_qiskit,
                                unscale_Hamiltonian)

    # --- Plots 2 & 3 need `all_results` from the notebook ------------
    # Try to load a cached copy if it exists; otherwise skip gracefully.
    CACHE = "all_results_cache.npz"
    if os.path.isfile(CACHE):
        print(f"\n  Loading cached stress-test results from {CACHE}")
        npz = np.load(CACHE, allow_pickle=True)
        all_results = npz["all_results"].item()

        print("\n[2/3] Error vs NN-distance scatter …")
        plot_error_vs_nn_distance(all_results)

        print("\n[3/3] Per-sigma % error histograms …")
        plot_pct_error_histograms(all_results)
    else:
        print(f"\n  ⚠ {CACHE} not found — skipping per-sigma plots.")
        print("    To create the cache, add the following line at the end")
        print("    of the stress-test loop cell in the notebook:")
        print(f'      np.savez("{CACHE}", all_results=all_results)')

    print("\nDone.")


if __name__ == "__main__":
    main()
