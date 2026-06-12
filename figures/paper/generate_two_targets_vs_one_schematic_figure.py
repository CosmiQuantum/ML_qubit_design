"""Schematic: why the two-target (f_q, alpha) surrogate-defined loss kept
claw_length inside the valid region while the single-target (E_C) loss does not.

Plot 1 (two_targets_schematic): the two surrogate output heads define two
nearly-coincident constraint bands in design space; the inverse model must
satisfy both, so it is pinned near the training cloud (band tilts exaggerated
for visibility).
Plot 2 (one_target_schematic): the single E_C head is satisfied along an
entire iso-E_C line, half of which lies in the invalid claw > cross region.

Training points are the real SQuADDS TransmonCross designs loaded from the
experiment metadata.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

GREEN = "#3D8B3D"
ORANGE = "#E87A00"
PURPLE = "#7B68AE"
TEXT_MAIN = "#222222"
TEXT_DIM = "#555555"
INVALID = "#C0392B"

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#666666",
    "axes.labelcolor": TEXT_MAIN,
    "xtick.color": TEXT_MAIN,
    "ytick.color": TEXT_MAIN,
})

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "model_predict_qubit_TransmonCross_Hamiltonian_params"
PLOTS_DIR = EXPERIMENT_DIR / "plots"

## load the training designs (um) from the experiment metadata
with open(EXPERIMENT_DIR / "metadata" / "qubit-TransmonCross-EC.json") as f:
    records = json.load(f)

claw = np.array([float(r["design"]["design_options"]["connection_pads"]["readout"]["claw_length"].replace("um", ""))
                 for r in records])
cross = np.array([float(r["design"]["design_options"]["cross_length"].replace("um", ""))
                  for r in records])

LIMS = [60, 440]

## anchor: a training-like design; bands/line pass through it
claw0, cross0 = 115.0, 190.0


def make_base_axes():
    """Shared backdrop: training cloud, claw = cross boundary, invalid region."""
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    ax.scatter(cross, claw, s=9, c="0.78", alpha=0.6, zorder=2,
               label="SQuADDS training designs")
    ax.plot(LIMS, LIMS, ls="--", lw=1.2, color=TEXT_DIM, zorder=3)
    ax.fill_between(LIMS, LIMS, 440, color=INVALID, alpha=0.07, zorder=1)
    ax.text(68, 432, "invalid\n(claw > cross)", color=INVALID, fontsize=10,
            ha="left", va="top", style="italic")
    ax.set_xlim(*LIMS)
    ax.set_ylim(60, 440)
    ax.set_xlabel(r"cross_length ($\mu$m)", fontsize=11)
    ax.set_ylabel(r"claw_length ($\mu$m)", fontsize=11)
    ax.text(398, 372, "claw = cross", color=TEXT_DIM, fontsize=9,
            rotation=45, ha="center", va="center")
    return fig, ax


## ---------- plot 1: two-target loss (f_q and alpha) ----------
fig, ax = make_base_axes()

claw_axis = np.linspace(70, 400, 200)
half_w = 9.0

## two constraint bands with small opposing tilts (exaggerated)
for slope, color in [(-0.12, ORANGE), (+0.12, PURPLE)]:
    center = cross0 + slope * (claw_axis - claw0)
    ax.fill_betweenx(claw_axis, center - half_w, center + half_w,
                     color=color, alpha=0.30, zorder=3)
ax.text(cross0 - 0.12 * (360 - claw0) - 13, 360, r"$f_q$ target",
        color=ORANGE, fontsize=11, ha="right")
ax.text(cross0 + 0.12 * (360 - claw0) + 13, 360, r"$\alpha$ target",
        color=PURPLE, fontsize=11, ha="left")

ax.add_patch(Ellipse((cross0, claw0), 36, 58, facecolor=GREEN, edgecolor=GREEN,
                     alpha=0.85, zorder=5))
ax.text(cross0 + 28, claw0, "both\nsatisfied", color=GREEN, fontsize=10,
        va="center", zorder=6,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

ax.set_title(r"two-target loss: $f_q$ and $\alpha$", fontsize=12)
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
fig.text(0.005, 0.008, "schematic — constraint-band tilts exaggerated", fontsize=8,
         color=TEXT_DIM, style="italic")

fig.tight_layout()
out = PLOTS_DIR / "two_targets_schematic"
fig.savefig(f"{out}.png", dpi=300, facecolor="white")
fig.savefig(f"{out}.pdf")
print(f"Written {out}.png and {out}.pdf")
plt.close(fig)

## ---------- plot 2: single-target loss (E_C) ----------
fig, ax = make_base_axes()

## the single-target loss is satisfied along the whole iso-E_C line
ax.plot([cross0, cross0], [70, 400], color=GREEN, lw=5, solid_capstyle="butt",
        zorder=4, alpha=0.85)
ax.text(cross0 + 12, 365, r"$E_C$ target", color=GREEN, fontsize=11,
        zorder=6, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

ax.set_title(r"single-target loss: $E_C$", fontsize=12)
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

fig.tight_layout()
out = PLOTS_DIR / "one_target_schematic"
fig.savefig(f"{out}.png", dpi=300, facecolor="white")
fig.savefig(f"{out}.pdf")
print(f"Written {out}.png and {out}.pdf")
plt.close(fig)
