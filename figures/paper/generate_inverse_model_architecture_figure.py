#!/usr/bin/env python3
r"""
Generate a single-column tandem-pipeline architecture diagram for the
inverse model.

Pipeline:
    Target Hamiltonian (f_q, alpha)
        -> Inverse MLP (2 -> 64 -> 3, LeakyReLU, 387 trainable params)
        -> Predicted geometry (ell_claw, s_ground, ell_cross)
        -> Frozen forward surrogate (3 -> 736 -> 2, LeakyReLU, 4,418 params)
        -> Reconstructed Hamiltonian (f_q_hat, alpha_hat)
    Loss is the average absolute difference H_target - H_hat; only the
    inverse-MLP weights are updated.

Combines the high-level information from
    plots/model_architecture_paper_theme_combined.pdf
(param-count header, "Training" dashed bracket, feedback arrow) with a
concrete neuron-level visualization.

Outputs:
    manuscript_exports/inverse_model_architecture.{pdf,png}
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _paths import MANUSCRIPT_EXPORTS_DIR

plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

# Shared with the overview-figure palette.
FROST           = "#D6E5EE"
PALE_ICE        = "#B0CCDE"
DUSTY_BLUE      = "#8AABC8"
FROST_DARK      = "#567A90"
PALE_ICE_DARK   = "#3F6F8B"
DUSTY_BLUE_DARK = "#17384F"
FROZEN_FILL     = "#C8D6DE"
FROZEN_EDGE     = "#6E8390"
TEXT_MAIN       = "#222222"
TEXT_DIM        = "#555555"
CONNECTION      = "#8AABC8"
LOSS_COLOR      = "#17384F"
TRAINING_EDGE   = "#3F6F8B"

FIG_W_IN = 3.4
FIG_H_IN = 3.25

# Five-column layout (axis coordinates).
INPUT_X        = 0.05
INV_HIDDEN_X   = 0.26
GEOM_X         = 0.50
SURR_HIDDEN_X  = 0.75
OUTPUT_X       = 0.95

# Y-zones (top to bottom).
HEADER_Y       = 1.06
CAPTION_Y      = 0.95
TRAIN_TOP_Y    = 0.88
TRAIN_BOT_Y    = 0.18
SUBCAPTION_Y   = 0.08      # sub-captions sit just below the training box
LOSS_TEXT_Y    = -0.05     # loss formula sits tight under the sub-captions

# Visible neurons per hidden column (ellipsis stands in for the rest).
INV_HIDDEN_Y    = [0.81, 0.71, 0.61, 0.45, 0.35, 0.25]
INV_ELLIPSIS_Y  = 0.53
SURR_HIDDEN_Y   = [0.81, 0.71, 0.61, 0.45, 0.35, 0.25]
SURR_ELLIPSIS_Y = 0.53
INPUT_Y         = [0.60, 0.40]
GEOM_Y          = [0.66, 0.50, 0.34]
OUTPUT_Y        = [0.60, 0.40]

# Bigger nodes and fonts so labels render close to body-text size when the
# PDF is included at \linewidth in single-column layout.
NODE_SIZE      = 50
IO_NODE_SIZE   = 100
LABEL_FS       = 9.4
GEOM_LABEL_FS  = 8.4
CAPTION_FS     = 9.4
SUBCAPTION_FS  = 8.4
HEADER_FS      = 8.0
LOSS_FS        = 9.4

fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
ax.set_xlim(0, 1)
ax.set_ylim(-0.11, 1.12)
ax.axis("off")

# ---- header strip: parameter counts -----------------------------------------
ax.text(0.50, HEADER_Y,
        "387 trainable  +  4,418 frozen  =  4,805 total",
        ha="center", va="center",
        fontsize=HEADER_FS, color=TEXT_DIM)

# ---- column captions --------------------------------------------------------
ax.text(INPUT_X,       CAPTION_Y, "Target",         ha="center", va="center",
        fontsize=CAPTION_FS, fontweight="bold", color=FROST_DARK)
ax.text(INV_HIDDEN_X,  CAPTION_Y, "Inverse",        ha="center", va="center",
        fontsize=CAPTION_FS, fontweight="bold", color=PALE_ICE_DARK)
ax.text(GEOM_X,        CAPTION_Y, "Geometry",       ha="center", va="center",
        fontsize=CAPTION_FS, fontweight="bold", color=PALE_ICE_DARK)
ax.text(SURR_HIDDEN_X, CAPTION_Y, "Surrogate",      ha="center", va="center",
        fontsize=CAPTION_FS, fontweight="bold", color=FROZEN_EDGE)
ax.text(OUTPUT_X,      CAPTION_Y, "Output",         ha="center", va="center",
        fontsize=CAPTION_FS, fontweight="bold", color=DUSTY_BLUE_DARK)

# ---- "Training" dashed bracket around inverse + geometry + surrogate -------
train_x0 = INV_HIDDEN_X - 0.075
train_x1 = SURR_HIDDEN_X + 0.075
training_box = FancyBboxPatch(
    (train_x0, TRAIN_BOT_Y),
    train_x1 - train_x0,
    TRAIN_TOP_Y - TRAIN_BOT_Y,
    boxstyle="round,pad=0,rounding_size=0.018",
    linewidth=0.95,
    edgecolor=TRAINING_EDGE,
    facecolor="none",
    linestyle=(0, (3, 2.5)),
    zorder=1,
)
ax.add_patch(training_box)
ax.text(train_x1 - 0.02, TRAIN_TOP_Y, " Training ",
        ha="right", va="center",
        fontsize=SUBCAPTION_FS + 0.4, fontstyle="italic",
        color=TRAINING_EDGE,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6),
        zorder=2)


# ---- helpers ----------------------------------------------------------------
def draw_connections(x1, x2, ys1, ys2, *, color=CONNECTION, lw=0.30, alpha=0.55):
    for ya in ys1:
        for yb in ys2:
            ax.plot([x1, x2], [ya, yb], color=color, linewidth=lw,
                    alpha=alpha, zorder=2)


def draw_column(xs_y, x, *, size, fill, edge, lw=1.0):
    ax.scatter([x] * len(xs_y), xs_y, s=size, c=fill,
               edgecolors=edge, linewidths=lw, zorder=4)


def text_label(x, y, text, *, ha, fontsize=LABEL_FS, color=TEXT_MAIN,
               weight="normal", bbox=None):
    ax.text(x, y, text, ha=ha, va="center",
            fontsize=fontsize, color=color, fontweight=weight,
            bbox=bbox, zorder=5)


# ---- connections (drawn before nodes) ---------------------------------------
draw_connections(INPUT_X,       INV_HIDDEN_X,  INPUT_Y,       INV_HIDDEN_Y)
draw_connections(INV_HIDDEN_X,  GEOM_X,        INV_HIDDEN_Y,  GEOM_Y)
draw_connections(GEOM_X,        SURR_HIDDEN_X, GEOM_Y,        SURR_HIDDEN_Y,
                 color=FROZEN_EDGE, alpha=0.40)
draw_connections(SURR_HIDDEN_X, OUTPUT_X,      SURR_HIDDEN_Y, OUTPUT_Y,
                 color=FROZEN_EDGE, alpha=0.40)

# ---- nodes ------------------------------------------------------------------
draw_column(INPUT_Y,       INPUT_X,       size=IO_NODE_SIZE, fill=FROST,       edge=FROST_DARK,      lw=1.05)
draw_column(INV_HIDDEN_Y,  INV_HIDDEN_X,  size=NODE_SIZE,    fill=PALE_ICE,    edge=PALE_ICE_DARK)
draw_column(GEOM_Y,        GEOM_X,        size=IO_NODE_SIZE, fill=PALE_ICE,    edge=PALE_ICE_DARK,   lw=1.05)
draw_column(SURR_HIDDEN_Y, SURR_HIDDEN_X, size=NODE_SIZE,    fill=FROZEN_FILL, edge=FROZEN_EDGE)
draw_column(OUTPUT_Y,      OUTPUT_X,      size=IO_NODE_SIZE, fill=DUSTY_BLUE,  edge=DUSTY_BLUE_DARK, lw=1.05)

ax.text(INV_HIDDEN_X,  INV_ELLIPSIS_Y,  r"$\vdots$", ha="center", va="center",
        fontsize=12, color=PALE_ICE_DARK, zorder=5)
ax.text(SURR_HIDDEN_X, SURR_ELLIPSIS_Y, r"$\vdots$", ha="center", va="center",
        fontsize=12, color=FROZEN_EDGE, zorder=5)

# ---- node labels ------------------------------------------------------------
for y, lbl in zip(INPUT_Y, [r"$f_q$", r"$\alpha$"]):
    text_label(INPUT_X - 0.035, y, lbl, ha="right")

geom_labels = [r"$\tilde{\ell}_{\mathrm{claw}}$",
               r"$\tilde{s}_{\mathrm{ground}}$",
               r"$\tilde{\ell}_{\mathrm{cross}}$"]
for y, lbl in zip(GEOM_Y, geom_labels):
    text_label(GEOM_X + 0.05, y, lbl, ha="left",
               fontsize=GEOM_LABEL_FS,
               bbox=dict(facecolor="white", edgecolor="none", pad=0.4))

for y, lbl in zip(OUTPUT_Y, [r"$\hat{f}_q$", r"$\hat{\alpha}$"]):
    text_label(OUTPUT_X + 0.035, y, lbl, ha="left")

# ---- sub-captions inside the training box, below the hidden columns --------
ax.text(INV_HIDDEN_X, SUBCAPTION_Y, "64 hidden, LeakyReLU",
        ha="center", va="center",
        fontsize=SUBCAPTION_FS, fontstyle="italic", color=PALE_ICE_DARK)
ax.text(SURR_HIDDEN_X, SUBCAPTION_Y, "736 hidden, frozen",
        ha="center", va="center",
        fontsize=SUBCAPTION_FS, fontstyle="italic", color=FROZEN_EDGE,
        fontweight="bold")

# ---- loss formula, centered tight under the sub-captions --------------------
ax.text(0.50, LOSS_TEXT_Y,
        r"loss $=\ \langle\,|\,H_{\mathrm{target}} - \hat{H}\,|\,\rangle$",
        ha="center", va="center",
        fontsize=LOSS_FS, color=LOSS_COLOR, fontweight="bold")

pdf_path = MANUSCRIPT_EXPORTS_DIR / "inverse_model_architecture.pdf"
png_path = MANUSCRIPT_EXPORTS_DIR / "inverse_model_architecture.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04, facecolor="white")
fig.savefig(png_path, bbox_inches="tight", pad_inches=0.04, dpi=300, facecolor="white")
print(f"wrote {pdf_path}")
print(f"wrote {png_path}")
