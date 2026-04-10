#!/usr/bin/env python3
r"""
Generate the Gaussian stress-test methodology figure.

Same palette family and rendering approach as generate_workflow_mpl.py:
  - matplotlib + mathtext (no LaTeX install needed)
  - orange = physics / qubit accent, green = ML / data action,
    purple = validation / stress-test accent
  - rounded FancyBboxPatch cards with a title bar + body text
  - horizontal three-step pipeline: Pick Seeds -> Add Noise -> Vary Sigma
  - headline banner at top and punchline strip at bottom

Outputs:
    paper_plots/stress_test_methodology.pdf
    paper_plots/stress_test_methodology.svg

Usage:
    python3 generate_stress_test_methodology.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# Use mathtext (built in, ships with matplotlib) — NOT full LaTeX
plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

# ---- Color palette (same family as generate_workflow_mpl.py) ------------
GREEN         = "#3D8B3D"
GREEN_LIGHT   = "#E8F5E8"
GREEN_DARK    = "#2E6B2E"

ORANGE        = "#E87A00"
ORANGE_LIGHT  = "#FFF4E6"
ORANGE_DARK   = "#A85600"

PURPLE        = "#7B68AE"
PURPLE_LIGHT  = "#E8E4F0"
PURPLE_DARK   = "#4A3D78"

NEUTRAL_FILL  = "#F5F5F5"
NEUTRAL_STROKE = "#999999"

TEXT_MAIN     = "#222222"
TEXT_DIM      = "#555555"

ARROW         = "#555555"

# Step cards rotate through physics -> ml -> validation accents, so each
# step feels visually distinct but all three live in the same family.
STEP_STYLES = [
    dict(title="1. Pick seeds",
         fill=ORANGE_LIGHT, stroke=ORANGE, accent=ORANGE_DARK,
         body=[
             r"Choose 30 held-out test samples",
             r"from SQuADDS and treat each one",
             r"as the center of a Gaussian ball",
             r"in Qiskit Metal parameter space.",
         ]),
    dict(title="2. Add Gaussian noise",
         fill=GREEN_LIGHT, stroke=GREEN, accent=GREEN_DARK,
         body=[
             r"For each parameter $i$:",
             r"$p_i^{\rm new} = p_i^{\rm orig} + \mathcal{N}(0,\ \sigma_i)$",
             r"with $\sigma_i = f \cdot \mathrm{range}(p_i)$.",
             r"Reject samples that leave the box",
             r"or violate layout constraints.",
         ]),
    dict(title="3. Sweep sigma",
         fill=PURPLE_LIGHT, stroke=PURPLE, accent=PURPLE_DARK,
         body=[
             r"Sweep $f$ across",
             r"$\{1\%,\ 2\%,\ 5\%,\ 10\%,\ 20\%\}$",
             r"of each parameter's range.",
             r"Draw 6,000 samples per $\sigma$",
             r"$\rightarrow$ 30,000 total.",
         ]),
]

# ---- Layout --------------------------------------------------------------
FIG_W_IN = 12
FIG_H_IN = 7.8

# Canvas coordinate system (arbitrary units)
W = 100
H = 66

# Title (at the very top)
TITLE_Y = 62.5

# Headline banner (top) — "Goal" strip
BANNER_X, BANNER_Y = 6, 44
BANNER_W, BANNER_H = 88, 13

# Three step cards
CARD_W = 28
CARD_H = 24
CARD_Y = 16
CARD_GAP = 3.5
CARDS_TOTAL_W = 3 * CARD_W + 2 * CARD_GAP
CARDS_LEFT = (W - CARDS_TOTAL_W) / 2

# Punchline strip (bottom)
PUNCH_X, PUNCH_Y = 6, 4
PUNCH_W, PUNCH_H = 88, 8

# ---- Build figure --------------------------------------------------------
fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")

# ---- Title above the banner ---------------------------------------------
ax.text(
    W / 2, TITLE_Y,
    r"Gaussian stress-test methodology",
    ha="center", va="center",
    fontsize=15, fontweight="bold", color=TEXT_MAIN,
)

# ---- Goal banner (neutral card with green label) ------------------------
banner = FancyBboxPatch(
    (BANNER_X, BANNER_Y), BANNER_W, BANNER_H,
    boxstyle="round,pad=0,rounding_size=1.2",
    linewidth=1.5,
    edgecolor=NEUTRAL_STROKE,
    facecolor=NEUTRAL_FILL,
)
ax.add_patch(banner)

# Bold "Goal:" label + wrapped question + italic subtitle.
ax.text(
    W / 2, BANNER_Y + BANNER_H * 0.78,
    r"$\bf{Goal:}$  How well does the inverse + surrogate pipeline generalize to",
    ha="center", va="center",
    fontsize=11, color=TEXT_MAIN,
)
ax.text(
    W / 2, BANNER_Y + BANNER_H * 0.52,
    r"Qiskit Metal parameters it has never seen before",
    ha="center", va="center",
    fontsize=11, color=TEXT_MAIN,
)
ax.text(
    W / 2, BANNER_Y + BANNER_H * 0.20,
    r"Probe the in-between regions of SQuADDS by perturbing held-out test "
    r"points with controllable Gaussian noise.",
    ha="center", va="center",
    fontsize=9.5, fontstyle="italic", color=TEXT_DIM,
)

# ---- Three step cards ----------------------------------------------------
card_centers_x = []
for i, style in enumerate(STEP_STYLES):
    x = CARDS_LEFT + i * (CARD_W + CARD_GAP)
    y = CARD_Y
    card_centers_x.append(x + CARD_W / 2)

    # Card body
    card = FancyBboxPatch(
        (x, y), CARD_W, CARD_H,
        boxstyle="round,pad=0,rounding_size=1.1",
        linewidth=1.8,
        edgecolor=style["stroke"],
        facecolor=style["fill"],
    )
    ax.add_patch(card)

    # Accent bar on top of the card (title strip)
    title_bar_h = 4.2
    title_bar = Rectangle(
        (x, y + CARD_H - title_bar_h), CARD_W, title_bar_h,
        linewidth=0,
        facecolor=style["stroke"],
        alpha=0.18,
    )
    ax.add_patch(title_bar)

    # Title text (dark accent color)
    ax.text(
        x + CARD_W / 2, y + CARD_H - title_bar_h / 2,
        style["title"],
        ha="center", va="center",
        fontsize=12, fontweight="bold", color=style["accent"],
    )

    # Body text — left-aligned, one line per entry
    body_top = y + CARD_H - title_bar_h - 2.0
    line_dy = 2.3
    for j, line in enumerate(style["body"]):
        ax.text(
            x + 1.6, body_top - j * line_dy,
            line,
            ha="left", va="top",
            fontsize=9.5, color=TEXT_MAIN,
        )

# ---- Arrows between cards ------------------------------------------------
for i in range(2):
    x_from = card_centers_x[i] + CARD_W / 2
    x_to   = card_centers_x[i + 1] - CARD_W / 2
    y_mid  = CARD_Y + CARD_H / 2
    arrow = FancyArrowPatch(
        (x_from + 0.3, y_mid),
        (x_to - 0.3, y_mid),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.0,
        color=ARROW,
    )
    ax.add_patch(arrow)

# ---- Punchline strip at the bottom --------------------------------------
punch = FancyBboxPatch(
    (PUNCH_X, PUNCH_Y), PUNCH_W, PUNCH_H,
    boxstyle="round,pad=0,rounding_size=1.1",
    linewidth=1.8,
    edgecolor=ORANGE,
    facecolor=ORANGE_LIGHT,
)
ax.add_patch(punch)

ax.text(
    PUNCH_X + PUNCH_W / 2, PUNCH_Y + PUNCH_H / 2,
    r"Surrogate evaluates all 30,000 samples in seconds "
    r"($\sim$5000 Ansys-hours equivalent).",
    ha="center", va="center",
    fontsize=11.5, fontweight="bold", color=ORANGE_DARK,
)

# ---- Save ---------------------------------------------------------------
plt.savefig("paper_plots/stress_test_methodology.pdf", bbox_inches="tight", pad_inches=0.15)
plt.savefig("paper_plots/stress_test_methodology.svg", bbox_inches="tight", pad_inches=0.15)
print("Written stress_test_methodology.pdf and stress_test_methodology.svg")
