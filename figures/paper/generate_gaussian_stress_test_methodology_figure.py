#!/usr/bin/env python3
r"""
Generate the stress-test methodology figure.

Same palette family and rendering approach as generate_workflow_mpl.py:
  - matplotlib + mathtext (no LaTeX install needed)
  - orange = physics / qubit accent, green = ML / data action,
    purple = validation / stress-test accent
  - rounded FancyBboxPatch cards with a title bar + body text
  - horizontal three-step pipeline: Sample → Measure → Select
  - bold goal-statement banner at top and punchline strip at bottom

Outputs:
    manuscript_exports/stress_test_methodology.pdf

Usage:
    python3 generate_gaussian_stress_test_methodology_figure.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from _paths import MANUSCRIPT_EXPORTS_DIR

# Use mathtext (built in, ships with matplotlib), NOT full LaTeX
plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

# Color palette (same family as generate_workflow_mpl.py)
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

ARROW         = "#2C7A7B"   # teal, matching the step-card accent

# These three cards are sequential process steps. They deliberately use a
# single TEAL accent (NOT the flowchart blue shades) so the figure does not
# read as if it were re-using the workflow slides' blue color code
# (light blue = Hamiltonian params, dark blue = compute loss). Teal is
# colorblind-safe and is not used anywhere else in the paper, whose accents
# are reserved: blue (workflow), orange (physics / f_q, alpha), green
# (ML / data), purple (validation). A uniform teal border + soft teal title
# bar on a white body keeps the cards clean and avoids any shade-coding.
STEP_FILL   = "#FFFFFF"
STEP_STROKE = "#2C7A7B"   # teal border + title bar tint
STEP_ACCENT = "#1D4E4F"   # dark teal title text
## card bodies read like little algorithms: a lead line at indent 0
## and its sub-steps indented one level, per reviewer request
STEP_STYLES = [
    dict(title="1. Sample uniformly",
         fill=STEP_FILL, stroke=STEP_STROKE, accent=STEP_ACCENT,
         body=[
             (0, r"Generate 50,000 random design"),
             (0, r"parameter sets:"),
             (1, r"uniform inside the convex hull"),
             (1, r"of the training data"),
             (1, r"(interpolation only)."),
         ]),
    dict(title="2. Measure NN distance",
         fill=STEP_FILL, stroke=STEP_STROKE, accent=STEP_ACCENT,
         body=[
             (0, r"For each random point:"),
             (1, r"compute $d_{\rm NN}$ = Euclidean"),
             (1, r"distance to its nearest training"),
             (1, r"sample in scaled $[0,\,1]$ space."),
         ]),
    dict(title="3. Bin and select",
         fill=STEP_FILL, stroke=STEP_STROKE, accent=STEP_ACCENT,
         body=[
             (0, r"Split the $d_{\rm NN}$ range into 10"),
             (0, r"equal-width bins (close $\rightarrow$ far):"),
             (1, r"draw 10 random points per bin"),
             (1, r"$\rightarrow$ 100 total for EM validation."),
         ]),
]

# Layout
FIG_W_IN = 12
FIG_H_IN = 7.1

# Canvas coordinate system (arbitrary units)
W = 100
H = 60

# Headline banner (top), goal statement strip.
# Widened near the full canvas width so the long italic subtitle fits.
BANNER_X, BANNER_Y = 0.5, 43
BANNER_W, BANNER_H = 99, 13

# Three step cards. Cards widened from 28 to 30 each, gaps set to 3.5,
# so the larger card titles fit and the connecting arrows are clearly
# visible in the space between cards.
CARD_W = 30
CARD_H = 24
CARD_Y = 16
CARD_GAP = 3.5
CARDS_TOTAL_W = 3 * CARD_W + 2 * CARD_GAP
CARDS_LEFT = (W - CARDS_TOTAL_W) / 2

# Punchline strip (bottom). Widened to match the banner.
PUNCH_X, PUNCH_Y = 0.5, 4
PUNCH_W, PUNCH_H = 99, 8

# Build figure
fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")

# Goal banner (neutral card)
banner = FancyBboxPatch(
    (BANNER_X, BANNER_Y), BANNER_W, BANNER_H,
    boxstyle="round,pad=0,rounding_size=1.2",
    linewidth=1.5,
    edgecolor=NEUTRAL_STROKE,
    facecolor=NEUTRAL_FILL,
)
ax.add_patch(banner)

## goal stated as a bold sentence (same sans font as everything else),
## not a question and no separate mathtext "Goal" label
ax.text(
    W / 2, BANNER_Y + BANNER_H * 0.78,
    "Quantify how well the forward surrogate generalizes to",
    ha="center", va="center",
    fontsize=16, fontweight="bold", color=TEXT_MAIN,
)
ax.text(
    W / 2, BANNER_Y + BANNER_H * 0.52,
    "design parameters it has never seen before",
    ha="center", va="center",
    fontsize=16, fontweight="bold", color=TEXT_MAIN,
)
ax.text(
    W / 2, BANNER_Y + BANNER_H * 0.20,
    r"Probe the in-between regions of SQuADDS by sampling uniformly inside "
    r"the training cloud and binning by proximity.",
    ha="center", va="center",
    fontsize=12, fontstyle="italic", color=TEXT_DIM,
)

# Three step cards
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

    # Accent bar on top of the card (title strip).
    # Sized to fit a 15 pt bold title with a bit of vertical padding.
    title_bar_h = 4.6
    title_bar = Rectangle(
        (x, y + CARD_H - title_bar_h), CARD_W, title_bar_h,
        linewidth=0,
        facecolor=style["stroke"],
        alpha=0.18,
    )
    ax.add_patch(title_bar)

    # Title text (dark accent color). 15 pt fits "2. Measure NN distance"
    # (the longest title) inside a 30unit wide card with margin to spare.
    ax.text(
        x + CARD_W / 2, y + CARD_H - title_bar_h / 2,
        style["title"],
        ha="center", va="center",
        fontsize=15, fontweight="bold", color=style["accent"],
    )

    # Body text leftaligned, one line per entry, indented by level.
    # 11 pt keeps the math formula from overflowing card width.
    body_top = y + CARD_H - title_bar_h - 2.0
    line_dy = 2.5
    indent_dx = 2.2
    for j, (indent, line) in enumerate(style["body"]):
        ax.text(
            x + 1.6 + indent * indent_dx, body_top - j * line_dy,
            line,
            ha="left", va="top",
            fontsize=11, color=TEXT_MAIN,
        )

# Arrows between cards
# Larger arrowhead and thicker line to stand out between the cards. With
# CARD_GAP=3.5 the arrow has roughly 3.3 units of horizontal run.
for i in range(2):
    x_from = card_centers_x[i] + CARD_W / 2
    x_to   = card_centers_x[i + 1] - CARD_W / 2
    y_mid  = CARD_Y + CARD_H / 2
    arrow = FancyArrowPatch(
        (x_from + 0.1, y_mid),
        (x_to - 0.1, y_mid),
        arrowstyle="-|>",
        mutation_scale=26,
        linewidth=3.0,
        color=ARROW,
    )
    ax.add_patch(arrow)

# Punchline strip at the bottom
punch = FancyBboxPatch(
    (PUNCH_X, PUNCH_Y), PUNCH_W, PUNCH_H,
    boxstyle="round,pad=0,rounding_size=1.1",
    linewidth=1.8,
    edgecolor=NEUTRAL_STROKE,
    facecolor=NEUTRAL_FILL,
)
ax.add_patch(punch)

# Two-line punchline, vertically centered in the strip.
ax.text(
    PUNCH_X + PUNCH_W / 2, PUNCH_Y + PUNCH_H * 0.65,
    r"Surrogate evaluates all 50,000 samples in seconds.",
    ha="center", va="center",
    fontsize=13, fontweight="bold", color=TEXT_MAIN,
)
ax.text(
    PUNCH_X + PUNCH_W / 2, PUNCH_Y + PUNCH_H * 0.30,
    r"100 selected across distance bins sent to the EM solver for validation.",
    ha="center", va="center",
    fontsize=13, fontweight="bold", color=TEXT_MAIN,
)

# Save
stress_test_pdf = MANUSCRIPT_EXPORTS_DIR / "stress_test_methodology.pdf"

plt.savefig(stress_test_pdf, bbox_inches="tight", pad_inches=0.15)
print(f"Written {stress_test_pdf}")
