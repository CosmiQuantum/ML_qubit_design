#!/usr/bin/env python3
r"""
Generate the end-to-end inverse-design workflow figure.

Uses matplotlib's mathtext renderer so variables like $\omega_q$, $\hat{y}_q$,
$\mathbb{R}^{d_{in}}$ appear as proper typeset math — no LaTeX install needed.

Outputs:
    manuscript_exports/workflow.pdf

Usage:
    python3 generate_inverse_design_workflow_figure.py
"""

import matplotlib

matplotlib.use("Agg")

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

from _paths import MANUSCRIPT_EXPORTS_DIR

# Use mathtext (built in, ships with matplotlib) NOT full LaTeX
plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "cm"       # Computer Modern look for math
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

FLOWCHART_COLOR_SCHEME = os.environ.get("FLOWCHART_COLOR_SCHEME", "blue").strip().lower()
FLOWCHART_COLOR_SCHEME = {"current": "blue", "new": "blue", "old": "legacy", "classic": "legacy"}.get(
    FLOWCHART_COLOR_SCHEME,
    FLOWCHART_COLOR_SCHEME,
)
if FLOWCHART_COLOR_SCHEME == "legacy":
    FROST         = "#FFF4E6"  # Physics targets
    PALE_ICE      = "#E8F5E8"  # ML components
    DUSTY_BLUE    = "#E8E4F0"  # Validation
    FROST_DARK      = "#E87A00"
    PALE_ICE_DARK   = "#3D8B3D"
    DUSTY_BLUE_DARK = "#7B68AE"
else:
    FROST         = "#D6E5EE"  # Physics targets
    PALE_ICE      = "#B0CCDE"  # ML components
    DUSTY_BLUE    = "#8AABC8"  # Validation
    FROST_DARK      = "#567A90"
    PALE_ICE_DARK   = "#3F6F8B"
    DUSTY_BLUE_DARK = "#17384F"

NEUTRAL_FILL  = "#F5F5F5"
NEUTRAL_STROKE = "#999999"

TEXT_MAIN     = "#222222"
TEXT_DIM      = "#555555"

ARROW         = "#555555"
FEEDBACK      = DUSTY_BLUE_DARK

# Stage content.
# Keep the figure terse. The manuscript text explains the details.
STAGES = [
    ("inputs",
     "Target Hamiltonian",
     [r"$\omega_q,\ \alpha$",
      r"$\omega_r,\ g,\ \kappa$"],
     "physics"),

    ("map",
     "Physics features",
     [r"$\omega_q \approx \sqrt{8E_JE_C}-E_C$",
      r"$\alpha \approx -E_C$"],
     "physics"),

    ("pre",
     "Scale and encode",
     [r"Min-max scaling",
      r"Categorical masks"],
     "physics"),

    ("mlps",
     "Inverse MLPs",
     [r"TransmonCross",
      r"NCap Coupler",
      r"Resonator"],
     "ml"),

    ("post",
     "Decode predictions",
     [r"Physical units",
      r"Valid design fields"],
     "ml"),

    ("fwd",
     "Forward check",
     [r"Quantum Metal layout",
      r"Ansys or surrogate"],
     "valid"),

    ("back",
     "Recovered\nHamiltonian",
     [r"$\hat{\omega}_q,\ \hat{\alpha};\ \hat{\omega}_r,\ \hat{g}$"],
     "valid"),

    ("cmp",
     "Compare",
     [r"Percent error",
      r"Iterate if needed"],
     "valid"),
]

FEEDBACK_LABEL = "Iterate"

CATEGORY_STYLE = {
    "neutral": dict(fill=NEUTRAL_FILL, stroke=NEUTRAL_STROKE, title=TEXT_MAIN, body=TEXT_DIM),
    "physics": dict(fill=FROST,      stroke=FROST_DARK,      title=FROST_DARK,      body=TEXT_MAIN),
    "ml":      dict(fill=PALE_ICE,   stroke=PALE_ICE_DARK,   title=PALE_ICE_DARK,   body=TEXT_MAIN),
    "valid":   dict(fill=DUSTY_BLUE_DARK, stroke=DUSTY_BLUE_DARK, title="#B8D4E3", body="#FFFFFF"),
}

CATEGORY_BADGE = {
    "physics": ("Physics targets", FROST_DARK),
    "ml":      ("ML", PALE_ICE_DARK),
    "valid":   ("Validation", DUSTY_BLUE_DARK),
}

# Compact two-row layout. The path runs left to right across the top row,
# then down and right to left across the bottom row.
FIG_W_IN = 8.7
FIG_H_IN = 3.15

BOX_W = 23.2
BOX_H = 10.5
ML_BOX_H = 12.0
GAP_X = 3.0
LEFT = 2.0
TOP_Y = 24.4
BOTTOM_Y = 6.0
TITLE_PAD_X = 1.0
TITLE_PAD_Y = 2.1
BODY_GAP = 2.6

COL_X = [LEFT + i * (BOX_W + GAP_X) for i in range(4)]
TOP_ROW = ["inputs", "map", "pre", "mlps"]
BOTTOM_ROW = ["cmp", "back", "fwd", "post"]

stage_lookup = {sid: (title, body, cat) for sid, title, body, cat in STAGES}
stage_rects = {
    "inputs": (COL_X[0], TOP_Y, BOX_W, BOX_H),
    "map": (COL_X[1], TOP_Y, BOX_W, BOX_H),
    "pre": (COL_X[2], TOP_Y, BOX_W, BOX_H),
    "mlps": (COL_X[3], TOP_Y + BOX_H - ML_BOX_H, BOX_W, ML_BOX_H),
    "post": (COL_X[3], BOTTOM_Y, BOX_W, BOX_H),
    "fwd": (COL_X[2], BOTTOM_Y, BOX_W, BOX_H),
    "back": (COL_X[1], BOTTOM_Y, BOX_W, BOX_H),
    "cmp": (COL_X[0], BOTTOM_Y, BOX_W, BOX_H),
}

fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
ax.set_xlim(0, 108)
ax.set_ylim(3.8, 38.7)
ax.axis("off")


def draw_group(x: float, y: float, w: float, h: float, cat: str) -> None:
    label, edge = CATEGORY_BADGE[cat]
    style = CATEGORY_STYLE[cat]
    group = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.8",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=style["fill"],
        alpha=0.28,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax.add_patch(group)
    # Use a darkened version of the edge color for the badge text
    # so it reads well against the semi-transparent lane background.
    badge_color = {"physics": "#3A5766", "ml": "#2A4F65", "valid": "#0D2636"}.get(cat, edge)
    ax.text(
        x + w / 2, y + h - 0.6,
        label,
        ha="center", va="top",
        fontsize=8.5, fontweight="bold", fontstyle="italic",
        color=badge_color,
        zorder=7,
    )


draw_group(COL_X[0] - 1.0, TOP_Y - 1.2, 3 * BOX_W + 2 * GAP_X + 2.0, BOX_H + 3.6, "physics")
draw_group(COL_X[3] - 1.0, BOTTOM_Y - 1.2, BOX_W + 2.0, TOP_Y + BOX_H - BOTTOM_Y + 3.6, "ml")
draw_group(COL_X[0] - 1.0, BOTTOM_Y - 1.2, 3 * BOX_W + 2 * GAP_X + 2.0, BOX_H + 3.6, "valid")


def draw_box(sid: str) -> None:
    x, y, w, h = stage_rects[sid]
    title, body, cat = stage_lookup[sid]
    style = CATEGORY_STYLE[cat]
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=1.0",
        linewidth=1.55,
        edgecolor=style["stroke"],
        facecolor=style["fill"],
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x + TITLE_PAD_X, y + h - TITLE_PAD_Y,
        title,
        ha="left", va="top",
        fontsize=8.9, fontweight="bold",
        color=style["title"],
        linespacing=0.9,
        zorder=5,
    )
    title_lines = title.count("\n") + 1
    body_start_offset = TITLE_PAD_Y + 2.9 + (title_lines - 1) * 2.3
    for idx, line in enumerate(body):
        ax.text(
            x + TITLE_PAD_X, y + h - body_start_offset - idx * BODY_GAP,
            line,
            ha="left", va="top",
            fontsize=7.9,
            color=style["body"],
            zorder=5,
        )


for sid in TOP_ROW + BOTTOM_ROW:
    draw_box(sid)


def edge_mid(sid: str, side: str) -> tuple[float, float]:
    x, y, w, h = stage_rects[sid]
    if side == "right":
        return x + w, y + h / 2
    if side == "left":
        return x, y + h / 2
    if side == "top":
        return x + w / 2, y + h
    if side == "bottom":
        return x + w / 2, y
    raise ValueError(side)


def draw_arrow(start_sid: str, start_side: str, end_sid: str, end_side: str, label: str, label_offset=(0, 0)) -> None:
    start = edge_mid(start_sid, start_side)
    end = edge_mid(end_sid, end_side)
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.85,
        color=ARROW,
        shrinkA=3.0,
        shrinkB=3.0,
        zorder=4,
    )
    ax.add_patch(arrow)
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(
            mx, my,
            label,
            ha="center", va="center",
            fontsize=7.2,
            color=TEXT_DIM,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.5),
            zorder=6,
        )


draw_arrow("inputs", "right", "map", "left", "")
draw_arrow("map", "right", "pre", "left", "")
pre_right = edge_mid("pre", "right")
mlps_x, _, _, _ = stage_rects["mlps"]
ax.add_patch(
    FancyArrowPatch(
        pre_right,
        (mlps_x, pre_right[1]),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.85,
        color=ARROW,
        shrinkA=3.0,
        shrinkB=3.0,
        zorder=4,
    )
)
draw_arrow("mlps", "bottom", "post", "top", "")
draw_arrow("post", "left", "fwd", "right", "")
draw_arrow("fwd", "left", "back", "right", "")
draw_arrow("back", "left", "cmp", "right", "")

map_bottom = edge_mid("map", "bottom")
cmp_left = edge_mid("cmp", "left")

feedback_start = (cmp_left[0] - 0.1, cmp_left[1])
feedback_elbow_1 = (0.6, cmp_left[1])
feedback_elbow_2 = (0.6, 21.2)
feedback_elbow_3 = (map_bottom[0], 21.2)

arrow_tip = (map_bottom[0], map_bottom[1] - 0.02)
arrow_base_y = arrow_tip[1] - 1.2

ax.plot(
    [
        feedback_start[0],
        feedback_elbow_1[0],
        feedback_elbow_2[0],
        feedback_elbow_3[0],
        arrow_tip[0],
    ],
    [
        feedback_start[1],
        feedback_elbow_1[1],
        feedback_elbow_2[1],
        feedback_elbow_3[1],
        arrow_base_y,
    ],
    color=FEEDBACK,
    linewidth=2.0,
    linestyle=(0, (6, 3)),
    dash_capstyle="round",
    dash_joinstyle="round",
    zorder=9,
)
ax.add_patch(
    Polygon(
        [
            arrow_tip,
            (arrow_tip[0] - 0.72, arrow_base_y),
            (arrow_tip[0] + 0.72, arrow_base_y),
        ],
        closed=True,
        facecolor=FEEDBACK,
        edgecolor=FEEDBACK,
        linewidth=0,
        zorder=10,
    )
)

ax.text(
    13.0,
    22.35,
    FEEDBACK_LABEL,
    ha="center", va="center",
    fontsize=7.6, fontweight="bold", fontstyle="italic",
    color=FEEDBACK,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.5),
    zorder=10,
)

# Save
workflow_pdf = MANUSCRIPT_EXPORTS_DIR / "workflow.pdf"

plt.savefig(workflow_pdf, bbox_inches="tight", pad_inches=0.04)
print(f"Written {workflow_pdf}")
