#!/usr/bin/env python3
r"""
Generate the end-to-end inverse-design workflow figure.

Uses matplotlib's mathtext renderer so variables like $\omega_q$, $\hat{y}_q$,
$\mathbb{R}^{d_{in}}$ appear as proper typeset math — no LaTeX install needed.

Outputs:
    workflow.svg
    workflow.pdf

Usage:
    python3 generate_workflow_mpl.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.transforms import Bbox

# Use mathtext (built in, ships with matplotlib) — NOT full LaTeX
plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "cm"       # Computer Modern look for math
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

# ---- Color palette (same family as inverse_pipeline figure) -------------
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
FEEDBACK      = GREEN

# ---- Stage content ------------------------------------------------------
# Each stage: id, title, body lines (mathtext allowed via $...$), category, height
STAGES = [
    ("inputs",
     "Inputs — target Hamiltonian",
     [r"$\omega_q,\ \alpha,\ \omega_r,\ g,\ \kappa,\ \ldots$   (user-specified targets)"],
     "neutral"),

    ("map",
     "Physics mapping",
     [r"Convert Hamiltonian targets to ML-friendly features",
      r"using analytic relations, e.g. $\alpha \approx -E_C$ and",
      r"$\omega_q \approx \sqrt{8 E_J E_C} - E_C$."],
     "physics"),

    ("pre",
     "Preprocessing",
     [r"Scale features (min–max fit on training set);",
      r'one-hot encode categoricals and "exists" masks.'],
     "physics"),

    ("mlps",
     "Three trained inverse MLPs",
     [r"$\bullet$  TransmonCross  (cap$\_$matrix):     $\mathbf{x}_q \rightarrow \mathbf{y}_q$",
      r"$\bullet$  Coupler / NCap  (cap$\_$matrix):     $\mathbf{x}_c \rightarrow \mathbf{y}_c$",
      r"$\bullet$  Cavity / resonator  (eigenmode):  $\mathbf{x}_r \rightarrow \mathbf{y}_r$"],
     "ml"),

    ("post",
     "Postprocessing",
     [r"Unscale predictions back to physical units;",
      r'decode categorical outputs and apply "exists" masks.'],
     "ml"),

    ("fwd",
     "Forward validation",
     [r"Look up closest design via SQuADDS; assemble layout",
      r"in Qiskit Metal; run pyEPR $\rightarrow$ PyAEDT $\rightarrow$",
      r"Ansys Q3D (cap. matrix) and HFSS eigenmode (resonator)."],
     "valid"),

    ("back",
     "Map back to Hamiltonian space",
     [r"Convert extracted capacitances and mode frequencies",
      r"back to achieved $\omega_q,\ \alpha,\ \omega_r,\ g$",
      r"via the inverse physics map."],
     "valid"),

    ("cmp",
     "Compare and iterate",
     [r"Validation loss: RMSPE between reference (SQuADDS)",
      r"and predicted (forward-pass) Hamiltonian values;",
      r"optionally refine features and rerun the pipeline."],
     "valid"),
]

# Annotations: data object flowing INTO each stage (except the first).
ANNOTATIONS = {
    "map":  ("target vector",
             [r"$\mathbf{H}_{\mathrm{target}} = (\omega_q,\ \alpha,\ \omega_r,\ g,\ \ldots)$"]),
    "pre":  ("feature vector",
             [r"$\mathbf{x}_{\mathrm{raw}} \in \mathbb{R}^{d_{\mathrm{in}}}$",
              r"physically-meaningful quantities"]),
    "mlps": ("scaled features",
             [r"$\mathbf{x} = \mathrm{scaler}(\mathbf{x}_{\mathrm{raw}})$",
              r"ready for NN input"]),
    "post": ("raw NN outputs",
             [r"$\hat{\mathbf{y}}_q,\ \hat{\mathbf{y}}_c,\ \hat{\mathbf{y}}_r$",
              r"in scaled / encoded form"]),
    "fwd":  ("Qiskit Metal params",
             [r"$\mathbf{y}_q,\ \mathbf{y}_c,\ \mathbf{y}_r$   ($\mu$m, counts, …)",
              r"+ categorical design choices"]),
    "back": ("extracted quantities",
             [r"$C_{ij}$   (capacitance matrix)",
              r"$f_{\mathrm{mode}}$  (HFSS eigenmode)"]),
    "cmp":  ("achieved Hamiltonian",
             [r"$\mathbf{H}_{\mathrm{achieved}} = (\hat{\omega}_q,\ \hat{\alpha},\ \hat{\omega}_r,\ \hat{g},\ \ldots)$"]),
}

FEEDBACK_LABEL = "refined targets / retry"

CATEGORY_STYLE = {
    "neutral": dict(fill=NEUTRAL_FILL, stroke=NEUTRAL_STROKE, title=TEXT_MAIN, body=TEXT_DIM),
    "physics": dict(fill=ORANGE_LIGHT, stroke=ORANGE, title=ORANGE_DARK, body=TEXT_MAIN),
    "ml":      dict(fill=GREEN_LIGHT,  stroke=GREEN,  title=GREEN_DARK,  body=TEXT_MAIN),
    "valid":   dict(fill=PURPLE_LIGHT, stroke=PURPLE, title=PURPLE_DARK, body=TEXT_MAIN),
}

CATEGORY_BADGE = {
    "physics": ("Physics",      ORANGE),
    "ml":      ("ML surrogate", GREEN),
    "valid":   ("Validation",   PURPLE),
}

# ---- Layout (in figure data coordinates, arbitrary units) ---------------
# We use a plain (0..100, 0..100) coordinate system and size the figure to
# match the aspect ratio we want.
FIG_W_IN = 12
FIG_H_IN = 13

BOX_X      = 14      # left edge of pipeline boxes
BOX_W      = 60      # pipeline box width — wider to fit long lines
TITLE_DY   = 3.2     # vertical offset from box top to title baseline
LINE_DY    = 2.6     # vertical spacing between body lines
TITLE_PAD  = 1.6     # space from title to first body line

ANN_X      = 78      # annotation cards
ANN_W      = 28

# Stage heights are computed from number of body lines.
def stage_height(body_lines):
    return 2.0 + TITLE_PAD + TITLE_DY + LINE_DY * len(body_lines) + 1.5

# Y layout — top-down
Y_TOP = 97
GAP = 1.8

stage_rects = {}  # id -> (x, y_top, w, h)
y_cursor = Y_TOP
for sid, title, body, cat in STAGES:
    h = stage_height(body)
    stage_rects[sid] = (BOX_X, y_cursor, BOX_W, h)
    y_cursor -= (h + GAP)
Y_BOTTOM = y_cursor + GAP   # bottom of last box

# ---- Build figure -------------------------------------------------------
fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
ax.set_xlim(0, ANN_X + ANN_W + 4)
ax.set_ylim(Y_BOTTOM - 2, 100)
ax.set_aspect("equal")
ax.axis("off")

# ---- Draw category lane backgrounds ------------------------------------
# Group consecutive same-category stages into a lane rectangle.
lanes = []
cur_cat = None
lane_top = None
prev_bot = None
for sid, _, _, cat in STAGES:
    x, y_top, w, h = stage_rects[sid]
    y_bot = y_top - h
    if cat != cur_cat:
        if cur_cat is not None:
            lanes.append((cur_cat, lane_top, prev_bot))
        cur_cat = cat
        lane_top = y_top
    prev_bot = y_bot
lanes.append((cur_cat, lane_top, prev_bot))

LANE_PAD_X = 3.5
LANE_PAD_Y = 1.1

for cat, top, bot in lanes:
    if cat == "neutral":
        continue
    style = CATEGORY_STYLE[cat]
    lane = Rectangle(
        (BOX_X - LANE_PAD_X, bot - LANE_PAD_Y),
        BOX_W + 2 * LANE_PAD_X,
        (top - bot) + 2 * LANE_PAD_Y,
        linewidth=1.2,
        edgecolor=style["stroke"],
        facecolor=style["fill"],
        alpha=0.45,
        linestyle="--",
    )
    ax.add_patch(lane)
    # Badge label, vertical, on the left
    badge, badge_col = CATEGORY_BADGE[cat]
    ax.text(
        BOX_X - LANE_PAD_X - 1.3,
        (top + bot) / 2,
        badge,
        rotation=90,
        ha="center", va="center",
        fontsize=10, fontweight="bold", fontstyle="italic",
        color=badge_col,
    )

# ---- Draw pipeline boxes ------------------------------------------------
for sid, title, body, cat in STAGES:
    x, y_top, w, h = stage_rects[sid]
    y_bot = y_top - h
    style = CATEGORY_STYLE[cat]

    box = FancyBboxPatch(
        (x, y_bot), w, h,
        boxstyle="round,pad=0,rounding_size=1.0",
        linewidth=1.8,
        edgecolor=style["stroke"],
        facecolor=style["fill"],
    )
    ax.add_patch(box)

    # Title
    ax.text(
        x + 1.2, y_top - TITLE_DY,
        title,
        ha="left", va="top",
        fontsize=11, fontweight="bold",
        color=style["title"],
    )
    # Body lines
    for i, line in enumerate(body):
        ax.text(
            x + 1.2,
            y_top - TITLE_DY - TITLE_PAD - (i + 1) * LINE_DY,
            line,
            ha="left", va="top",
            fontsize=9.5,
            color=style["body"],
        )

# ---- Forward arrows between pipeline boxes ------------------------------
for i in range(len(STAGES) - 1):
    _, y_top_a, w, h_a = stage_rects[STAGES[i][0]]
    _, y_top_b, _, _   = stage_rects[STAGES[i + 1][0]]
    cx = BOX_X + BOX_W / 2
    y_a_bot = y_top_a - h_a
    arrow = FancyArrowPatch(
        (cx, y_a_bot),
        (cx, y_top_b),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.6,
        color=ARROW,
    )
    ax.add_patch(arrow)

# ---- Annotation cards ---------------------------------------------------
for i in range(1, len(STAGES)):
    sid = STAGES[i][0]
    if sid not in ANNOTATIONS:
        continue
    label, details = ANNOTATIONS[sid]
    x, y_top, w, h = stage_rects[sid]

    n_lines = 1 + len(details)
    card_h = 2.0 + TITLE_DY * 0.9 + LINE_DY * len(details) + 1.5
    # Align card vertically with its pipeline box
    card_top = y_top - 0.2

    card = FancyBboxPatch(
        (ANN_X, card_top - card_h), ANN_W, card_h,
        boxstyle="round,pad=0,rounding_size=0.7",
        linewidth=1.0,
        edgecolor="#CCCCCC",
        facecolor="#FAFAFA",
    )
    ax.add_patch(card)

    # Label
    ax.text(
        ANN_X + 1.0, card_top - 1.6,
        label,
        ha="left", va="top",
        fontsize=9, fontweight="bold",
        color=TEXT_MAIN,
    )
    # Details
    for j, d in enumerate(details):
        ax.text(
            ANN_X + 1.0,
            card_top - 1.6 - TITLE_PAD - (j + 1) * LINE_DY * 0.9,
            d,
            ha="left", va="top",
            fontsize=9,
            color=TEXT_DIM,
        )
    # Dashed connector from box right edge to card left edge
    y_mid = card_top - card_h / 2
    ax.plot(
        [BOX_X + BOX_W, ANN_X],
        [y_mid, y_mid],
        color="#BBBBBB", linewidth=0.8, linestyle=(0, (3, 3)),
    )

# ---- Feedback loop: compare/iterate -> physics mapping ------------------
map_x, map_y_top, map_w, map_h = stage_rects["map"]
cmp_x, cmp_y_top, cmp_w, cmp_h = stage_rects["cmp"]

map_mid_y = map_y_top - map_h / 2
cmp_mid_y = cmp_y_top - cmp_h / 2
FB_X = BOX_X - 8

# Draw the three segments manually as lines + arrowhead
ax.plot([BOX_X, FB_X], [cmp_mid_y, cmp_mid_y],
        color=FEEDBACK, linewidth=1.8, linestyle=(0, (7, 4)))
ax.plot([FB_X, FB_X], [cmp_mid_y, map_mid_y],
        color=FEEDBACK, linewidth=1.8, linestyle=(0, (7, 4)))
fb_arrow = FancyArrowPatch(
    (FB_X, map_mid_y),
    (BOX_X, map_mid_y),
    arrowstyle="-|>",
    mutation_scale=14,
    linewidth=1.8,
    color=FEEDBACK,
    linestyle=(0, (7, 4)),
)
ax.add_patch(fb_arrow)

# Feedback label, vertical on the return arm
ax.text(
    FB_X - 1.2, (cmp_mid_y + map_mid_y) / 2,
    FEEDBACK_LABEL,
    rotation=90, ha="center", va="center",
    fontsize=9, fontweight="bold", fontstyle="italic",
    color=FEEDBACK,
)

# ---- Save ---------------------------------------------------------------
plt.savefig("paper_plots/workflow.pdf", bbox_inches="tight", pad_inches=0.1)
plt.savefig("paper_plots/workflow.svg", bbox_inches="tight", pad_inches=0.1)
print("Written workflow.pdf and workflow.svg")
