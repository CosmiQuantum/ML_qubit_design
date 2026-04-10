#!/usr/bin/env python3
"""
Generate the end-to-end inverse-design workflow figure.

Produces a two-column flowchart:
  - left column: the linear pipeline, grouped into three colored "lanes"
    (Physics / ML / Validation) so the reader sees the structure at a glance
  - right column: compact annotations showing what data object flows
    between each pair of stages (targets, feature vectors, Qiskit parameters,
    extracted capacitances, achieved Hamiltonian values, errors)

Outputs:
    workflow.svg   (vector, for web / quick preview)
    workflow.pdf   (vector, for Overleaf \\includegraphics)

Usage:
    python3 generate_workflow_svg.py
"""

import io

# ---- Color palette (same family as inverse_pipeline figure) -------------
BG            = "#FFFFFF"

# Green — ML / trained-model stages
GREEN         = "#3D8B3D"
GREEN_LIGHT   = "#E8F5E8"
GREEN_DARK    = "#2E6B2E"

# Orange — physics / domain-knowledge stages (Qubit Subsystem accent)
ORANGE        = "#E87A00"
ORANGE_LIGHT  = "#FFF4E6"
ORANGE_DARK   = "#A85600"

# Purple — validation / EM-simulation stages (CPW Cavity accent)
PURPLE        = "#7B68AE"
PURPLE_LIGHT  = "#E8E4F0"
PURPLE_DARK   = "#4A3D78"

# Neutral utility box (the initial "Inputs" target row)
NEUTRAL_FILL  = "#F5F5F5"
NEUTRAL_STROKE = "#999999"

TEXT_MAIN     = "#222222"
TEXT_DIM      = "#555555"
TEXT_MONO     = "#444444"

ARROW         = "#555555"
FEEDBACK      = GREEN

# ---- Layout -------------------------------------------------------------
W = 980
H = 960

# Pipeline column
BOX_X     = 130        # left edge of pipeline boxes
BOX_W     = 430        # pipeline box width (wider to fit longest lines)
BOX_R     = 10         # corner radius

# Annotation column (to the right of pipeline)
ANN_X     = 620
ANN_W     = 320

# Stages: (id, title, body_lines, category, height)
# category controls color: 'neutral' | 'physics' | 'ml' | 'valid'
STAGES = [
    ("inputs",
     "Inputs — target Hamiltonian",
     [r"ω_q, α, ω_r, g, κ, …   (user-specified targets)"],
     "neutral", 56),

    ("map",
     "Physics mapping",
     ["Convert Hamiltonian targets to ML-friendly features",
      "using analytic relations (e.g. α ≈ −E_C, ω_q ≈ √(8 E_J E_C) − E_C)."],
     "physics", 72),

    ("pre",
     "Preprocessing",
     ["Scale features (min–max fit on train set);",
      "one-hot encode categoricals and “exists” masks."],
     "physics", 72),

    ("mlps",
     "Three trained inverse MLPs",
     ["• TransmonCross  (cap_matrix):       x_q → y_q",
      "• Coupler / NCap (cap_matrix):       x_c → y_c",
      "• Cavity / resonator (eigenmode):    x_r → y_r"],
     "ml", 104),

    ("post",
     "Postprocessing",
     ["Unscale predictions back to physical units;",
      "decode categorical outputs and apply “exists” masks."],
     "ml", 72),

    ("fwd",
     "Forward validation",
     ["Assemble design in Qiskit Metal; run Ansys Q3D (capacitance)",
      "and HFSS eigenmode (resonator) to extract physical quantities."],
     "valid", 72),

    ("back",
     "Map back to Hamiltonian space",
     ["Convert extracted capacitances and mode frequencies",
      "back to achieved ω_q, α, ω_r, g via the inverse physics map."],
     "valid", 72),

    ("cmp",
     "Compare and iterate",
     ["RMSPE between target and achieved Hamiltonian values;",
      "optionally refine features and rerun the pipeline."],
     "valid", 72),
]

# Annotations: data object that flows INTO each stage (except the first).
# Keyed by stage id.  Each entry: (short_label, detail_lines)
ANNOTATIONS = {
    "map":  ("target vector",
             ["H_target = (ω_q, α, ω_r, g, …)"]),
    "pre":  ("feature vector",
             ["x_raw  ∈  R^d_in",
              "physically-meaningful quantities"]),
    "mlps": ("scaled features",
             ["x  =  scaler(x_raw)",
              "ready for NN input"]),
    "post": ("raw NN outputs",
             ["y_q(pred), y_c(pred), y_r(pred)",
              "in scaled / encoded form"]),
    "fwd":  ("Qiskit Metal params",
             ["y_q, y_c, y_r   (µm, counts, …)",
              "+ categorical design choices"]),
    "back": ("extracted quantities",
             ["C_ij   (capacitance matrix)",
              "f_mode (HFSS eigenmode)"]),
    "cmp":  ("achieved Hamiltonian",
             ["H_pred = (ω_q, α, ω_r, g, …)_achieved"]),
}

# Feedback annotation (on the return arrow)
FEEDBACK_LABEL = "refined targets / retry"

CATEGORY_STYLE = {
    "neutral": (NEUTRAL_FILL, NEUTRAL_STROKE, TEXT_MAIN, TEXT_DIM),
    "physics": (ORANGE_LIGHT, ORANGE,        ORANGE_DARK, TEXT_MAIN),
    "ml":      (GREEN_LIGHT,  GREEN,         GREEN_DARK,  TEXT_MAIN),
    "valid":   (PURPLE_LIGHT, PURPLE,        PURPLE_DARK, TEXT_MAIN),
}

CATEGORY_BADGE = {
    "physics": ("Physics",     ORANGE),
    "ml":      ("ML surrogate", GREEN),
    "valid":   ("Validation",  PURPLE),
}

# ---- Precompute y-coordinates for each stage ----------------------------
Y_START = 40
GAP     = 22

stage_y = {}
y = Y_START
for sid, _, _, _, h in STAGES:
    stage_y[sid] = (y, y + h)   # (top, bottom)
    y += h + GAP
TOTAL_H = y

# ---- SVG construction ---------------------------------------------------
out = io.StringIO()

out.write(f'''<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 {W} {TOTAL_H + 40}"
     font-family="'Helvetica Neue', Arial, Helvetica, sans-serif">
  <rect width="{W}" height="{TOTAL_H + 40}" fill="{BG}"/>

  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{ARROW}"/>
    </marker>
    <marker id="arrowFb" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{FEEDBACK}"/>
    </marker>
    <marker id="tick" viewBox="0 0 10 10" refX="5" refY="5"
            markerWidth="5" markerHeight="5" orient="auto">
      <circle cx="5" cy="5" r="2.5" fill="{ARROW}"/>
    </marker>
  </defs>
''')

# ---- Lane backgrounds (grouped color bands behind the pipeline) ---------
# Group consecutive stages with the same category to form a shaded lane.
lanes = []  # list of (category, y_top, y_bot)
cur_cat = None
lane_top = None
for sid, _, _, cat, _ in STAGES:
    top, bot = stage_y[sid]
    if cat != cur_cat:
        if cur_cat is not None:
            lanes.append((cur_cat, lane_top, prev_bot))
        cur_cat = cat
        lane_top = top
    prev_bot = bot
lanes.append((cur_cat, lane_top, prev_bot))

LANE_X = BOX_X - 30
LANE_W = BOX_W + 60

for cat, top, bot in lanes:
    if cat == "neutral":
        continue  # skip the neutral "inputs" lane
    fill, stroke, _, _ = CATEGORY_STYLE[cat]
    badge, badge_col = CATEGORY_BADGE[cat]
    out.write(
        f'  <rect x="{LANE_X}" y="{top - 10}" width="{LANE_W}" '
        f'height="{bot - top + 20}" rx="16" ry="16" '
        f'fill="{fill}" fill-opacity="0.45" '
        f'stroke="{stroke}" stroke-width="1.5" stroke-opacity="0.55" '
        f'stroke-dasharray="6,4"/>\n'
    )
    # Badge label, vertical, on left side of the lane
    badge_cx = LANE_X - 8
    badge_cy = (top + bot) / 2
    out.write(
        f'  <text x="{badge_cx}" y="{badge_cy}" '
        f'text-anchor="middle" font-size="12" font-weight="bold" '
        f'font-style="italic" fill="{badge_col}" '
        f'transform="rotate(-90 {badge_cx} {badge_cy})">{badge}</text>\n'
    )

# ---- Pipeline boxes -----------------------------------------------------
for sid, title, body, cat, h in STAGES:
    top, bot = stage_y[sid]
    fill, stroke, title_col, body_col = CATEGORY_STYLE[cat]

    out.write(
        f'  <rect x="{BOX_X}" y="{top}" width="{BOX_W}" height="{h}" '
        f'rx="{BOX_R}" ry="{BOX_R}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>\n'
    )
    # Title
    out.write(
        f'  <text x="{BOX_X + 14}" y="{top + 22}" '
        f'font-size="14" font-weight="bold" fill="{title_col}">{title}</text>\n'
    )
    # Body lines
    line_y = top + 42
    for line in body:
        # Use a monospace-feeling weight for bullet / equation lines
        is_mono = line.startswith("•") or "→" in line
        fcol = TEXT_MONO if is_mono else body_col
        fsize = 12 if is_mono else 12
        fam = "Menlo, Consolas, monospace" if is_mono else "inherit"
        out.write(
            f'  <text x="{BOX_X + 14}" y="{line_y}" '
            f'font-size="{fsize}" fill="{fcol}" '
            f'font-family="{fam}">{line}</text>\n'
        )
        line_y += 16

# ---- Forward arrows between pipeline boxes ------------------------------
for i in range(len(STAGES) - 1):
    _, bot = stage_y[STAGES[i][0]]
    top, _ = stage_y[STAGES[i + 1][0]]
    cx = BOX_X + BOX_W / 2
    out.write(
        f'  <line x1="{cx}" y1="{bot}" x2="{cx}" y2="{top}" '
        f'stroke="{ARROW}" stroke-width="2" marker-end="url(#arrow)"/>\n'
    )

# ---- Annotations (right column) -----------------------------------------
# For each forward arrow, attach a small labeled card beside the arrow's
# destination stage showing what data flows in.
for i in range(1, len(STAGES)):
    sid = STAGES[i][0]
    if sid not in ANNOTATIONS:
        continue
    label, details = ANNOTATIONS[sid]
    top, _ = stage_y[sid]

    card_h = 18 + 14 * len(details) + 8
    card_y = top
    # Card
    out.write(
        f'  <rect x="{ANN_X}" y="{card_y}" width="{ANN_W}" height="{card_h}" '
        f'rx="6" ry="6" fill="#FAFAFA" stroke="#CCCCCC" stroke-width="1"/>\n'
    )
    # Label
    out.write(
        f'  <text x="{ANN_X + 10}" y="{card_y + 15}" '
        f'font-size="11" font-weight="bold" fill="{TEXT_MAIN}">{label}</text>\n'
    )
    # Detail lines (monospace, dim)
    dy = card_y + 30
    for d in details:
        out.write(
            f'  <text x="{ANN_X + 10}" y="{dy}" font-size="11" '
            f'font-family="Menlo, Consolas, monospace" '
            f'fill="{TEXT_DIM}">{d}</text>\n'
        )
        dy += 14
    # Short connector dashed line from pipeline box edge to card left edge
    out.write(
        f'  <line x1="{BOX_X + BOX_W}" y1="{card_y + card_h/2}" '
        f'x2="{ANN_X}" y2="{card_y + card_h/2}" '
        f'stroke="#BBBBBB" stroke-width="1" stroke-dasharray="3,3"/>\n'
    )

# ---- Feedback loop: cmp → map -------------------------------------------
map_top, _ = stage_y["map"]
_, cmp_bot = stage_y["cmp"]
cmp_mid_y = (stage_y["cmp"][0] + stage_y["cmp"][1]) / 2
map_mid_y = (stage_y["map"][0] + stage_y["map"][1]) / 2

# Path: from left edge of cmp, out to x=80, up to map's left edge
FB_X = 80
out.write(
    f'  <path d="M {BOX_X},{cmp_mid_y} '
    f'L {FB_X},{cmp_mid_y} '
    f'L {FB_X},{map_mid_y} '
    f'L {BOX_X},{map_mid_y}" '
    f'fill="none" stroke="{FEEDBACK}" stroke-width="2.5" '
    f'stroke-dasharray="7,4" marker-end="url(#arrowFb)"/>\n'
)
# Label on feedback arm, rotated
fb_label_y = (cmp_mid_y + map_mid_y) / 2
out.write(
    f'  <text x="{FB_X - 6}" y="{fb_label_y}" '
    f'text-anchor="middle" font-size="11" font-style="italic" '
    f'font-weight="bold" fill="{FEEDBACK}" '
    f'transform="rotate(-90 {FB_X - 6} {fb_label_y})">{FEEDBACK_LABEL}</text>\n'
)

out.write("</svg>\n")

SVG = out.getvalue()

# ---- Write SVG ----------------------------------------------------------
with open("paper_plots/workflow.svg", "w") as f:
    f.write(SVG)
print("Written paper_plots/workflow.svg")

# ---- Write PDF via svglib (pure-Python, no Cairo needed) ---------------
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

drawing = svg2rlg(io.StringIO(SVG))
renderPDF.drawToFile(drawing, "paper_plots/workflow.pdf")
print("Written paper_plots/workflow.pdf")
