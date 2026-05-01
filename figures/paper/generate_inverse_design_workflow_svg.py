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
    outputs/workflow.svg   (vector, for web / quick preview)
    outputs/workflow.pdf   (vector, for Overleaf \\includegraphics)

Usage:
    python3 generate_inverse_design_workflow_svg.py
"""

import io
import os

from _paths import OUTPUTS_DIR

# Color palette (same family as inverse_pipeline figure)
BG            = "#FFFFFF"

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

# Neutral utility box (the initial "Inputs" target row)
NEUTRAL_FILL  = "#F5F5F5"
NEUTRAL_STROKE = "#999999"

TEXT_MAIN     = "#222222"
TEXT_DIM      = "#555555"
TEXT_MONO     = "#444444"

ARROW         = "#555555"
FEEDBACK      = DUSTY_BLUE_DARK

# Layout
W = 540
H = 680

# Pipeline column
BOX_X     = 95
BOX_W     = 225
BOX_R     = 6

# Annotation column (to the right of pipeline)
ANN_X     = 350
ANN_W     = 170

# Stages (id, title, body_lines, category, height)
STAGES = [
    ("inputs",
     "Inputs — target Hamiltonian",
     ["ω_q, α, ω_r, g, κ, …",
      "(user-specified targets)"],
     "physics", 60),

    ("map",
     "Physics mapping",
     ["Convert Hamiltonian targets to",
      "ML-friendly features using",
      "analytic relations (e.g. α ≈ −E_C)."],
     "physics", 75),

    ("pre",
     "Preprocessing",
     ["Scale features (min–max fit",
      "on train set); one-hot encode",
      "categoricals and “exists” masks."],
     "physics", 75),

    ("mlps",
     "Three trained inverse MLPs",
     ["• TransmonCross: x_q → a_q",
      "• Coupler / NCap: x_c → a_c",
      "• Cavity / res: x_r → a_r"],
     "ml", 75),

    ("post",
     "Postprocessing",
     ["Unscale predictions back to",
      "physical units; decode rules",
      "and apply “exists” masks."],
     "ml", 75),

    ("fwd",
     "Forward validation",
     ["Assemble design in Quantum Metal;",
      "run Ansys Q3D (capacitance) and",
      "HFSS to extract physical values."],
     "valid", 75),

    ("back",
     "Map back to Hamiltonian",
     ["Convert capacitances & modes",
      "back to achieved ω_q, α, ω_r, g",
      "via inverse physics map."],
     "valid", 75),

    ("cmp",
     "Compare and iterate",
     ["RMSPE between target",
      "and achieved Hamiltonian;",
      "optionally refine targets."],
     "valid", 75),
]

ANNOTATIONS = {
    "map":  ("target vector",
             ["H_target = (ω_q, α, …)"]),
    "pre":  ("feature vector",
             ["x_raw ∈ R^d_in",
              "physical quantities"]),
    "mlps": ("scaled features",
             ["x = scaler(x_raw)"]),
    "post": ("raw NN outputs",
             ["a_q, a_c, a_r",
              "in scaled form"]),
    "fwd":  ("Quantum Metal params",
             ["y_q, y_c, y_r",
              "+ design choices"]),
    "back": ("extracted quantities",
             ["C_ij matrix",
              "f_mode (HFSS)"]),
    "cmp":  ("achieved Hamiltonian",
             ["H_pred = (ω_q, α, …)"]),
}

# Feedback annotation
FEEDBACK_LABEL = "refined targets / retry"

CATEGORY_STYLE = {
    "neutral": (NEUTRAL_FILL, NEUTRAL_STROKE, TEXT_MAIN, TEXT_DIM),
    "physics": (FROST,      FROST_DARK,      FROST_DARK,      TEXT_MAIN),
    "ml":      (PALE_ICE,   PALE_ICE_DARK,   PALE_ICE_DARK,   TEXT_MAIN),
    "valid":   (DUSTY_BLUE, DUSTY_BLUE_DARK, DUSTY_BLUE_DARK, TEXT_MAIN),
}

CATEGORY_BADGE = {
    "physics": ("Physics targets", FROST_DARK),
    "ml":      ("ML surrogate", PALE_ICE_DARK),
    "valid":   ("Validation", DUSTY_BLUE_DARK),
}

# Precompute ycoordinates for each stage
Y_START = 20
GAP     = 16

stage_y = {}
y = Y_START
for sid, _, _, _, h in STAGES:
    stage_y[sid] = (y, y + h)
    y += h + GAP
TOTAL_H = y

# SVG construction
out = io.StringIO()

out.write(f'''<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 {W} {TOTAL_H + 20}"
     font-family="'Helvetica Neue', Arial, Helvetica, sans-serif">
  <rect width="{W}" height="{TOTAL_H + 20}" fill="{BG}"/>

  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{ARROW}"/>
    </marker>
    <marker id="arrowFb" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{FEEDBACK}"/>
    </marker>
  </defs>
''')

# Lane backgrounds
lanes = []
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

LANE_X = BOX_X - 16
LANE_W = BOX_W + 32

for cat, top, bot in lanes:
    if cat == "neutral":
        continue
    fill, stroke, _, _ = CATEGORY_STYLE[cat]
    badge, badge_col = CATEGORY_BADGE[cat]
    out.write(
        f'  <rect x="{LANE_X}" y="{top - 8}" width="{LANE_W}" '
        f'height="{bot - top + 16}" rx="10" ry="10" '
        f'fill="{fill}" fill-opacity="0.45" '
        f'stroke="{stroke}" stroke-width="1.5" stroke-opacity="0.55" '
        f'stroke-dasharray="4,3"/>\n'
    )
    badge_cx = LANE_X - 12
    badge_cy = (top + bot) / 2
    out.write(
        f'  <text x="{badge_cx}" y="{badge_cy}" '
        f'text-anchor="middle" font-size="10" font-weight="bold" '
        f'font-style="italic" fill="{badge_col}" '
        f'transform="rotate(-90 {badge_cx} {badge_cy})">{badge}</text>\n'
    )

# Pipeline boxes
for sid, title, body, cat, h in STAGES:
    top, bot = stage_y[sid]
    fill, stroke, title_col, body_col = CATEGORY_STYLE[cat]

    out.write(
        f'  <rect x="{BOX_X}" y="{top}" width="{BOX_W}" height="{h}" '
        f'rx="{BOX_R}" ry="{BOX_R}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>\n'
    )
    out.write(
        f'  <text x="{BOX_X + 10}" y="{top + 18}" '
        f'font-size="12" font-weight="bold" fill="{title_col}">{title}</text>\n'
    )
    line_y = top + 34
    for line in body:
        is_mono = line.startswith("•") or "→" in line
        fcol = TEXT_MONO if is_mono else body_col
        fsize = 9 if is_mono else 10
        fam = "Menlo, Consolas, monospace" if is_mono else "inherit"
        out.write(
            f'  <text x="{BOX_X + 10}" y="{line_y}" '
            f'font-size="{fsize}" fill="{fcol}" '
            f'font-family="{fam}">{line}</text>\n'
        )
        line_y += 14

# Forward arrows
for i in range(len(STAGES) - 1):
    _, bot = stage_y[STAGES[i][0]]
    top, _ = stage_y[STAGES[i + 1][0]]
    cx = BOX_X + BOX_W / 2
    out.write(
        f'  <line x1="{cx}" y1="{bot}" x2="{cx}" y2="{top}" '
        f'stroke="{ARROW}" stroke-width="1.5" marker-end="url(#arrow)"/>\n'
    )

# Annotations (right column)
for i in range(1, len(STAGES)):
    sid = STAGES[i][0]
    if sid not in ANNOTATIONS:
        continue
    label, details = ANNOTATIONS[sid]
    top, _ = stage_y[sid]

    card_h = 16 + 12 * len(details) + 6
    card_y = top
    out.write(
        f'  <rect x="{ANN_X}" y="{card_y}" width="{ANN_W}" height="{card_h}" '
        f'rx="4" ry="4" fill="#FAFAFA" stroke="#CCCCCC" stroke-width="1"/>\n'
    )
    out.write(
        f'  <text x="{ANN_X + 8}" y="{card_y + 14}" '
        f'font-size="10" font-weight="bold" fill="{TEXT_MAIN}">{label}</text>\n'
    )
    dy = card_y + 28
    for d in details:
        out.write(
            f'  <text x="{ANN_X + 8}" y="{dy}" font-size="9" '
            f'font-family="Menlo, Consolas, monospace" '
            f'fill="{TEXT_DIM}">{d}</text>\n'
        )
        dy += 12
    out.write(
        f'  <line x1="{BOX_X + BOX_W}" y1="{card_y + card_h/2}" '
        f'x2="{ANN_X}" y2="{card_y + card_h/2}" '
        f'stroke="#BBBBBB" stroke-width="1" stroke-dasharray="3,3"/>\n'
    )

# Feedback loop
map_top, _ = stage_y["map"]
_, cmp_bot = stage_y["cmp"]
cmp_mid_y = (stage_y["cmp"][0] + stage_y["cmp"][1]) / 2
map_mid_y = (stage_y["map"][0] + stage_y["map"][1]) / 2

FB_X = 44
out.write(
    f'  <path d="M {BOX_X},{cmp_mid_y} '
    f'L {FB_X},{cmp_mid_y} '
    f'L {FB_X},{map_mid_y} '
    f'L {BOX_X},{map_mid_y}" '
    f'fill="none" stroke="{FEEDBACK}" stroke-width="2" '
    f'stroke-dasharray="5,3" marker-end="url(#arrowFb)"/>\n'
)
fb_label_y = (cmp_mid_y + map_mid_y) / 2
out.write(
    f'  <text x="{FB_X - 6}" y="{fb_label_y}" '
    f'text-anchor="middle" font-size="9" font-style="italic" '
    f'font-weight="bold" fill="{FEEDBACK}" '
    f'transform="rotate(-90 {FB_X - 6} {fb_label_y})">{FEEDBACK_LABEL}</text>\n'
)

out.write("</svg>\n")

SVG = out.getvalue()

# Write files
workflow_svg = OUTPUTS_DIR / "workflow.svg"
workflow_pdf = OUTPUTS_DIR / "workflow.pdf"

with workflow_svg.open("w", encoding="utf-8") as f:
    f.write(SVG)
print(f"Written {workflow_svg}")

try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF
    drawing = svg2rlg(io.StringIO(SVG))
    renderPDF.drawToFile(drawing, str(workflow_pdf))
    print(f"Written {workflow_pdf}")
except ImportError:
    print(f"svglib / reportlab missing, could not write {workflow_pdf}")
