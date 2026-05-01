#!/usr/bin/env python3
"""
Generate the Inverse Design Training Pipeline SVG figure.

This produces an overall flowchart of the inverse-design training loop:
  Desired Hamiltonian -> Inverse MLP -> Quantum Metal params
      -> Ansys Surrogate MLP -> Hamiltonian Reconstruction
      -> Loss (fed back to Inverse MLP)
  and finally: best Quantum Metal design output.

Usage:
    python3 generate_inverse_training_pipeline_figure.py
    # => produces manuscript_exports/inverse_pipeline.svg and manuscript_exports/inverse_pipeline.pdf
"""

import os

from _paths import MANUSCRIPT_EXPORTS_DIR

BG              = "#FFFFFF"

FLOWCHART_COLOR_SCHEME = os.environ.get("FLOWCHART_COLOR_SCHEME", "blue").strip().lower()
FLOWCHART_COLOR_SCHEME = {"current": "blue", "new": "blue", "old": "legacy", "classic": "legacy"}.get(
    FLOWCHART_COLOR_SCHEME,
    FLOWCHART_COLOR_SCHEME,
)
if FLOWCHART_COLOR_SCHEME == "legacy":
    FROST           = "#FFF4E6"  # Physics targets
    PALE_ICE        = "#E8F5E8"  # ML components
    DUSTY_BLUE      = "#E8E4F0"  # Validation
    FROST_DARK      = "#E87A00"
    PALE_ICE_DARK   = "#3D8B3D"
    DUSTY_BLUE_DARK = "#7B68AE"
else:
    FROST           = "#D6E5EE"  # Physics targets
    PALE_ICE        = "#B0CCDE"  # ML components
    DUSTY_BLUE      = "#8AABC8"  # Validation
    FROST_DARK      = "#567A90"
    PALE_ICE_DARK   = "#3F6F8B"
    DUSTY_BLUE_DARK = "#17384F"

# Outer "Training" container stays light so the training stages stand out.
TRAIN_FILL      = "#FFFFFF"
TRAIN_STROKE    = PALE_ICE_DARK
TRAIN_LABEL     = PALE_ICE_DARK

INPUT_FILL      = FROST
INPUT_STROKE    = FROST_DARK
INPUT_TEXT      = "#222222"

OUTPUT_FILL     = "#FFFFFF"
OUTPUT_STROKE   = PALE_ICE_DARK
OUTPUT_TEXT     = "#222222"

INV_FILL        = PALE_ICE
INV_STROKE      = PALE_ICE_DARK
INV_TEXT        = "#222222"

# Intermediate data node is the learned model output.
PARAM_FILL      = PALE_ICE
PARAM_STROKE    = PALE_ICE_DARK
PARAM_TEXT      = "#333333"

# Surrogate MLP is an ML component.
SURR_FILL       = PALE_ICE
SURR_STROKE     = PALE_ICE_DARK
SURR_TEXT       = "#333333"
SURR_TITLE      = PALE_ICE_DARK

# Reconstruction is part of the training graph in Fig. 1B.
RECON_FILL      = PALE_ICE
RECON_STROKE    = PALE_ICE_DARK
RECON_TEXT      = "#333333"
RECON_TITLE     = PALE_ICE_DARK

LOSS_FILL       = DUSTY_BLUE_DARK
LOSS_STROKE     = DUSTY_BLUE_DARK
LOSS_BOX_TEXT   = "#FFFFFF"

# Arrows
ARROW_MAIN      = "#555555"   # neutral grey arrows on the forward path
ARROW_OUT       = "#555555"
FEEDBACK        = PALE_ICE_DARK

# Loss text
LOSS_TEXT       = "#333333"
LOSS_HL_IN      = FROST_DARK
LOSS_HL_OUT     = DUSTY_BLUE_DARK

SVG = f"""<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 560 550"
     font-family="'Helvetica Neue', Arial, Helvetica, sans-serif">
<rect width="560" height="550" fill="{BG}"/>
<defs>
    <marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{ARROW_MAIN}"/>
    </marker>
    <marker id="arrowOut" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{ARROW_OUT}"/>
    </marker>
    <marker id="arrowFb" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{FEEDBACK}"/>
    </marker>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="2"/>
      <feOffset dx="0" dy="2" result="offsetblur"/>
      <feComponentTransfer>
        <feFuncA type="linear" slope="0.25"/>
      </feComponentTransfer>
      <feMerge>
        <feMergeNode/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>


<rect x="138" y="20" width="284" height="48" rx="10" ry="10"
        fill="{INPUT_FILL}" stroke="{INPUT_STROKE}" stroke-width="2"
        filter="url(#softShadow)"/>
  <text x="280" y="50" text-anchor="middle"
        font-size="17" font-weight="bold" fill="{INPUT_TEXT}">
    Desired Hamiltonian Input
  </text>


<rect x="64" y="82" width="432" height="400" rx="16" ry="16"
        fill="{TRAIN_FILL}" stroke="{TRAIN_STROKE}" stroke-width="2.5"
        stroke-dasharray="8,4"
        filter="url(#softShadow)"/>
<text x="480" y="108" text-anchor="end"
        font-size="15" font-style="italic" font-weight="bold"
        fill="{TRAIN_LABEL}">Training</text>

<line x1="280" y1="72" x2="280" y2="88"
        stroke="{ARROW_OUT}" stroke-width="3.2"/>
<polygon points="280,98 273,86 287,86" fill="{ARROW_OUT}"/>

<rect x="210" y="96" width="140" height="54" rx="10" ry="10"
        fill="{INV_FILL}" stroke="{INV_STROKE}" stroke-width="2"/>
  <text x="280" y="119" text-anchor="middle"
        font-size="16" font-weight="bold" fill="{INV_TEXT}">Inverse</text>
  <text x="280" y="138" text-anchor="middle"
        font-size="16" font-weight="bold" fill="{INV_TEXT}">MLP</text>

<line x1="280" y1="154" x2="280" y2="164"
        stroke="{ARROW_MAIN}" stroke-width="3.2"/>
<polygon points="280,174 273,162 287,162" fill="{ARROW_MAIN}"/>

<rect x="122" y="174" width="316" height="46" rx="10" ry="10"
        fill="{PARAM_FILL}" stroke="{PARAM_STROKE}" stroke-width="2"/>
  <text x="280" y="202" text-anchor="middle"
        font-size="16" font-weight="bold" fill="{PARAM_TEXT}">
    Best Quantum Metal Parameter Guess
  </text>

<line x1="280" y1="224" x2="280" y2="234"
        stroke="{ARROW_MAIN}" stroke-width="3.2"/>
<polygon points="280,244 273,232 287,232" fill="{ARROW_MAIN}"/>

<rect x="172" y="244" width="216" height="58" rx="10" ry="10"
        fill="{SURR_FILL}" stroke="{SURR_STROKE}" stroke-width="2"/>
  <text x="280" y="267" text-anchor="middle"
        font-size="16" font-weight="bold" fill="{SURR_TEXT}">Ansys Surrogate</text>
  <text x="280" y="286" text-anchor="middle"
        font-size="16" font-weight="bold" fill="{SURR_TEXT}">MLP</text>

<line x1="280" y1="306" x2="280" y2="316"
        stroke="{ARROW_MAIN}" stroke-width="3.2"/>
<polygon points="280,326 273,314 287,314" fill="{ARROW_MAIN}"/>

<rect x="172" y="326" width="216" height="58" rx="10" ry="10"
        fill="{RECON_FILL}" stroke="{RECON_STROKE}" stroke-width="2"/>
  <text x="280" y="349" text-anchor="middle"
        font-size="16" font-weight="bold" fill="{RECON_TEXT}">Hamiltonian</text>
  <text x="280" y="368" text-anchor="middle"
        font-size="16" font-weight="bold" fill="{RECON_TEXT}">Reconstruction</text>

<line x1="280" y1="388" x2="280" y2="393"
        stroke="{ARROW_MAIN}" stroke-width="3.2"/>
<polygon points="280,403 273,391 287,391" fill="{ARROW_MAIN}"/>

<rect x="126" y="405" width="308" height="72" rx="10" ry="10"
        fill="{LOSS_FILL}" stroke="{LOSS_STROKE}" stroke-width="2"/>
  <text x="280" y="425" text-anchor="middle"
        font-size="16" font-weight="bold" fill="{LOSS_BOX_TEXT}">Compute loss</text>
  <text x="280" y="446" text-anchor="middle"
        font-size="13" fill="{LOSS_BOX_TEXT}">Average absolute difference between</text>
  <text x="280" y="463" text-anchor="middle"
        font-size="13" fill="{LOSS_BOX_TEXT}">target and reconstructed Hamiltonian</text>

<line x1="280" y1="481" x2="280" y2="491"
        stroke="{ARROW_OUT}" stroke-width="3.2"/>
<polygon points="280,501 273,489 287,489" fill="{ARROW_OUT}"/>

<path d="M 126,441 L 104,441 L 104,123 L 198,123"
        fill="none" stroke="{FEEDBACK}" stroke-width="3.0"
        stroke-dasharray="7,4"/>
<polygon points="210,123 198,116 198,130" fill="{FEEDBACK}"/>
  <text x="96" y="275" text-anchor="middle"
        font-size="14" font-style="italic" fill="{FEEDBACK}"
        transform="rotate(-90 96 275)">
    update inverse weights
  </text>

<rect x="120" y="499" width="320" height="46" rx="10" ry="10"
        fill="{OUTPUT_FILL}" stroke="{OUTPUT_STROKE}" stroke-width="2"
        filter="url(#softShadow)"/>
  <text x="280" y="528" text-anchor="middle"
        font-size="17" font-weight="bold" fill="{OUTPUT_TEXT}">
    Best Quantum Metal Design Output
  </text>

</svg>
"""

inverse_pipeline_svg = MANUSCRIPT_EXPORTS_DIR / "inverse_pipeline.svg"
inverse_pipeline_pdf = MANUSCRIPT_EXPORTS_DIR / "inverse_pipeline.pdf"

with inverse_pipeline_svg.open("w", encoding="utf-8") as f:
    f.write(SVG)
print(f"Written {inverse_pipeline_svg}")

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import io

drawing = svg2rlg(io.StringIO(SVG))
renderPDF.drawToFile(drawing, str(inverse_pipeline_pdf))
print(f"Written {inverse_pipeline_pdf}")
