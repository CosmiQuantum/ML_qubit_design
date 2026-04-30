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
    # => produces outputs/inverse_pipeline.svg and outputs/inverse_pipeline.pdf
"""

from _paths import OUTPUTS_DIR

# Color palette (aligned with the Testing Pipeline figure)
# White background, green as the primary highlight color, pale grey
# utility boxes, and orange/purple accents matching the lowersection
# Qubit/CPW subsystem annotations in the reference figure.
BG              = "#FFFFFF"

# Green primary (matches "ML Model predictions" / pyEPR boxes)
GREEN           = "#3D8B3D"
GREEN_LIGHT     = "#E8F5E8"
GREEN_DARK      = "#2E6B2E"

# Outer "Training" container light green fill, green dashed border
TRAIN_FILL      = GREEN_LIGHT
TRAIN_STROKE    = GREEN
TRAIN_LABEL     = GREEN_DARK

# Endpoint boxes (input / output) same green style as ML Model box
ENDPOINT_FILL   = GREEN_LIGHT
ENDPOINT_STROKE = GREEN
ENDPOINT_TEXT   = "#222222"

# Inverse MLP node the "active/highlighted" node (solid green, like pyEPR)
INV_FILL        = GREEN
INV_STROKE      = GREEN_DARK
INV_TEXT        = "#FFFFFF"

# Intermediate data node (Quantum Metal params) neutral grey utility box
PARAM_FILL      = "#E8E8E8"
PARAM_STROKE    = "#999999"
PARAM_TEXT      = "#333333"

# Surrogate MLP (forward) orange, matching Qubit Subsystem accent
SURR_FILL       = "#FFF4E6"
SURR_STROKE     = "#E87A00"
SURR_TEXT       = "#333333"
SURR_TITLE      = "#E87A00"

# Reconstruction node purple, matching CPW Cavity Subsystem accent
RECON_FILL      = "#E8E4F0"
RECON_STROKE    = "#7B68AE"
RECON_TEXT      = "#333333"
RECON_TITLE     = "#7B68AE"

# Arrows
ARROW_MAIN      = "#555555"   # neutral grey arrows on the forward path
ARROW_OUT       = "#555555"
FEEDBACK        = GREEN       # green feedback loop (primary highlight)

# Loss text
LOSS_TEXT       = "#333333"
LOSS_HL_IN      = "#E87A00"   # orange ties to "input" side (Qubit accent)
LOSS_HL_OUT     = "#7B68AE"   # purple ties to "reconstruction" side (Cavity accent)

SVG = f"""<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 560 590"
     font-family="'Helvetica Neue', Arial, Helvetica, sans-serif">
<rect width="560" height="590" fill="{BG}"/>
<defs>
    <marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{ARROW_MAIN}"/>
    </marker>
    <marker id="arrowOut" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{ARROW_OUT}"/>
    </marker>
    <marker id="arrowFb" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
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
        fill="{ENDPOINT_FILL}" stroke="{ENDPOINT_STROKE}" stroke-width="2"
        filter="url(#softShadow)"/>
  <text x="280" y="50" text-anchor="middle"
        font-size="14" font-weight="bold" fill="{ENDPOINT_TEXT}">
    Desired Hamiltonian Input
  </text>

<line x1="280" y1="72" x2="280" y2="92"
        stroke="{ARROW_OUT}" stroke-width="2.5"
        marker-end="url(#arrowOut)"/>

<rect x="64" y="88" width="432" height="410" rx="16" ry="16"
        fill="{TRAIN_FILL}" stroke="{TRAIN_STROKE}" stroke-width="2.5"
        stroke-dasharray="8,4"
        filter="url(#softShadow)"/>
<text x="480" y="116" text-anchor="end"
        font-size="13" font-style="italic" font-weight="bold"
        fill="{TRAIN_LABEL}">Training</text>

<rect x="210" y="106" width="140" height="54" rx="10" ry="10"
        fill="{INV_FILL}" stroke="{INV_STROKE}" stroke-width="2"/>
  <text x="280" y="129" text-anchor="middle"
        font-size="14" font-weight="bold" fill="{INV_TEXT}">Inverse</text>
  <text x="280" y="148" text-anchor="middle"
        font-size="14" font-weight="bold" fill="{INV_TEXT}">MLP</text>

<line x1="280" y1="164" x2="280" y2="184"
        stroke="{ARROW_MAIN}" stroke-width="2.5"
        marker-end="url(#arrowMain)"/>

<rect x="122" y="188" width="316" height="46" rx="10" ry="10"
        fill="{PARAM_FILL}" stroke="{PARAM_STROKE}" stroke-width="2"/>
  <text x="280" y="216" text-anchor="middle"
        font-size="14" font-weight="bold" fill="{PARAM_TEXT}">
    Best Quantum Metal Parameter Guess
  </text>

<line x1="280" y1="238" x2="280" y2="258"
        stroke="{ARROW_MAIN}" stroke-width="2.5"
        marker-end="url(#arrowMain)"/>

<rect x="172" y="262" width="216" height="58" rx="10" ry="10"
        fill="{SURR_FILL}" stroke="{SURR_STROKE}" stroke-width="2"/>
  <text x="280" y="285" text-anchor="middle"
        font-size="14" font-weight="bold" fill="{SURR_TEXT}">Ansys Surrogate</text>
  <text x="280" y="304" text-anchor="middle"
        font-size="14" font-weight="bold" fill="{SURR_TEXT}">MLP</text>

<line x1="280" y1="324" x2="280" y2="344"
        stroke="{ARROW_MAIN}" stroke-width="2.5"
        marker-end="url(#arrowMain)"/>

<rect x="172" y="348" width="216" height="58" rx="10" ry="10"
        fill="{RECON_FILL}" stroke="{RECON_STROKE}" stroke-width="2"/>
  <text x="280" y="371" text-anchor="middle"
        font-size="14" font-weight="bold" fill="{RECON_TEXT}">Hamiltonian</text>
  <text x="280" y="390" text-anchor="middle"
        font-size="14" font-weight="bold" fill="{RECON_TEXT}">Reconstruction</text>

<text x="92" y="440" text-anchor="start"
        font-size="12" fill="{LOSS_TEXT}">
    Loss = MAE(<tspan font-weight="bold" fill="{LOSS_HL_IN}">Hamiltonian Input</tspan>, <tspan font-weight="bold" fill="{LOSS_HL_OUT}">Hamiltonian Reconstructed</tspan>)
  </text>

<path d="M 172,377 L 104,377 L 104,133 L 210,133"
        fill="none" stroke="{FEEDBACK}" stroke-width="2.5"
        stroke-dasharray="6,4" marker-end="url(#arrowFb)"/>
  <text x="96" y="255" text-anchor="middle"
        font-size="12" font-style="italic" fill="{FEEDBACK}"
        transform="rotate(-90 96 255)">
    backprop / update
  </text>

<line x1="280" y1="502" x2="280" y2="522"
        stroke="{ARROW_OUT}" stroke-width="2.5"
        marker-end="url(#arrowOut)"/>

<rect x="120" y="526" width="320" height="46" rx="10" ry="10"
        fill="{ENDPOINT_FILL}" stroke="{ENDPOINT_STROKE}" stroke-width="2"
        filter="url(#softShadow)"/>
  <text x="280" y="555" text-anchor="middle"
        font-size="14" font-weight="bold" fill="{ENDPOINT_TEXT}">
    Best Quantum Metal Design Output
  </text>

</svg>
"""

inverse_pipeline_svg = OUTPUTS_DIR / "inverse_pipeline.svg"
inverse_pipeline_pdf = OUTPUTS_DIR / "inverse_pipeline.pdf"

with inverse_pipeline_svg.open("w", encoding="utf-8") as f:
    f.write(SVG)
print(f"Written {inverse_pipeline_svg}")

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import io

drawing = svg2rlg(io.StringIO(SVG))
renderPDF.drawToFile(drawing, str(inverse_pipeline_pdf))
print(f"Written {inverse_pipeline_pdf}")
