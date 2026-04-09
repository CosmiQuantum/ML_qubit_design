#!/usr/bin/env python3
"""
Generate the Inverse Design Training Pipeline SVG figure.

This produces an overall flowchart of the inverse-design training loop:
  Desired Hamiltonian -> Inverse MLP -> Qiskit Metal params
      -> Ansys Surrogate MLP -> Hamiltonian Reconstruction
      -> Loss (fed back to Inverse MLP)
  and finally: best Qiskit Metal design output.

Usage:
    python3 generate_inverse_pipeline_svg.py
    # => produces inverse_pipeline.svg
"""

# ---- Color palette (aligned with the Testing Pipeline figure) ----------
# White background, green as the primary highlight color, pale grey
# utility boxes, and orange/purple accents matching the lower-section
# Qubit/CPW subsystem annotations in the reference figure.
BG              = "#FFFFFF"

# Green primary (matches "ML Model predictions" / pyEPR boxes)
GREEN           = "#3D8B3D"
GREEN_LIGHT     = "#E8F5E8"
GREEN_DARK      = "#2E6B2E"

# Outer "Training" container — light green fill, green dashed border
TRAIN_FILL      = GREEN_LIGHT
TRAIN_STROKE    = GREEN
TRAIN_LABEL     = GREEN_DARK

# Endpoint boxes (input / output) — same green style as ML Model box
ENDPOINT_FILL   = GREEN_LIGHT
ENDPOINT_STROKE = GREEN
ENDPOINT_TEXT   = "#222222"

# Inverse MLP node — the "active/highlighted" node (solid green, like pyEPR)
INV_FILL        = GREEN
INV_STROKE      = GREEN_DARK
INV_TEXT        = "#FFFFFF"

# Intermediate data node (Qiskit Metal params) — neutral grey utility box
PARAM_FILL      = "#E8E8E8"
PARAM_STROKE    = "#999999"
PARAM_TEXT      = "#333333"

# Surrogate MLP (forward) — orange, matching Qubit Subsystem accent
SURR_FILL       = "#FFF4E6"
SURR_STROKE     = "#E87A00"
SURR_TEXT       = "#333333"
SURR_TITLE      = "#E87A00"

# Reconstruction node — purple, matching CPW Cavity Subsystem accent
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
LOSS_HL_IN      = "#E87A00"   # orange — ties to "input" side (Qubit accent)
LOSS_HL_OUT     = "#7B68AE"   # purple — ties to "reconstruction" side (Cavity accent)

SVG = f"""<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 560 760"
     font-family="'Helvetica Neue', Arial, Helvetica, sans-serif">

  <!-- Background -->
  <rect width="560" height="760" fill="{BG}"/>

  <!-- Arrow marker definitions -->
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

  <!-- ============================================================== -->
  <!-- TOP: Desired Hamiltonian Input                                  -->
  <!-- ============================================================== -->
  <rect x="140" y="24" width="280" height="52" rx="10" ry="10"
        fill="{ENDPOINT_FILL}" stroke="{ENDPOINT_STROKE}" stroke-width="2"
        filter="url(#softShadow)"/>
  <text x="280" y="56" text-anchor="middle"
        font-size="17" font-weight="bold" fill="{ENDPOINT_TEXT}">
    Desired Hamiltonian Input
  </text>

  <!-- Arrow into training container -->
  <line x1="280" y1="80" x2="280" y2="112"
        stroke="{ARROW_OUT}" stroke-width="2.5"
        marker-end="url(#arrowOut)"/>

  <!-- ============================================================== -->
  <!-- TRAINING CONTAINER                                              -->
  <!-- ============================================================== -->
  <rect x="60" y="116" width="440" height="520" rx="18" ry="18"
        fill="{TRAIN_FILL}" stroke="{TRAIN_STROKE}" stroke-width="2.5"
        stroke-dasharray="8,4"
        filter="url(#softShadow)"/>

  <!-- "Training" label, top-right of container -->
  <text x="480" y="146" text-anchor="end"
        font-size="16" font-style="italic" font-weight="bold"
        fill="{TRAIN_LABEL}">Training</text>

  <!-- ── Inverse MLP node ───────────────────────────────────────── -->
  <rect x="210" y="150" width="140" height="60" rx="10" ry="10"
        fill="{INV_FILL}" stroke="{INV_STROKE}" stroke-width="2"/>
  <text x="280" y="176" text-anchor="middle"
        font-size="16" font-weight="bold" fill="{INV_TEXT}">Inverse</text>
  <text x="280" y="196" text-anchor="middle"
        font-size="16" font-weight="bold" fill="{INV_TEXT}">MLP</text>

  <!-- Arrow: Inverse MLP -> Qiskit params -->
  <line x1="280" y1="212" x2="280" y2="244"
        stroke="{ARROW_MAIN}" stroke-width="2.5"
        marker-end="url(#arrowMain)"/>

  <!-- ── Best Qiskit Metal Parameter guess ──────────────────────── -->
  <rect x="120" y="248" width="320" height="52" rx="10" ry="10"
        fill="{PARAM_FILL}" stroke="{PARAM_STROKE}" stroke-width="2"/>
  <text x="280" y="280" text-anchor="middle"
        font-size="16" font-weight="bold" fill="{PARAM_TEXT}">
    Best Qiskit Metal Parameter guess
  </text>

  <!-- Arrow: params -> Ansys Surrogate -->
  <line x1="280" y1="302" x2="280" y2="334"
        stroke="{ARROW_MAIN}" stroke-width="2.5"
        marker-end="url(#arrowMain)"/>

  <!-- ── Ansys Surrogate MLP ────────────────────────────────────── -->
  <rect x="170" y="338" width="220" height="68" rx="10" ry="10"
        fill="{SURR_FILL}" stroke="{SURR_STROKE}" stroke-width="2"/>
  <text x="280" y="365" text-anchor="middle"
        font-size="16" font-weight="bold" fill="{SURR_TEXT}">Ansys Surrogate</text>
  <text x="280" y="388" text-anchor="middle"
        font-size="16" font-weight="bold" fill="{SURR_TEXT}">MLP</text>

  <!-- Arrow: Surrogate -> Reconstruction -->
  <line x1="280" y1="408" x2="280" y2="440"
        stroke="{ARROW_MAIN}" stroke-width="2.5"
        marker-end="url(#arrowMain)"/>

  <!-- ── Hamiltonian Reconstruction ─────────────────────────────── -->
  <rect x="170" y="444" width="220" height="68" rx="10" ry="10"
        fill="{RECON_FILL}" stroke="{RECON_STROKE}" stroke-width="2"/>
  <text x="280" y="471" text-anchor="middle"
        font-size="16" font-weight="bold" fill="{RECON_TEXT}">Hamiltonian</text>
  <text x="280" y="494" text-anchor="middle"
        font-size="16" font-weight="bold" fill="{RECON_TEXT}">Reconstruction</text>

  <!-- ── Loss line ─────────────────────────────────────────────── -->
  <text x="100" y="548" text-anchor="start"
        font-size="14" fill="{LOSS_TEXT}">
    Loss = MAE(<tspan font-weight="bold" fill="{LOSS_HL_IN}">Hamiltonian Input</tspan>, <tspan font-weight="bold" fill="{LOSS_HL_OUT}">Hamiltonian</tspan>
  </text>
  <text x="100" y="568" text-anchor="start"
        font-size="14" font-weight="bold" fill="{LOSS_HL_OUT}">
    Reconstructed<tspan font-weight="normal" fill="{LOSS_TEXT}"> )</tspan>
  </text>

  <!-- ── Feedback loop: from Reconstruction/Loss back to Inverse MLP ── -->
  <!-- Path: start at left side of Reconstruction box, go left, up, and
       into the left side of the Inverse MLP box -->
  <path d="M 170,478 L 110,478 L 110,180 L 210,180"
        fill="none" stroke="{FEEDBACK}" stroke-width="2.5"
        stroke-dasharray="6,4" marker-end="url(#arrowFb)"/>
  <text x="102" y="330" text-anchor="end"
        font-size="12" font-style="italic" fill="{FEEDBACK}"
        transform="rotate(-90 102 330)">
    backprop / update
  </text>

  <!-- ============================================================== -->
  <!-- BOTTOM: Best Qiskit Metal design output                         -->
  <!-- ============================================================== -->

  <!-- Arrow out of container -->
  <line x1="280" y1="640" x2="280" y2="672"
        stroke="{ARROW_OUT}" stroke-width="2.5"
        marker-end="url(#arrowOut)"/>

  <rect x="120" y="676" width="320" height="52" rx="10" ry="10"
        fill="{ENDPOINT_FILL}" stroke="{ENDPOINT_STROKE}" stroke-width="2"
        filter="url(#softShadow)"/>
  <text x="280" y="708" text-anchor="middle"
        font-size="17" font-weight="bold" fill="{ENDPOINT_TEXT}">
    Best Qiskit Metal Design Output
  </text>

</svg>
"""

with open("paper_plots/inverse_pipeline.svg", "w") as f:
    f.write(SVG)
print("Written inverse_pipeline.svg")

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import io

drawing = svg2rlg(io.StringIO(SVG))
renderPDF.drawToFile(drawing, "paper_plots/inverse_pipeline.pdf")
print("Written inverse_pipeline.pdf")
