#!/usr/bin/env python3
"""
Generate the Testing Pipeline (Forward Pass) figure as a standalone PDF.

This shows the forward-validation tool chain:
    ML Model → SQuADDS → Quantum Metal → pyEPR ↔ PyAEDT ↔ Ansys HFSS & Q3D

with the validation-loss cross comparing Reference (SQuADDS dataset)
vs Predicted (forward-pass simulation) results.

Output:
    manuscript_exports/testing_pipeline.pdf
"""
import os

import cairosvg

from _paths import MANUSCRIPT_EXPORTS_DIR

FLOWCHART_COLOR_SCHEME = os.environ.get("FLOWCHART_COLOR_SCHEME", "blue").strip().lower()
FLOWCHART_COLOR_SCHEME = {"current": "blue", "new": "blue", "old": "legacy", "classic": "legacy"}.get(
    FLOWCHART_COLOR_SCHEME,
    FLOWCHART_COLOR_SCHEME,
)
if FLOWCHART_COLOR_SCHEME == "legacy":
    FROST = "#FFF4E6"  # Physics targets
    PALE_ICE = "#E8F5E8"  # ML components
    DUSTY_BLUE = "#E8E4F0"  # Validation
    FROST_DARK = "#E87A00"
    ML_STROKE = "#3D8B3D"
    VALID_STROKE = "#7B68AE"
else:
    FROST = "#D6E5EE"  # Physics targets
    PALE_ICE = "#B0CCDE"  # ML components
    DUSTY_BLUE = "#8AABC8"  # Validation
    FROST_DARK = "#567A90"
    ML_STROKE = "#3F6F8B"
    VALID_STROKE = "#17384F"
OUT_PDF = MANUSCRIPT_EXPORTS_DIR / "testing_pipeline.pdf"

# The figure is placed at \textwidth in a figure* (spans both columns). On-page
# font size scales as font_svg * (\textwidth / PAGE_W), so a tight canvas keeps
# the labels legible. Fonts are enlarged and box padding trimmed so the canvas
# stays compact (bigger text at the same printed width).
PAGE_W = 1030
PAGE_H = 362

SVG_TEMPLATE = """\
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 %(PAGE_W)s %(PAGE_H)s"
     width="%(PAGE_W)s" height="%(PAGE_H)s"
     font-family="Arial, Helvetica, sans-serif">
  <rect width="%(PAGE_W)s" height="%(PAGE_H)s" fill="white"/>
  <text x="515" y="36"
        text-anchor="middle" font-size="22" font-weight="bold" fill="#222"
        >Testing Pipeline (Forward Pass)</text>

  <!-- Validation-loss comparison node -->
  <text x="430" y="80"
        text-anchor="middle" font-size="17" font-weight="bold"
        fill="%(VALID_STROKE)s">Validation loss</text>
  <circle cx="430" cy="112" r="14" fill="white"
          stroke="%(VALID_STROKE)s" stroke-width="2.5"/>
  <line x1="416" y1="112" x2="444" y2="112"
        stroke="%(VALID_STROKE)s" stroke-width="2.5"/>
  <line x1="430" y1="98" x2="430" y2="126"
        stroke="%(VALID_STROKE)s" stroke-width="2.5"/>
  <text x="235" y="132"
        text-anchor="middle" font-size="17" font-weight="bold"
        fill="%(VALID_STROKE)s">Reference</text>
  <text x="235" y="149"
        text-anchor="middle" font-size="17" fill="#555">SQuADDS dataset results</text>
  <text x="690" y="132"
        text-anchor="middle" font-size="17" font-weight="bold"
        fill="%(VALID_STROKE)s">Predicted</text>
  <text x="690" y="149"
        text-anchor="middle" font-size="17" fill="#555">forward pass results</text>
  <path d="M 98,178 L 98,112 L 414,112"
        fill="none" stroke="%(VALID_STROKE)s" stroke-width="2.5"/>
  <polygon points="414,112 405,107 405,117" fill="%(VALID_STROKE)s"/>
  <path d="M 610,178 L 610,112 L 446,112"
        fill="none" stroke="%(VALID_STROKE)s" stroke-width="2.5"/>
  <polygon points="446,112 455,107 455,117" fill="%(VALID_STROKE)s"/>

  <!-- ML model output -->
  <rect x="20" y="178" width="156" height="56" rx="4" ry="4"
        fill="%(PALE_ICE)s" stroke="%(ML_STROKE)s" stroke-width="2.5"/>
  <text x="98" y="202" text-anchor="middle"
        font-size="18" font-weight="bold" fill="#333">ML Model</text>
  <text x="98" y="224" text-anchor="middle"
        font-size="18" font-weight="bold" fill="#333">predictions</text>
  <text x="98" y="256" text-anchor="middle"
        font-size="17" fill="#555" font-style="italic">Layout with target</text>
  <text x="98" y="275" text-anchor="middle"
        font-size="17" fill="#555" font-style="italic">design params for</text>
  <text x="98" y="294" text-anchor="middle"
        font-size="17" fill="#555" font-style="italic">transmon</text>
  <text x="98" y="313" text-anchor="middle"
        font-size="17" fill="#555" font-style="italic">system:</text>
  <text x="98" y="337" text-anchor="middle"
        font-size="17" font-weight="bold" font-style="italic" fill="%(FROST_DARK)s"
        >&#x03B1;, f_qubit</text>

  <!-- Python tool chain -->
  <rect x="192" y="153" width="816" height="112" rx="5" ry="5"
        fill="none" stroke="#888" stroke-width="2" stroke-dasharray="8,4"/>
  <text x="210" y="173"
        font-size="15" fill="#888" font-style="italic" font-weight="bold">Python</text>
  <line x1="177" y1="206" x2="207" y2="206"
        stroke="%(ML_STROKE)s" stroke-width="2.5"/>
  <polygon points="207,206 198,201 198,211" fill="%(ML_STROKE)s"/>

  <rect x="209" y="182" width="120" height="48" rx="4" ry="4"
        fill="%(DUSTY_BLUE)s" stroke="%(VALID_STROKE)s" stroke-width="1.5"/>
  <text x="269" y="212" text-anchor="middle"
        font-size="18" font-weight="bold" fill="#333">SQuADDS</text>
  <line x1="329" y1="206" x2="357" y2="206" stroke="#777" stroke-width="2"/>
  <polygon points="357,206 349,201 349,211" fill="#777"/>

  <rect x="359" y="182" width="170" height="48" rx="4" ry="4"
        fill="%(DUSTY_BLUE)s" stroke="%(VALID_STROKE)s" stroke-width="1.5"/>
  <text x="444" y="212" text-anchor="middle"
        font-size="18" font-weight="bold" fill="#333">Quantum Metal</text>
  <line x1="529" y1="206" x2="557" y2="206" stroke="#777" stroke-width="2"/>
  <polygon points="557,206 549,201 549,211" fill="#777"/>

  <rect x="559" y="182" width="102" height="48" rx="4" ry="4"
        fill="%(DUSTY_BLUE)s" stroke="%(VALID_STROKE)s" stroke-width="2.5"/>
  <text x="610" y="212" text-anchor="middle"
        font-size="18" font-weight="bold" fill="#333">pyEPR</text>
  <path d="M 269,182 C 269,156 610,156 610,182"
        fill="none" stroke="#777" stroke-width="1.5" stroke-dasharray="6,3"/>
  <polygon points="610,182 605,174 615,174" fill="#777"/>

  <line x1="661" y1="199" x2="711" y2="199" stroke="#777" stroke-width="1.5"/>
  <polygon points="711,199 703,195 703,203" fill="#777"/>
  <line x1="711" y1="213" x2="661" y2="213"
        stroke="%(VALID_STROKE)s" stroke-width="1.5"/>
  <polygon points="661,213 669,209 669,217" fill="%(VALID_STROKE)s"/>

  <rect x="713" y="182" width="114" height="48" rx="4" ry="4"
        fill="%(DUSTY_BLUE)s" stroke="%(VALID_STROKE)s" stroke-width="1.5"/>
  <text x="770" y="212" text-anchor="middle"
        font-size="18" font-weight="bold" fill="#333">PyAEDT</text>

  <line x1="827" y1="199" x2="877" y2="199" stroke="#777" stroke-width="1.5"/>
  <polygon points="877,199 869,195 869,203" fill="#777"/>
  <line x1="877" y1="213" x2="827" y2="213" stroke="#777" stroke-width="1.5"/>
  <polygon points="827,213 835,209 835,217" fill="%(VALID_STROKE)s"/>

  <rect x="879" y="182" width="122" height="48" rx="4" ry="4"
        fill="%(DUSTY_BLUE)s" stroke="%(VALID_STROKE)s" stroke-width="1.5"/>
  <text x="940" y="212" text-anchor="middle"
        font-size="18" font-weight="bold" fill="#333">EM solver</text>
</svg>"""

svg = SVG_TEMPLATE % {
    "PAGE_W": PAGE_W,
    "PAGE_H": PAGE_H,
    "PALE_ICE": PALE_ICE,
    "DUSTY_BLUE": DUSTY_BLUE,
    "FROST_DARK": FROST_DARK,
    "ML_STROKE": ML_STROKE,
    "VALID_STROKE": VALID_STROKE,
}

cairosvg.svg2pdf(
    bytestring=svg.encode("utf-8"),
    write_to=str(OUT_PDF),
)
print(f"Written {OUT_PDF}")
