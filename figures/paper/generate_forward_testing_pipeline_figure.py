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
PIPELINE_SHIFT = 46

OUT_PDF = MANUSCRIPT_EXPORTS_DIR / "testing_pipeline.pdf"

PAGE_W = 1120
PAGE_H = 340

SVG_TEMPLATE = """\
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 %(PAGE_W)s %(PAGE_H)s"
     width="%(PAGE_W)s" height="%(PAGE_H)s"
     font-family="Arial, Helvetica, sans-serif">
  <rect width="%(PAGE_W)s" height="%(PAGE_H)s" fill="white"/>
<text x="560" y="36"
        text-anchor="middle" font-size="16" font-weight="bold" fill="#222"
        >Testing Pipeline (Forward Pass)</text>

  <g transform="translate(%(PIPELINE_SHIFT)s,0)">
<text x="515" y="74"
          text-anchor="middle" font-size="14" font-weight="bold"
          fill="%(VALID_STROKE)s">Validation loss</text>
    <circle cx="515" cy="105" r="12" fill="white"
            stroke="%(VALID_STROKE)s" stroke-width="2.5"/>
    <line x1="489" y1="105" x2="541" y2="105"
          stroke="%(VALID_STROKE)s" stroke-width="2.5"/>
    <line x1="515" y1="79" x2="515" y2="131"
          stroke="%(VALID_STROKE)s" stroke-width="2.5"/>
<text x="220" y="120"
          text-anchor="middle" font-size="14" font-weight="bold"
          fill="%(VALID_STROKE)s">Reference</text>
    <text x="220" y="138"
          text-anchor="middle" font-size="12" fill="#555">SQuADDS dataset results</text>

<text x="700" y="120"
          text-anchor="middle" font-size="14" font-weight="bold"
          fill="%(VALID_STROKE)s">Predicted</text>
    <text x="700" y="138"
          text-anchor="middle" font-size="12" fill="#555">forward pass results</text>
<path d="M 88,175 L 88,105 L 489,105"
          fill="none" stroke="%(VALID_STROKE)s" stroke-width="2.5"/>
    <polygon points="489,105 480,100 480,110" fill="%(VALID_STROKE)s"/>

    <path d="M 603,175 L 603,105 L 541,105"
          fill="none" stroke="%(VALID_STROKE)s" stroke-width="2.5"/>
    <polygon points="541,105 550,100 550,110" fill="%(VALID_STROKE)s"/>
<rect x="18" y="175" width="140" height="54" rx="4" ry="4"
          fill="%(PALE_ICE)s" stroke="%(ML_STROKE)s" stroke-width="2.5"/>
    <text x="88" y="198" text-anchor="middle"
          font-size="14" font-weight="bold" fill="#333">ML Model</text>
    <text x="88" y="218" text-anchor="middle"
          font-size="14" font-weight="bold" fill="#333">predictions</text>
<text x="88" y="248" text-anchor="middle"
          font-size="13" fill="#555" font-style="italic">Layout with target design</text>
    <text x="88" y="268" text-anchor="middle"
          font-size="13" fill="#555" font-style="italic">params for transmon-</text>
    <text x="88" y="288" text-anchor="middle"
          font-size="13" fill="#555" font-style="italic">resonator system:</text>
    <text x="88" y="310" text-anchor="middle"
          font-size="13" font-weight="bold" font-style="italic" fill="%(FROST_DARK)s"
          >&#x03B1;, f_qubit, f_res, &#x03BA;</text>
<rect x="185" y="153" width="660" height="104" rx="5" ry="5"
          fill="none" stroke="#888" stroke-width="2" stroke-dasharray="8,4"/>
    <text x="202" y="171"
          font-size="13" fill="#888" font-style="italic" font-weight="bold">Python</text>
<line x1="158" y1="202" x2="200" y2="202"
          stroke="%(ML_STROKE)s" stroke-width="2.5"/>
    <polygon points="200,202 191,197 191,207" fill="%(ML_STROKE)s"/>
<rect x="203" y="180" width="130" height="46" rx="4" ry="4"
          fill="%(DUSTY_BLUE)s" stroke="%(VALID_STROKE)s" stroke-width="1.5"/>
    <text x="268" y="208" text-anchor="middle"
          font-size="14" font-weight="bold" fill="#333">SQuADDS</text>
<line x1="333" y1="203" x2="360" y2="203"
          stroke="#777" stroke-width="2"/>
    <polygon points="360,203 352,198 352,208" fill="#777"/>
<rect x="363" y="180" width="150" height="46" rx="4" ry="4"
          fill="%(DUSTY_BLUE)s" stroke="%(VALID_STROKE)s" stroke-width="1.5"/>
    <text x="438" y="208" text-anchor="middle"
          font-size="14" font-weight="bold" fill="#333">Quantum Metal</text>
<line x1="513" y1="203" x2="540" y2="203"
          stroke="#777" stroke-width="2"/>
    <polygon points="540,203 532,198 532,208" fill="#777"/>
<rect x="543" y="180" width="120" height="46" rx="4" ry="4"
          fill="%(DUSTY_BLUE)s" stroke="%(VALID_STROKE)s" stroke-width="2.5"/>
    <text x="603" y="208" text-anchor="middle"
          font-size="14" font-weight="bold" fill="#333">pyEPR</text>
<path d="M 268,180 C 268,151 603,151 603,180"
          fill="none" stroke="#777" stroke-width="1.5" stroke-dasharray="6,3"/>
    <polygon points="603,180 598,172 608,172" fill="#777"/>
<line x1="663" y1="195" x2="710" y2="195"
          stroke="#777" stroke-width="1.5"/>
    <polygon points="710,195 703,191 703,199" fill="#777"/>
    <line x1="710" y1="209" x2="663" y2="209"
          stroke="%(VALID_STROKE)s" stroke-width="1.5"/>
    <polygon points="663,209 670,205 670,213" fill="%(VALID_STROKE)s"/>
<rect x="713" y="180" width="120" height="46" rx="4" ry="4"
          fill="%(DUSTY_BLUE)s" stroke="%(VALID_STROKE)s" stroke-width="1.5"/>
    <text x="773" y="208" text-anchor="middle"
          font-size="14" font-weight="bold" fill="#333">PyAEDT</text>
<line x1="833" y1="195" x2="888" y2="195"
          stroke="#777" stroke-width="1.5"/>
    <polygon points="888,195 881,191 881,199" fill="#777"/>
    <line x1="888" y1="209" x2="833" y2="209"
          stroke="#777" stroke-width="1.5"/>
    <polygon points="833,209 840,205 840,213" fill="%(VALID_STROKE)s"/>
<rect x="891" y="180" width="120" height="46" rx="4" ry="4"
          fill="%(DUSTY_BLUE)s" stroke="%(VALID_STROKE)s" stroke-width="1.5"/>
    <text x="951" y="198" text-anchor="middle"
          font-size="14" font-weight="bold" fill="#333">Ansys</text>
    <text x="951" y="216" text-anchor="middle"
          font-size="14" font-weight="bold" fill="#333">HFSS &amp; Q3D</text>

  </g>
</svg>"""

svg = SVG_TEMPLATE % {
    "PAGE_W": PAGE_W,
    "PAGE_H": PAGE_H,
    "PIPELINE_SHIFT": PIPELINE_SHIFT,
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
