#!/usr/bin/env python3
"""
Generate the Transmon–Resonator System figure.

Composites the fragments.pdf vector image with SVG annotation overlays
(leader lines + parameter boxes) into a single PDF.

Output:
    paper_plots/pipeline_updated.pdf
"""
from pathlib import Path
import cairosvg
import fitz  # pymupdf

FIG_ORANGE = "#E87A00"
FIG_PURPLE = "#7B68AE"

OUT_DIR = Path("paper_plots")
FRAGMENTS_PDF = OUT_DIR / "fragments.pdf"
OVERLAY_SVG = OUT_DIR / "pipeline_overlay.svg"
OVERLAY_PDF = OUT_DIR / "pipeline_overlay.pdf"
FINAL_PDF = OUT_DIR / "pipeline_updated.pdf"

PAGE_W = 1120
PAGE_H = 640

# Where the fragments.pdf page should appear
FIG_X = 85
FIG_Y = 50
FIG_W = 950
FIG_H = 450

SVG_TEMPLATE = """\
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 %(PAGE_W)s %(PAGE_H)s"
     width="%(PAGE_W)s" height="%(PAGE_H)s"
     font-family="Arial, Helvetica, sans-serif">

  <!-- transparent background -->

  <text x="560" y="32"
        text-anchor="middle" font-size="22" font-weight="bold" fill="#333">
    Transmon&#x2013;Resonator System: Parameter Identification
  </text>

  <!-- leader lines from figure to annotation boxes -->
  <line x1="269" y1="307" x2="157" y2="552"
        stroke="%(FIG_ORANGE)s" stroke-width="1.5" stroke-dasharray="5,3"/>
  <line x1="399" y1="307" x2="433" y2="552"
        stroke="%(FIG_ORANGE)s" stroke-width="1.5" stroke-dasharray="5,3"/>
  <line x1="750" y1="338" x2="709" y2="552"
        stroke="%(FIG_PURPLE)s" stroke-width="1.5" stroke-dasharray="5,3"/>
  <line x1="850" y1="298" x2="979" y2="552"
        stroke="%(FIG_PURPLE)s" stroke-width="1.5" stroke-dasharray="5,3"/>

  <!-- annotation boxes -->
  <rect x="22" y="552" width="270" height="62" rx="5" ry="5"
        fill="#FFF4E6" stroke="%(FIG_ORANGE)s" stroke-width="1.5"/>
  <text x="157" y="576" text-anchor="middle"
        font-size="16" font-weight="bold" fill="%(FIG_ORANGE)s">
    &#x03B1; (anharmonicity)
  </text>
  <text x="157" y="600" text-anchor="middle"
        font-size="16" fill="#555">&#x2248; &#x2212;E_C from Q3D</text>

  <rect x="298" y="552" width="270" height="62" rx="5" ry="5"
        fill="#FFF4E6" stroke="%(FIG_ORANGE)s" stroke-width="1.5"/>
  <text x="433" y="576" text-anchor="middle"
        font-size="16" font-weight="bold" fill="%(FIG_ORANGE)s">
    f_qubit (qubit frequency)
  </text>
  <text x="433" y="600" text-anchor="middle"
        font-size="16" fill="#555">&#x2248; &#x221A;(8 E_J E_C) &#x2212; E_C from Q3D</text>

  <rect x="574" y="552" width="270" height="62" rx="5" ry="5"
        fill="#E8E4F0" stroke="%(FIG_PURPLE)s" stroke-width="1.5"/>
  <text x="709" y="576" text-anchor="middle"
        font-size="16" font-weight="bold" fill="%(FIG_PURPLE)s">
    f_res (resonator freq.)
  </text>
  <text x="709" y="600" text-anchor="middle"
        font-size="16" fill="#555">HFSS eigenmode solver</text>

  <rect x="850" y="552" width="258" height="62" rx="5" ry="5"
        fill="#E8E4F0" stroke="%(FIG_PURPLE)s" stroke-width="1.5"/>
  <text x="979" y="576" text-anchor="middle"
        font-size="16" font-weight="bold" fill="%(FIG_PURPLE)s">
    &#x03BA; (resonator linewidth)
  </text>
  <text x="979" y="600" text-anchor="middle"
        font-size="16" fill="#555">HFSS via 50&#x2126; port loss</text>
</svg>
"""

svg = SVG_TEMPLATE % {
    "PAGE_W": PAGE_W,
    "PAGE_H": PAGE_H,
    "FIG_ORANGE": FIG_ORANGE,
    "FIG_PURPLE": FIG_PURPLE,
}

OVERLAY_SVG.write_text(svg, encoding="utf-8")
print(f"Written {OVERLAY_SVG}")

cairosvg.svg2pdf(
    bytestring=svg.encode("utf-8"),
    write_to=str(OVERLAY_PDF),
)
print(f"Written {OVERLAY_PDF}")

fragments_doc = fitz.open(str(FRAGMENTS_PDF))
overlay_doc = fitz.open(str(OVERLAY_PDF))

final_doc = fitz.open()
page = final_doc.new_page(width=PAGE_W, height=PAGE_H)

# Layer 1: white background
page.draw_rect(fitz.Rect(0, 0, PAGE_W, PAGE_H), color=None, fill=(1, 1, 1))

# Layer 2: fragments.pdf image placed in the target rectangle
target_rect = fitz.Rect(FIG_X, FIG_Y, FIG_X + FIG_W, FIG_Y + FIG_H)
page.show_pdf_page(target_rect, fragments_doc, 0, keep_proportion=True, overlay=True)

# Layer 3: SVG overlay (title, leaders, annotation boxes) on top
full_rect = fitz.Rect(0, 0, PAGE_W, PAGE_H)
page.show_pdf_page(full_rect, overlay_doc, 0, keep_proportion=True, overlay=True)

final_doc.save(str(FINAL_PDF))
print(f"Written {FINAL_PDF}")