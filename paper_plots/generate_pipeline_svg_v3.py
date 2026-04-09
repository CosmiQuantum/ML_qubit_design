#!/usr/bin/env python3
from pathlib import Path
import cairosvg
import fitz  # pymupdf

GREEN = "#3D8B3D"
GREEN_LIGHT = "#E8F5E8"
FIG_ORANGE = "#E87A00"
FIG_PURPLE = "#6A0DAD"
PIPELINE_SHIFT = 46

OUT_DIR = Path("paper_plots")
FRAGMENTS_PDF = OUT_DIR / "fragments.pdf"
OVERLAY_SVG = OUT_DIR / "pipeline_overlay.svg"
OVERLAY_PDF = OUT_DIR / "pipeline_overlay.pdf"
FINAL_PDF = OUT_DIR / "pipeline_updated.pdf"

PAGE_W = 1120
PAGE_H = 970

FIG_X = 85
FIG_Y = 380
FIG_W = 950
FIG_H = 450

SVG_TEMPLATE = """\
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 1120 970"
     width="1120" height="970"
     font-family="Arial, Helvetica, sans-serif">

  <!-- transparent background -->

  <text x="560" y="36"
        text-anchor="middle" font-size="22" font-weight="bold" fill="#222">
    Testing Pipeline (Forward Pass)
  </text>

  <g transform="translate(%(PIPELINE_SHIFT)s,0)">
    <text x="515" y="74"
          text-anchor="middle" font-size="16" font-weight="bold"
          fill="%(GREEN)s">Validation loss</text>

    <circle cx="515" cy="105" r="12" fill="white"
            stroke="%(GREEN)s" stroke-width="2.5"/>
    <line x1="489" y1="105" x2="541" y2="105"
          stroke="%(GREEN)s" stroke-width="2.5"/>
    <line x1="515" y1="79" x2="515" y2="131"
          stroke="%(GREEN)s" stroke-width="2.5"/>

    <text x="260" y="126"
          text-anchor="middle" font-size="16" font-weight="bold"
          fill="%(GREEN)s">Reference
      <tspan fill="#555" font-weight="normal"> — SQuADDS dataset results</tspan>
    </text>

    <text x="615" y="126"
          text-anchor="start" font-size="16" font-weight="bold"
          fill="%(GREEN)s">Predicted
      <tspan fill="#555" font-weight="normal"> — forward pass results</tspan>
    </text>

    <path d="M 88,175 L 88,105 L 489,105"
          fill="none" stroke="%(GREEN)s" stroke-width="2.5"/>
    <polygon points="489,105 480,100 480,110" fill="%(GREEN)s"/>

    <path d="M 603,175 L 603,105 L 541,105"
          fill="none" stroke="%(GREEN)s" stroke-width="2.5"/>
    <polygon points="541,105 550,100 550,110" fill="%(GREEN)s"/>

    <rect x="18" y="175" width="140" height="54" rx="4" ry="4"
          fill="%(GREEN_LIGHT)s" stroke="%(GREEN)s" stroke-width="2.5"/>
    <text x="88" y="198" text-anchor="middle"
          font-size="16" font-weight="bold" fill="#333">ML Model</text>
    <text x="88" y="218" text-anchor="middle"
          font-size="16" font-weight="bold" fill="#333">predictions</text>

    <text x="88" y="248" text-anchor="middle"
          font-size="16" fill="#555" font-style="italic">Layout with target design</text>
    <text x="88" y="268" text-anchor="middle"
          font-size="16" fill="#555" font-style="italic">params for transmon-</text>
    <text x="88" y="288" text-anchor="middle"
          font-size="16" fill="#555" font-style="italic">resonator system:</text>
    <text x="88" y="310" text-anchor="middle"
          font-size="16" font-weight="bold" font-style="italic" fill="#333">
      &#x03B1;, f_qubit, f_res, &#x03BA;
    </text>

    <rect x="185" y="153" width="660" height="104" rx="5" ry="5"
          fill="none" stroke="#888" stroke-width="2" stroke-dasharray="8,4"/>
    <text x="202" y="171"
          font-size="16" fill="#888" font-style="italic" font-weight="bold">Python</text>

    <line x1="158" y1="202" x2="200" y2="202"
          stroke="%(GREEN)s" stroke-width="2.5"/>
    <polygon points="200,202 191,197 191,207" fill="%(GREEN)s"/>

    <rect x="203" y="180" width="130" height="46" rx="4" ry="4"
          fill="#E8E8E8" stroke="#999" stroke-width="1.5"/>
    <text x="268" y="208" text-anchor="middle"
          font-size="16" font-weight="bold" fill="#333">SQuADDS</text>

    <line x1="333" y1="203" x2="360" y2="203"
          stroke="#777" stroke-width="2"/>
    <polygon points="360,203 352,198 352,208" fill="#777"/>

    <rect x="363" y="180" width="150" height="46" rx="4" ry="4"
          fill="#E8E8E8" stroke="#999" stroke-width="1.5"/>
    <text x="438" y="208" text-anchor="middle"
          font-size="16" font-weight="bold" fill="#333">Quantum Metal</text>

    <line x1="513" y1="203" x2="540" y2="203"
          stroke="#777" stroke-width="2"/>
    <polygon points="540,203 532,198 532,208" fill="#777"/>

    <rect x="543" y="180" width="120" height="46" rx="4" ry="4"
          fill="%(GREEN_LIGHT)s" stroke="%(GREEN)s" stroke-width="2.5"/>
    <text x="603" y="208" text-anchor="middle"
          font-size="16" font-weight="bold" fill="#333">pyEPR</text>

    <path d="M 268,180 C 268,151 603,151 603,180"
          fill="none" stroke="#777" stroke-width="1.5" stroke-dasharray="6,3"/>
    <polygon points="603,180 598,172 608,172" fill="#777"/>

    <line x1="663" y1="195" x2="710" y2="195"
          stroke="#777" stroke-width="1.5"/>
    <polygon points="710,195 703,191 703,199" fill="#777"/>
    <line x1="710" y1="209" x2="663" y2="209"
          stroke="%(GREEN)s" stroke-width="1.5"/>
    <polygon points="663,209 670,205 670,213" fill="%(GREEN)s"/>

    <rect x="713" y="180" width="120" height="46" rx="4" ry="4"
          fill="#E8E8E8" stroke="#999" stroke-width="1.5"/>
    <text x="773" y="208" text-anchor="middle"
          font-size="16" font-weight="bold" fill="#333">PyAEDT</text>

    <line x1="833" y1="195" x2="888" y2="195"
          stroke="#777" stroke-width="1.5"/>
    <polygon points="888,195 881,191 881,199" fill="#777"/>
    <line x1="888" y1="209" x2="833" y2="209"
          stroke="#777" stroke-width="1.5"/>
    <polygon points="833,209 840,205 840,213" fill="#777"/>

    <rect x="891" y="180" width="120" height="46" rx="4" ry="4"
          fill="#E8E8E8" stroke="#999" stroke-width="1.5"/>
    <text x="951" y="198" text-anchor="middle"
          font-size="16" font-weight="bold" fill="#333">Ansys</text>
    <text x="951" y="216" text-anchor="middle"
          font-size="16" font-weight="bold" fill="#333">HFSS &amp; Q3D</text>
  </g>

  <line x1="40" y1="330" x2="1080" y2="330"
        stroke="#CCC" stroke-width="1.5" stroke-dasharray="6,4"/>

  <text x="560" y="362"
        text-anchor="middle" font-size="22" font-weight="bold" fill="#333">
    Transmon&#x2013;Resonator System: Parameter Identification
  </text>

  <line x1="269" y1="637" x2="157" y2="882"
        stroke="%(FIG_ORANGE)s" stroke-width="1.5" stroke-dasharray="5,3"/>
  <line x1="399" y1="637" x2="433" y2="882"
        stroke="%(FIG_ORANGE)s" stroke-width="1.5" stroke-dasharray="5,3"/>
  <line x1="750" y1="668" x2="709" y2="882"
        stroke="%(FIG_PURPLE)s" stroke-width="1.5" stroke-dasharray="5,3"/>
  <line x1="850" y1="628" x2="979" y2="882"
        stroke="%(FIG_PURPLE)s" stroke-width="1.5" stroke-dasharray="5,3"/>

  <rect x="22" y="882" width="270" height="62" rx="5" ry="5"
        fill="#FFF4E6" stroke="%(FIG_ORANGE)s" stroke-width="1.5"/>
  <text x="157" y="906" text-anchor="middle"
        font-size="16" font-weight="bold" fill="%(FIG_ORANGE)s">
    &#x03B1; (anharmonicity)
  </text>
  <text x="157" y="930" text-anchor="middle"
        font-size="16" fill="#555">&#x2248; &#x2212;E_C from Q3D</text>

  <rect x="298" y="882" width="270" height="62" rx="5" ry="5"
        fill="#FFF4E6" stroke="%(FIG_ORANGE)s" stroke-width="1.5"/>
  <text x="433" y="906" text-anchor="middle"
        font-size="16" font-weight="bold" fill="%(FIG_ORANGE)s">
    f_qubit (qubit frequency)
  </text>
  <text x="433" y="930" text-anchor="middle"
        font-size="16" fill="#555">&#x2248; &#x221A;(8 E_J E_C) &#x2212; E_C from Q3D</text>

  <rect x="574" y="882" width="270" height="62" rx="5" ry="5"
        fill="#F0E6F6" stroke="%(FIG_PURPLE)s" stroke-width="1.5"/>
  <text x="709" y="906" text-anchor="middle"
        font-size="16" font-weight="bold" fill="%(FIG_PURPLE)s">
    f_res (resonator freq.)
  </text>
  <text x="709" y="930" text-anchor="middle"
        font-size="16" fill="#555">HFSS eigenmode solver</text>

  <rect x="850" y="882" width="258" height="62" rx="5" ry="5"
        fill="#F0E6F6" stroke="%(FIG_PURPLE)s" stroke-width="1.5"/>
  <text x="979" y="906" text-anchor="middle"
        font-size="16" font-weight="bold" fill="%(FIG_PURPLE)s">
    &#x03BA; (resonator linewidth)
  </text>
  <text x="979" y="930" text-anchor="middle"
        font-size="16" fill="#555">HFSS via 50&#x2126; port loss</text>
</svg>
"""

svg = SVG_TEMPLATE % {
    "PIPELINE_SHIFT": PIPELINE_SHIFT,
    "GREEN": GREEN,
    "GREEN_LIGHT": GREEN_LIGHT,
    "FIG_ORANGE": FIG_ORANGE,
    "FIG_PURPLE": FIG_PURPLE,
}

OVERLAY_SVG.write_text(svg, encoding="utf-8")
print("Written pipeline_overlay.svg")

cairosvg.svg2pdf(
    bytestring=svg.encode("utf-8"),
    write_to=str(OVERLAY_PDF),
    output_width=PAGE_W,
    output_height=PAGE_H,
)
print("Written pipeline_overlay.pdf")

fragments_doc = fitz.open(str(FRAGMENTS_PDF))
overlay_doc = fitz.open(str(OVERLAY_PDF))

final_doc = fitz.open()
page = final_doc.new_page(width=PAGE_W, height=PAGE_H)

# Layer 1: white background
page.draw_rect(fitz.Rect(0, 0, PAGE_W, PAGE_H), color=None, fill=(1, 1, 1))

# Layer 2: fragments.pdf image placed in the target rectangle
target_rect = fitz.Rect(FIG_X, FIG_Y, FIG_X + FIG_W, FIG_Y + FIG_H)
page.show_pdf_page(target_rect, fragments_doc, 0, keep_proportion=True, overlay=True)

# Layer 3: SVG overlay (titles, boxes, leaders, annotations) on top
full_rect = fitz.Rect(0, 0, PAGE_W, PAGE_H)
page.show_pdf_page(full_rect, overlay_doc, 0, keep_proportion=True, overlay=True)

final_doc.save(str(FINAL_PDF))
print("Written pipeline_updated.pdf")