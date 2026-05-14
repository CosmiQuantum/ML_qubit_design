#!/usr/bin/env python3
"""
Generate a combined overview figure from existing PDF figure assets.

This script deliberately composes PDF pages with PyMuPDF instead of rendering
them through matplotlib. That keeps the source artwork from fragments.pdf,
inverse_pipeline.pdf, and workflow.pdf at their native PDF quality.

Outputs:
    figures/paper/manuscript_exports/overview_workflow_compact.pdf
    figures/paper/manuscript_exports/overview_workflow_compact.png
    experiments/model_predict_qubit_TransmonCross_Hamiltonian_params/plots/
        overview_workflow_compact.pdf
        overview_workflow_compact.png
"""

from __future__ import annotations

from pathlib import Path

import fitz

from _paths import MANUSCRIPT_EXPORTS_DIR, SOURCE_MATERIALS_DIR


REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSMON_PLOTS_DIR = (
    REPO_ROOT
    / "experiments"
    / "model_predict_qubit_TransmonCross_Hamiltonian_params"
    / "plots"
)

FRAGMENTS_PDF = SOURCE_MATERIALS_DIR / "fragments.pdf"
INVERSE_PIPELINE_PDF = MANUSCRIPT_EXPORTS_DIR / "inverse_pipeline.pdf"
WORKFLOW_PDF = MANUSCRIPT_EXPORTS_DIR / "workflow.pdf"
OUT_STEM = "overview_workflow_compact"

PAGE_W = 7.10 * 72
PAGE_H = 8.45 * 72

TEXT = (0.13, 0.13, 0.13)
WHITE = (1, 1, 1)
BLUE_TEXT = (0.24, 0.43, 0.53)
BLUE_EDGE = (0.34, 0.48, 0.56)
BLUE_FILL = (0.84, 0.90, 0.93)


def fit_rect(src_rect: fitz.Rect, target_rect: fitz.Rect) -> fitz.Rect:
    """Return a centered rectangle preserving src_rect's aspect ratio."""

    src_aspect = src_rect.width / src_rect.height
    target_aspect = target_rect.width / target_rect.height
    if src_aspect >= target_aspect:
        width = target_rect.width
        height = width / src_aspect
    else:
        height = target_rect.height
        width = height * src_aspect
    x0 = target_rect.x0 + (target_rect.width - width) / 2
    y0 = target_rect.y0 + (target_rect.height - height) / 2
    return fitz.Rect(x0, y0, x0 + width, y0 + height)


def draw_section_title(page: fitz.Page, x: float, y: float, title: str) -> None:
    page.insert_text(
        fitz.Point(x, y),
        title,
        fontsize=11,
        fontname="helv",
        color=BLUE_TEXT,
    )


def draw_scope_card(page: fitz.Page, rect: fitz.Rect) -> None:
    header_h = 18
    page.draw_rect(
        rect,
        color=BLUE_EDGE,
        fill=BLUE_FILL,
        width=0.7,
        overlay=True,
    )
    page.draw_rect(
        fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + header_h),
        color=BLUE_EDGE,
        fill=BLUE_EDGE,
        width=0,
        overlay=True,
    )
    page.insert_text(
        fitz.Point(rect.x0 + 7, rect.y0 + 12.5),
        "Scope of transmon model",
        fontsize=10,
        fontname="helv",
        color=WHITE,
    )
    note_lines = (
        "Learned/varied: TransmonCross qubit",
        "geometry.",
        "Predicted: claw length, ground",
        "spacing, and cross length.",
        "Targets: qubit frequency and",
        "anharmonicity.",
        "Held fixed: all other SQuADDS",
        "layout settings.",
    )
    y = rect.y0 + header_h + 14
    for line in note_lines:
        page.insert_text(
            fitz.Point(rect.x0 + 7, y),
            line,
            fontsize=10,
            fontname="helv",
            color=TEXT,
        )
        y += 12


def compose_pdf(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.draw_rect(page.rect, color=None, fill=WHITE)

    # Panel a: directly embed the full original fragments.pdf artwork. This
    # preserves the complete qubit, coupled-layout, and right-side resonator
    # context at native PDF quality.
    fragments_doc = fitz.open(FRAGMENTS_PDF)
    fragments_clip = fragments_doc[0].rect
    panel_a_box = fitz.Rect(9, 38, 202, 183)
    panel_a_dest = fit_rect(fragments_clip, panel_a_box)
    draw_section_title(page, 10, 26, "Layout context")
    page.show_pdf_page(panel_a_dest, fragments_doc, 0, clip=fragments_clip, keep_proportion=True)
    fragments_doc.close()

    draw_scope_card(page, fitz.Rect(11, 198, 214, 332))

    # Panel b: keep the original inverse-training workflow from Fig. 1b.
    inverse_doc = fitz.open(INVERSE_PIPELINE_PDF)
    inverse_src = inverse_doc[0].rect
    panel_b_box = fitz.Rect(215, 38, PAGE_W - 9, 348)
    panel_b_dest = fit_rect(inverse_src, panel_b_box)
    draw_section_title(page, 215, 26, "Training-time inverse pipeline")
    page.show_pdf_page(panel_b_dest, inverse_doc, 0, keep_proportion=True)
    inverse_doc.close()

    # Panel c: keep the original end-to-end workflow from Fig. 2 at nearly full
    # text width, so the text size remains comparable to the original figure.
    workflow_doc = fitz.open(WORKFLOW_PDF)
    workflow_src = workflow_doc[0].rect
    panel_c_box = fitz.Rect(9, 400, PAGE_W - 9, PAGE_H - 16)
    panel_c_dest = fit_rect(workflow_src, panel_c_box)
    draw_section_title(page, 10, 386, "Inference-time transmon-cross workflow")
    page.show_pdf_page(panel_c_dest, workflow_doc, 0, keep_proportion=True)
    workflow_doc.close()

    doc.save(out_path)
    doc.close()
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


def render_png(pdf_path: Path, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
    pix.save(png_path)
    doc.close()
    print(f"wrote {png_path.relative_to(REPO_ROOT)}")


def main() -> None:
    output_dirs = (MANUSCRIPT_EXPORTS_DIR, TRANSMON_PLOTS_DIR)
    for out_dir in output_dirs:
        pdf_path = out_dir / f"{OUT_STEM}.pdf"
        png_path = out_dir / f"{OUT_STEM}.png"
        compose_pdf(pdf_path)
        render_png(pdf_path, png_path)


if __name__ == "__main__":
    main()
