#!/usr/bin/env python3
"""
Generate a three-panel overview figure for the manuscript.

The figure is composed directly as vector PDF with PyMuPDF so the panel labels,
wording, and font sizes stay under source control. The layout keeps panels
(a) and (b) on the top row and lets panel (c) span the full width below.

Outputs:
    figures/paper/manuscript_exports/overview_workflow_compact.pdf
    figures/paper/manuscript_exports/overview_workflow_compact.png
    figures/paper/manuscript_exports/overview_workflow_compact_{a,b,c}.pdf
    figures/paper/manuscript_exports/overview_workflow_compact_{a,b,c}.png
    experiments/model_predict_qubit_TransmonCross_Hamiltonian_params/plots/
        overview_workflow_compact.pdf
        overview_workflow_compact.png
        overview_workflow_compact_{a,b,c}.pdf
        overview_workflow_compact_{a,b,c}.png
"""

from __future__ import annotations

from math import atan2, cos, pi, sin
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
TRANSMON_PARAMS_PDF = MANUSCRIPT_EXPORTS_DIR / "figure1_transmon_qiskit_parameters.pdf"
OUT_STEM = "overview_workflow_compact"

PAGE_W = 7.10 * 72
PAGE_H = 7.45 * 72

TEXT = (0.13, 0.13, 0.13)
WHITE = (1, 1, 1)
PANEL_FILL = (0.985, 0.99, 0.995)
PANEL_EDGE = (0.78, 0.85, 0.88)
BLUE_TEXT = (0.22, 0.40, 0.50)
BLUE_EDGE = (0.31, 0.49, 0.58)
BLUE_FILL = (0.84, 0.91, 0.95)
BLUE_FILL_DARK = (0.69, 0.82, 0.89)
DARK_BLUE = (0.09, 0.25, 0.34)
LOOP_FILL = (0.95, 0.98, 0.99)
VALIDATION_FILL = (0.90, 0.94, 0.96)
ORANGE = (0.88, 0.42, 0.0)
PURPLE = (0.55, 0.16, 0.82)
GREEN = (0.17, 0.49, 0.20)
ARROW = (0.31, 0.31, 0.31)


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


def inset(rect: fitz.Rect, dx: float, dy: float | None = None) -> fitz.Rect:
    if dy is None:
        dy = dx
    return fitz.Rect(rect.x0 + dx, rect.y0 + dy, rect.x1 - dx, rect.y1 - dy)


def draw_panel(page: fitz.Page, rect: fitz.Rect, label: str | None, title: str) -> None:
    page.draw_rect(rect, color=PANEL_EDGE, fill=PANEL_FILL, width=0.8, overlay=True)
    title_x = rect.x0 + 7
    if label is not None:
        page.insert_text(
            fitz.Point(rect.x0 + 7, rect.y0 + 18),
            f"({label})",
            fontsize=13.0,
            fontname="helv",
            color=DARK_BLUE,
        )
        title_x = rect.x0 + 34
    page.insert_text(
        fitz.Point(title_x, rect.y0 + 18),
        title,
        fontsize=13.0,
        fontname="helv",
        color=BLUE_TEXT,
    )


def draw_centered_text(
    page: fitz.Page,
    rect: fitz.Rect,
    lines: tuple[str, ...] | list[str],
    *,
    fontsize: float,
    color=TEXT,
    lineheight: float | None = None,
) -> None:
    if lineheight is None:
        lineheight = fontsize * 1.18
    total_h = lineheight * len(lines)
    y = rect.y0 + (rect.height - total_h) / 2 + fontsize
    for line in lines:
        text_w = fitz.get_text_length(line, fontname="helv", fontsize=fontsize)
        page.insert_text(
            fitz.Point(rect.x0 + (rect.width - text_w) / 2, y),
            line,
            fontsize=fontsize,
            fontname="helv",
            color=color,
        )
        y += lineheight


def draw_textbox(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    *,
    fontsize: float,
    color=TEXT,
    align: int = fitz.TEXT_ALIGN_LEFT,
    lineheight: float | None = None,
) -> None:
    if lineheight is None:
        lineheight = fontsize * 1.18
    y = rect.y0 + fontsize
    for line in text.splitlines():
        text_w = fitz.get_text_length(line, fontname="helv", fontsize=fontsize)
        if align == fitz.TEXT_ALIGN_CENTER:
            x = rect.x0 + (rect.width - text_w) / 2
        elif align == fitz.TEXT_ALIGN_RIGHT:
            x = rect.x1 - text_w
        else:
            x = rect.x0
        page.insert_text(
            fitz.Point(x, y),
            line,
            fontsize=fontsize,
            fontname="helv",
            color=color,
        )
        y += lineheight


def draw_box(
    page: fitz.Page,
    rect: fitz.Rect,
    lines: tuple[str, ...] | list[str],
    *,
    fontsize: float = 9.4,
    fill=BLUE_FILL,
    edge=BLUE_EDGE,
    text_color=TEXT,
    width: float = 1.0,
) -> None:
    page.draw_rect(rect, color=edge, fill=fill, width=width, overlay=True)
    draw_centered_text(page, inset(rect, 3, 2), lines, fontsize=fontsize, color=text_color)


def draw_header_card(page: fitz.Page, rect: fitz.Rect, title: str, body: str) -> None:
    page.draw_rect(
        rect,
        color=BLUE_EDGE,
        fill=(0.91, 0.96, 0.985),
        width=0.9,
        overlay=True,
        radius=0.08,
    )
    page.insert_text(
        fitz.Point(rect.x0 + 9, rect.y0 + 15.2),
        title,
        fontsize=11.7,
        fontname="helv",
        color=DARK_BLUE,
    )
    page.draw_line(
        fitz.Point(rect.x0 + 9, rect.y0 + 22),
        fitz.Point(rect.x1 - 9, rect.y0 + 22),
        color=BLUE_EDGE,
        width=0.6,
        overlay=True,
    )
    draw_textbox(
        page,
        fitz.Rect(rect.x0 + 9, rect.y0 + 30, rect.x1 - 9, rect.y1 - 7),
        body,
        fontsize=8.7,
        color=TEXT,
        lineheight=10.0,
    )


def draw_arrow(
    page: fitz.Page,
    start: fitz.Point,
    end: fitz.Point,
    *,
    color=ARROW,
    width: float = 1.4,
    head: float = 5.5,
    dashes: str | None = None,
) -> None:
    page.draw_line(start, end, color=color, dashes=dashes, width=width, overlay=True)
    angle = atan2(end.y - start.y, end.x - start.x)
    wing = pi / 7
    p1 = fitz.Point(end.x - head * cos(angle - wing), end.y - head * sin(angle - wing))
    p2 = fitz.Point(end.x - head * cos(angle + wing), end.y - head * sin(angle + wing))
    page.draw_polyline([end, p1, p2], color=color, fill=color, closePath=True, overlay=True)


def draw_elbow_arrow(
    page: fitz.Page,
    points: list[fitz.Point],
    *,
    color=ARROW,
    width: float = 1.4,
    head: float = 5.5,
    dashes: str | None = None,
) -> None:
    for start, end in zip(points[:-2], points[1:-1]):
        page.draw_line(start, end, color=color, dashes=dashes, width=width, overlay=True)
    draw_arrow(page, points[-2], points[-1], color=color, width=width, head=head, dashes=dashes)


def mapped_rect(src: fitz.Rect, src_page: fitz.Rect, dest: fitz.Rect) -> fitz.Rect:
    scale = dest.width / src_page.width
    return fitz.Rect(
        dest.x0 + src.x0 * scale,
        dest.y0 + src.y0 * scale,
        dest.x0 + src.x1 * scale,
        dest.y0 + src.y1 * scale,
    )


def fit_rect_top(src_rect: fitz.Rect, target_rect: fitz.Rect) -> fitz.Rect:
    fitted = fit_rect(src_rect, target_rect)
    return fitz.Rect(fitted.x0, target_rect.y0, fitted.x1, target_rect.y0 + fitted.height)


def draw_panel_a(page: fitz.Page, rect: fitz.Rect, label: str | None = "a") -> None:
    draw_panel(page, rect, label, "Layout and model scope")

    fragments_doc = fitz.open(FRAGMENTS_PDF)
    fragments_page = fragments_doc[0]
    fragments_src = fragments_page.rect
    art_box = fitz.Rect(rect.x0 + 8, rect.y0 + 45, rect.x1 - 8, rect.y0 + 157)
    art_dest = fit_rect(fragments_src, art_box)
    fragments_pix = fragments_page.get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False)
    page.insert_image(art_dest, pixmap=fragments_pix)
    fragments_doc.close()

    # Replace acronym-heavy labels embedded in the source artwork.
    label_covers = [
        fitz.Rect(5, 116, 66, 150),
        fitz.Rect(168, 3, 236, 39),
    ]
    for src_label in label_covers:
        page.draw_rect(
            mapped_rect(src_label, fragments_src, art_dest),
            color=WHITE,
            fill=WHITE,
            width=0,
            overlay=True,
        )
    draw_textbox(
        page,
        fitz.Rect(rect.x0 + 14, rect.y0 + 31, rect.x0 + 86, rect.y0 + 44),
        "Qubit subsystem",
        fontsize=7.9,
        color=ORANGE,
        align=fitz.TEXT_ALIGN_CENTER,
        lineheight=8.5,
    )
    draw_textbox(
        page,
        fitz.Rect(rect.x0 + 88, rect.y0 + 24, rect.x1 - 10, rect.y0 + 45),
        "Coplanar waveguide\ncavity subsystem",
        fontsize=7.4,
        color=PURPLE,
        align=fitz.TEXT_ALIGN_CENTER,
        lineheight=8.0,
    )

    # Blown-up transmon-cross view with the three varied Quantum Metal
    # parameters annotated (absorbed from the former standalone Figure 1).
    params_doc = fitz.open(TRANSMON_PARAMS_PDF)
    params_page = params_doc[0]
    # Clip away the source figure's own title band. The annotated green
    # substrate square in the source PDF spans x 49..351.5, y 31.25..333.75 pt.
    params_clip = fitz.Rect(
        params_page.rect.x0,
        params_page.rect.y0 + 30,
        params_page.rect.x1,
        params_page.rect.y1,
    )
    green_square_src = fitz.Rect(49.0, 31.25, 351.5, 333.75)
    blowup_box = fitz.Rect(rect.x0 + 8, rect.y0 + 163, rect.x1 - 8, rect.y1 - 5)
    blowup_dest = fit_rect(params_clip, blowup_box)
    page.show_pdf_page(blowup_dest, params_doc, 0, clip=params_clip)
    params_doc.close()

    # Green square position inside the placed blow-up.
    blow_scale = blowup_dest.width / params_clip.width
    green_dest = fitz.Rect(
        blowup_dest.x0 + (green_square_src.x0 - params_clip.x0) * blow_scale,
        blowup_dest.y0 + (green_square_src.y0 - params_clip.y0) * blow_scale,
        blowup_dest.x0 + (green_square_src.x1 - params_clip.x0) * blow_scale,
        blowup_dest.y0 + (green_square_src.y1 - params_clip.y0) * blow_scale,
    )

    # Dashed zoom box centered on the transmon cross + claw in the layout art
    # (measured in fragments.pdf source coordinates), with matching guides
    # that land exactly on the top corners of the blown-up green square.
    zoom_src_fragments = fitz.Rect(101.0, 14.5, 143.5, 85.5)
    zoom_box = mapped_rect(zoom_src_fragments, fragments_src, art_dest)
    dash = "[2 2] 0"
    page.draw_rect(zoom_box, color=BLUE_EDGE, fill=None, width=0.9, dashes=dash, overlay=True)
    page.draw_rect(green_dest, color=BLUE_EDGE, fill=None, width=0.9, dashes=dash, overlay=True)
    page.draw_line(
        fitz.Point(zoom_box.x0, zoom_box.y1),
        fitz.Point(green_dest.x0, green_dest.y0),
        color=BLUE_EDGE,
        width=0.9,
        dashes=dash,
        overlay=True,
    )
    page.draw_line(
        fitz.Point(zoom_box.x1, zoom_box.y1),
        fitz.Point(green_dest.x1, green_dest.y0),
        color=BLUE_EDGE,
        width=0.9,
        dashes=dash,
        overlay=True,
    )


def draw_panel_b(page: fitz.Page, rect: fitz.Rect, label: str | None = "b") -> None:
    draw_panel(page, rect, label, "Training-time inverse pipeline")

    x0, y0 = rect.x0, rect.y0
    cx = (rect.x0 + rect.x1) / 2
    dash = "[3 3] 0"

    def rounded_box(
        box: fitz.Rect,
        lines: tuple[str, ...],
        *,
        fontsize: float,
        fill=BLUE_FILL_DARK,
        color=TEXT,
        lineheight: float | None = None,
        edge=BLUE_EDGE,
        width: float = 1.15,
    ) -> None:
        page.draw_rect(
            box,
            color=edge,
            fill=fill,
            width=width,
            overlay=True,
            radius=0.13,
        )
        draw_centered_text(
            page,
            inset(box, 4, 2),
            list(lines),
            fontsize=fontsize,
            color=color,
            lineheight=lineheight,
        )

    bw = 109
    frame_hw = 107

    def _hw(text: str, fs: float) -> float:
        return fitz.get_text_length(text, fontname="helv", fontsize=fs) / 2 + 13

    inv_hw  = _hw("Inverse neural network", 8.5)
    geo_hw  = _hw("Quantum Metal parameter prediction", 8.0)
    sur_hw  = _hw("Forward surrogate neural network", 8.0)
    rec_hw  = _hw("Hamiltonian reconstruction", 8.0)
    loss_hw = max(_hw("Compute loss", 7.4),
                  _hw("Average absolute difference between", 5.6),
                  _hw("target and reconstructed Hamiltonian", 5.6))

    cap = frame_hw - 4
    inv_hw  = min(inv_hw,  cap)
    geo_hw  = min(geo_hw,  cap)
    sur_hw  = min(sur_hw,  cap)
    rec_hw  = min(rec_hw,  cap)
    loss_hw = min(loss_hw, cap)

    top_box = fitz.Rect(cx - bw, y0 + 24, cx + bw, y0 + 46)
    frame = fitz.Rect(cx - frame_hw, y0 + 52, cx + frame_hw, y0 + 217)
    inverse      = fitz.Rect(cx - inv_hw,  y0 + 60, cx + inv_hw,  y0 + 80)
    geometry     = fitz.Rect(cx - geo_hw,  y0 + 90, cx + geo_hw,  y0 + 110)
    surrogate    = fitz.Rect(cx - sur_hw,  y0 + 120, cx + sur_hw, y0 + 140)
    reconstruction = fitz.Rect(cx - rec_hw, y0 + 150, cx + rec_hw, y0 + 170)
    loss         = fitz.Rect(cx - loss_hw, y0 + 178, cx + loss_hw, y0 + 206)
    output       = fitz.Rect(cx - bw, y0 + 219, cx + bw, y0 + 234)

    rounded_box(
        top_box,
        ("Requested Hamiltonian values", "qubit frequency, anharmonicity"),
        fontsize=7.6,
        lineheight=8.6,
        fill=BLUE_FILL,
    )
    page.draw_rect(
        frame,
        color=BLUE_EDGE,
        fill=None,
        dashes=dash,
        width=1.2,
        overlay=True,
        radius=0.075,
    )
    _training_w = fitz.get_text_length("Training", fontname="helv", fontsize=7.4)
    _tx = frame.x1 - _training_w - 9
    _ty = frame.y0 + 5   # baseline sits on the top border line
    page.draw_rect(
        fitz.Rect(_tx - 8, _ty - 9, _tx + _training_w + 8, _ty + 2),
        color=None, fill=WHITE, width=0, overlay=True,
    )
    page.insert_text(
        fitz.Point(_tx, _ty),
        "Training",
        fontsize=7.4,
        fontname="helv",
        color=BLUE_TEXT,
    )

    rounded_box(
        inverse,
        ("Inverse neural network",),
        fontsize=8.5,
    )
    rounded_box(
        geometry,
        ("Quantum Metal parameter prediction",),
        fontsize=8.0,
    )
    rounded_box(
        surrogate,
        ("Forward surrogate neural network",),
        fontsize=8.0,
    )
    rounded_box(
        reconstruction,
        ("Hamiltonian reconstruction",),
        fontsize=8.0,
    )

    page.draw_rect(loss, color=DARK_BLUE, fill=DARK_BLUE, width=1.0, overlay=True, radius=0.13)
    draw_centered_text(
        page,
        fitz.Rect(loss.x0 + 5, loss.y0 + 4, loss.x1 - 5, loss.y0 + 14),
        ("Compute loss",),
        fontsize=7.4,
        color=WHITE,
    )
    draw_centered_text(
        page,
        fitz.Rect(loss.x0 + 5, loss.y0 + 14, loss.x1 - 5, loss.y1 - 3),
        ("Average absolute difference between", "target and reconstructed Hamiltonian"),
        fontsize=5.6,
        color=WHITE,
        lineheight=6.4,
    )
    rounded_box(
        output,
        ("Trained inverse-model output",),
        fontsize=7.2,
        fill=WHITE,
        width=1.0,
    )

    arrow_x = cx
    vertical_arrows = [
        (top_box.y1, inverse.y0),
        (inverse.y1, geometry.y0),
        (geometry.y1, surrogate.y0),
        (surrogate.y1, reconstruction.y0),
        (reconstruction.y1, loss.y0),
    ]
    for start_y, end_y in vertical_arrows:
        draw_arrow(
            page,
            fitz.Point(arrow_x, start_y + 1),
            fitz.Point(arrow_x, end_y - 1),
            color=ARROW,
            width=1.1,
            head=4.8,
        )
    draw_arrow(
        page,
        fitz.Point(arrow_x, loss.y1 + 2),
        fitz.Point(arrow_x, output.y0 - 2),
        color=ARROW,
        width=1.55,
        head=5.8,
    )

    update_x = cx - geo_hw - 6  # just left of the widest (geometry) box
    draw_elbow_arrow(
        page,
        [
            fitz.Point(loss.x0, loss.y0 + 14),
            fitz.Point(update_x, loss.y0 + 14),
            fitz.Point(update_x, inverse.y0 + 10),
            fitz.Point(inverse.x0 - 2, inverse.y0 + 10),
        ],
        color=BLUE_EDGE,
        width=1.1,
        head=4.8,
        dashes=dash,
    )
    _uw = fitz.get_text_length("update inverse weights", fontname="helv", fontsize=6.5)
    _ux, _uy = update_x - 12, y0 + 162
    page.draw_rect(
        fitz.Rect(_ux - 3, _uy - _uw - 4, _ux + 6.5 + 3, _uy + 4),
        color=None, fill=WHITE, width=0, overlay=True,
    )
    page.insert_text(
        fitz.Point(_ux, _uy),
        "update inverse weights",
        fontsize=6.5,
        fontname="helv",
        color=BLUE_TEXT,
        rotate=90,
    )


def draw_workflow_acronym_overlays(
    page: fitz.Page,
    workflow_dest: fitz.Rect,
    workflow_src: fitz.Rect,
) -> None:
    """Patch old workflow labels while preserving the original panel layout."""

    scale = workflow_dest.width / workflow_src.width

    def map_rect(rect: fitz.Rect) -> fitz.Rect:
        return mapped_rect(rect, workflow_src, workflow_dest)

    def draw_replacement(
        cover_src: fitz.Rect,
        text: str,
        *,
        fill,
        color,
        fontsize: float,
        align: int = fitz.TEXT_ALIGN_CENTER,
        lineheight: float | None = None,
    ) -> None:
        cover = map_rect(cover_src)
        page.draw_rect(cover, color=None, fill=fill, width=0, overlay=True)
        draw_textbox(
            page,
            cover,
            text,
            fontsize=fontsize * scale,
            color=color,
            align=align,
            lineheight=(lineheight * scale if lineheight else None),
        )

    draw_replacement(
        fitz.Rect(366, 23.0, 469, 43.0),
        "Inverse neural\nnetwork",
        fill=BLUE_FILL_DARK,
        color=BLUE_TEXT,
        fontsize=7.9,
        lineheight=7.4,
    )
    draw_replacement(
        fitz.Rect(248, 148.0, 351, 166.0),
        "Surrogate or\nEM solver",
        fill=DARK_BLUE,
        color=WHITE,
        fontsize=7.4,
        lineheight=7.1,
    )


def draw_panel_c(page: fitz.Page, rect: fitz.Rect, label: str | None = "c") -> None:
    draw_panel(page, rect, label, "Inference and validation workflow")

    workflow_doc = fitz.open(WORKFLOW_PDF)
    workflow_page = workflow_doc[0]
    workflow_src = workflow_page.rect
    workflow_box = fitz.Rect(rect.x0 + 8, rect.y0 + 22, rect.x1 - 8, rect.y1 - 7)
    workflow_dest = fit_rect_top(workflow_src, workflow_box)
    workflow_pix = workflow_page.get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False)
    page.insert_image(workflow_dest, pixmap=workflow_pix)
    workflow_doc.close()

    draw_workflow_acronym_overlays(page, workflow_dest, workflow_src)


def compose_pdf(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.draw_rect(page.rect, color=None, fill=WHITE)

    panel_a = fitz.Rect(10, 14, 218, 301)
    panel_b = fitz.Rect(230, 14, PAGE_W - 10, 258)
    panel_c = fitz.Rect(10, 309, PAGE_W - 10, PAGE_H - 12)

    draw_panel_a(page, panel_a)
    draw_panel_b(page, panel_b)
    draw_panel_c(page, panel_c)

    doc.save(out_path, deflate=True, garbage=4)
    doc.close()
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


def compose_panel_pdf(out_path: Path, panel: str) -> None:
    panel_dims = {
        "a": (208, 287),
        "b": (PAGE_W - 240, 244),
        "c": (PAGE_W - 20, PAGE_H - 328),
    }
    panel_drawers = {
        "a": draw_panel_a,
        "b": draw_panel_b,
        "c": draw_panel_c,
    }
    width, height = panel_dims[panel]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.draw_rect(page.rect, color=None, fill=WHITE)
    panel_drawers[panel](page, fitz.Rect(0, 0, width, height), label=None)
    doc.save(out_path, deflate=True, garbage=4)
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
        for panel in ("a", "b", "c"):
            panel_pdf = out_dir / f"{OUT_STEM}_{panel}.pdf"
            panel_png = out_dir / f"{OUT_STEM}_{panel}.png"
            compose_panel_pdf(panel_pdf, panel)
            render_png(panel_pdf, panel_png)


if __name__ == "__main__":
    main()
