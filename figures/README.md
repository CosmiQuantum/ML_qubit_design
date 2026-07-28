# Figures

This folder holds checked in figure assets and generation scripts that support the paper and project documentation.

1. `paper` contains source assets, figure generation scripts, and committed PDF and PNG outputs for paper figures.

## Paper figures

`paper/build_manuscript_exports.py` is the main driver. It reads the exported CSV and JSON results under each experiment's `results` folder and writes the manuscript figures to `paper/manuscript_exports`. Data figures should be regenerated through it rather than edited by hand, so the numbers in the paper always trace back to a results file.

Standalone diagram scripts sit alongside it, one per schematic figure, for example `generate_forward_testing_pipeline_figure.py` and `generate_inverse_design_workflow_figure.py`. These build SVG directly and rasterize with `cairosvg`, so they do not depend on experiment results.

Supporting folders

1. `manuscript_exports` holds the generated figure files referenced by the manuscript.
2. `source_materials` holds the vector assets the diagram scripts draw from. `fragments.pdf` supplies the layout panels for the overview and transmon resonator figures, and `pred_design_76.pdf` and `ref_design_76.pdf` supply the predicted and reference layouts used by `Figure1.ipynb` and `Figure13.ipynb`.
