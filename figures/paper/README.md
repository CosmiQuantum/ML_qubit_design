# Paper Figures

This folder is organized into three parts:

- `source_materials/`: manually curated source material used by the generators.
- `outputs/`: checked-in generated SVG/PDF figures.
- `manuscript_exports/`: the paper-facing figure files that `sn-article.tex` points to.
- `generate_*.py`: figure-generation scripts.

## Scripts

- `generate_inverse_design_workflow_figure.py`: matplotlib-based end-to-end inverse-design workflow figure.
- `generate_inverse_design_workflow_svg.py`: pure-SVG alternative for the workflow figure.
- `generate_inverse_training_pipeline_figure.py`: inverse-training pipeline figure.
- `generate_forward_testing_pipeline_figure.py`: forward/testing pipeline figure.
- `generate_gaussian_stress_test_methodology_figure.py`: Gaussian stress-test methodology figure.
- `generate_transmon_resonator_system_figure.py`: composite transmon-resonator system figure built from `source_materials/fragments.pdf`.
- `build_manuscript_exports.py`: rebuilds the figure files used directly by the manuscript. When the repo still has the underlying data, this script regenerates the figure. When only the compiled paper preserved a figure, it exports a clean crop from the PDF as a fallback.
