# Paper Figures

This folder is organized into three parts.

1. `source_materials` contains manually curated source material used by the generators.
2. `manuscript_exports` contains the paper facing figure files used by the manuscript.
3. `generate_*.py` contains figure generation scripts.

## Scripts

1. `generate_inverse_design_workflow_figure.py` builds the matplotlib based end to end inverse design workflow figure.
2. `generate_inverse_design_workflow_svg.py` builds a pure SVG alternative for the workflow figure.
3. `generate_inverse_training_pipeline_figure.py` builds the inverse training pipeline figure.
4. `generate_forward_testing_pipeline_figure.py` builds the forward testing pipeline figure.
5. `generate_gaussian_stress_test_methodology_figure.py` builds the stress test methodology figure (uniform sampling + NN-distance binning).
6. `generate_transmon_resonator_system_figure.py` builds the composite transmon resonator system figure from `source_materials/fragments.pdf`.
7. `build_manuscript_exports.py` rebuilds the figure files used by the manuscript.
