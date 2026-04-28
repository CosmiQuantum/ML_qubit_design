# Cavity Claw Route-Meander Eigenmode

This experiment predicts resonator-side quantities for the cavity-claw route-meander design workflow.

- `ml_00_*` through `ml_03_*` cover the baseline data-analysis, training, evaluation, and tuner-analysis flow.
- `ml_10_*` through `ml_22_*` cover surrogate-model and defined-loss variants.
- `validation_00_AnsysHFSS_simulations.ipynb` contains downstream HFSS validation work.
- `plots/`, `sweep_outputs/`, and `sweeps/` store generated analysis artifacts next to the notebooks that created them.
- `metadata/` holds saved names/config artifacts; `results/` holds training, prediction, runtime, and validation exports.
