# Cavity Claw RouteMeander Eigenmode

This experiment predicts resonator side quantities for the cavity claw route meander design workflow.

1. `ml_00_*` through `ml_03_*` cover the baseline data analysis, training, evaluation, and tuner analysis flow.
2. `ml_10_*` through `ml_22_*` cover surrogate model and defined loss variants.
3. `validation_00_AnsysHFSS_simulations.ipynb` contains downstream HFSS validation work.
4. `plots`, `sweep_outputs`, and `sweeps` store generated analysis artifacts next to the notebooks that created them.
5. `metadata` holds saved names and config artifacts.
6. `results` holds training, prediction, runtime, and validation exports.
