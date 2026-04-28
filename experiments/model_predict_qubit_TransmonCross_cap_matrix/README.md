# Qubit TransmonCross Capacitance Matrix

This experiment predicts transmon-cross capacitance-matrix quantities and related validation outputs.

- `ml_00_*` through `ml_03_*` define the baseline workflow.
- `ml_10_*` through `ml_22_*` contain surrogate-model and defined-loss variants.
- `ml_14_surrogate_stress_test.ipynb` contains the local surrogate stress-test workflow.
- `validation_00_AnsysQ3D_simulations.ipynb` covers Q3D validation.
- `plots/`, `sweep_outputs/`, and `sweeps/` store generated results alongside the notebooks that created them.
- `metadata/` holds saved names/config artifacts; `results/` groups training, prediction, and validation exports.
