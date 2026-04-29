# Qubit TransmonCross Capacitance Matrix

This experiment predicts transmon cross capacitance matrix quantities and related validation outputs.

1. `ml_00_*` through `ml_03_*` define the baseline workflow.
2. `ml_10_*` through `ml_22_*` contain surrogate model and defined loss variants.
3. `ml_14_surrogate_stress_test.ipynb` contains the local surrogate stress test workflow.
4. `validation_00_AnsysQ3D_simulations.ipynb` covers Q3D validation.
5. `plots`, `sweep_outputs`, and `sweeps` store generated results alongside the notebooks that created them.
6. `metadata` holds saved names and config artifacts.
7. `results` groups training, prediction, and validation exports.
