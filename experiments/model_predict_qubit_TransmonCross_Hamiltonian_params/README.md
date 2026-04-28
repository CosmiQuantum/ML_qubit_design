# Qubit TransmonCross Hamiltonian Parameters

This experiment works in Hamiltonian-parameter space for the transmon-cross design and contains the most extensive validation and stress-test analysis notebooks in the repo.

- `ml_00_*` through `ml_03_*` cover the baseline Hamiltonian workflow.
- `ml_10_*` through `ml_22_*` contain surrogate-model and defined-loss variants.
- `validation_12_*`, `validation_20_*`, `validation_21_*`, and `validation_22_*` contain stress-test analysis, Ansys validation, and design-visualization work.
- `plots/`, `data/`, `model/`, `scalers/`, and tuning directories contain experiment-specific artifacts and cached outputs.
- `metadata/` holds saved names/config artifacts; `results/` groups training, prediction, validation, runtime, and cache outputs.
