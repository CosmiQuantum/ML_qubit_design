# Qubit TransmonCross Hamiltonian Parameters

This experiment works in Hamiltonian parameter space for the transmon cross design and contains the most extensive validation and stress test analysis notebooks in the repo.

1. `ml_10_*` through `ml_22_*` contain surrogate model and defined loss variants.
2. `ml_30_*` and `ml_31_*` cover runtime benchmarking against the EM simulator.
3. `ml_32_*` through `ml_40_*` contain the training data amount sweeps (corner far-to-near, uniform split, fixed HP multiseed, and diagnostic variations).
4. `validation_12_*`, `validation_20_*`, `validation_21_*`, and `validation_22_*` contain stress test analysis, EM simulator validation, and design visualization work.
5. `plots`, `data`, `model`, `scalers`, and tuning directories contain experiment specific artifacts and cached outputs.
6. `metadata` holds saved names and config artifacts.
7. `results` groups training, prediction, validation, runtime, and cache outputs. Sweep CSVs live in `results/data_amount_sweep_corner` and `results/data_amount_sweep_uniform`.
