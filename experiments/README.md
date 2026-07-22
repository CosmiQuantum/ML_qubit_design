# Experiments

The notebook workflows live here. Each experiment folder keeps its plots, exported CSV files, sweep results, and notebook variants together.

## Folders

1. `model_predict_cavity_claw_RouteMeander_eigenmode` handles resonator and cavity claw route meander eigenmode prediction.
2. `model_predict_coupler_NCap_cap_matrix` handles coupler `NCap` capacitance matrix prediction.
3. `model_predict_qubit_TransmonCross_cap_matrix` handles transmon cross capacitance matrix prediction.
4. `model_predict_qubit_TransmonCross_Hamiltonian_params` handles transmon cross Hamiltonian parameter prediction and stress test analysis.

## Common Conventions

1. `ml_00_*` contains data loading and exploratory analysis.
2. `ml_01_*` contains primary training notebooks.
3. `ml_02_*` contains result inspection and CSV export.
4. `ml_03_*` contains hyperparameter analysis.
5. `ml_10_*` and `ml_20_*` contain surrogate model or alternative loss variants.
6. `validation_*` contains downstream simulation and EM simulator validation notebooks.
7. `parameters*.py` contains notebook configuration for each training variant.

Large local artifacts such as cached datasets, trained models, and scalers are ignored through the root `.gitignore`.

Each experiment root keeps notebooks at top level while ancillary files are grouped into focused folders.

1. `metadata` contains schema files, saved column lists, and design or config JSON.
2. `results/training` contains loss histories and test summaries.
3. `results/predictions` contains exported prediction and reconstruction CSV files.
4. `results/validation` contains sweep summaries, candidate exports, and EM simulator comparison tables.
