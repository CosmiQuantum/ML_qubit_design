# Experiments

The repository's notebook workflows live here. Each experiment folder is largely self-contained and keeps its plots, exported CSVs, sweep results, and notebook variants together.

## Folders

- `model_predict_cavity_claw_RouteMeander_eigenmode`: resonator and cavity-claw route-meander eigenmode prediction workflow.
- `model_predict_coupler_NCap_cap_matrix`: coupler `NCap` capacitance-matrix prediction workflow.
- `model_predict_qubit_TransmonCross_cap_matrix`: transmon-cross capacitance-matrix prediction workflow.
- `model_predict_qubit-TransmonCross-Hamiltonian_params`: transmon-cross Hamiltonian-parameter workflow and stress-test analysis.

## Common Conventions

- `ml_00_*`: data loading and exploratory analysis.
- `ml_01_*`: primary training notebooks.
- `ml_02_*`: result inspection and CSV export.
- `ml_03_*`: hyperparameter analysis.
- `ml_10_*` and `ml_20_*`: surrogate-model or alternative-loss variants.
- `validation_*`: downstream simulation and Ansys validation notebooks.
- `parameters*.py`: notebook configuration for each training variant.

Large local-only artifacts such as cached datasets, trained models, and scalers are ignored through the root `.gitignore`.

Each experiment root is now structured so the notebooks stay at top level while ancillary files are grouped into:

- `metadata/`: schema files, saved column lists, and design/config JSON.
- `results/training/`: loss histories and test summaries.
- `results/predictions/`: exported prediction and reconstruction CSVs.
- `results/validation/`: sweep summaries, candidate exports, and Ansys-comparison tables.
