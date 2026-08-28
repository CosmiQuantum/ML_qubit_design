# Repository structure

## Why this file exists

This file explains where work belongs and which existing layouts must be preserved.

Its sources are the root README and the README files under `experiments`, `figures`, and `docs`.

Read this file when adding, moving, renaming, or saving notebooks, data, models, figures, results, documentation, or generated artifacts.

Review these rules when the repository structure changes. Remove paths that no longer exist and add new durable locations only after they are established.

These rules remain active while the named folders and artifact layouts exist. Replace or remove them when the repository deliberately adopts a new structure.

## Experiments

The `experiments` folder contains scientific notebooks, parameter files, models, data products, plots, and validation results.

The four main experiment areas are listed below.

* `model_predict_cavity_claw_RouteMeander_eigenmode` contains the cavity claw resonator work.
* `model_predict_coupler_NCap_cap_matrix` contains the NCap coupler capacitance work.
* `model_predict_qubit_TransmonCross_cap_matrix` contains the transmon cross capacitance work.
* `model_predict_qubit_TransmonCross_Hamiltonian_params` contains the transmon Hamiltonian and inverse design work.

Keep experiment specific configuration in the existing `parameters` files inside that experiment.

Keep generated CSV files, plots, model outputs, scaler files, tuner results, and validation outputs with the experiment that produced them.

Follow the local folder pattern instead of creating a new global output area.

Some older outputs live beside their notebooks while newer work uses `metadata` and `results` folders. Preserve both layouts during unrelated work.

Preserve notebook numbering because it communicates workflow order.

The common sequence begins with data analysis, continues through training and results, and then moves into surrogate and validation studies.

The transmon Hamiltonian experiment begins later because it shares earlier preparation work. Its later notebooks cover stress testing, result reporting, runtime benchmarks, data amount sweeps, architecture sweeps, and multi seed diagnostics.

## Models and related artifacts

Two small Keras models are tracked in the transmon Hamiltonian model folder.

* `best_keras_model_model2_surrogate.keras`
* `surrogate_loss_2in_3out_best_model.keras`

Other models and supplemental files may live outside Git. Do not assume missing external data, models, or scalers are available.

## Figures

The `figures` folder contains figure source files, generation scripts, and checked in outputs.

Paper figure scripts live under `figures/paper`.

Keep new figure sources and outputs with the existing paper workflow.

## Documentation

The `docs` folder contains setup notes and reference material.

Keep general project orientation in the root `README.md`.

Keep normal dependencies in `environment.yml` and Fermilab GPU dependencies in `environment-eaf-gpu.yml`.

Keep the four tutorial harness entry points in the repository root. Keep focused rule documents under `harness_engineering`.

## Preservation rules

Make focused changes in the smallest relevant area.

Do not move or rename notebooks, experiment folders, result folders, models, or figure outputs unless the task requires it.

Do not reorganize the repository as part of an unrelated change.

Do not clear notebook outputs, rewrite notebook metadata, save incidental execution counts, reorder cells, or keep unrelated output changes.

Do not replace local parameter files with a new configuration framework unless the task requires that redesign.

Follow existing naming and numbering patterns when adding a notebook, parameter variant, result, or figure.

Honor `.gitignore`. Do not force ignored data, models, scalers, tuner outputs, caches, or temporary files into Git.
