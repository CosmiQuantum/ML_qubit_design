# ML-Based Qubit Design

This repository contains notebook-driven machine learning experiments for predicting Qiskit Metal design parameters from target qubit, resonator, coupler, and Hamiltonian quantities.

## Quick Start

Create the conda environment:

```bash
./scripts/environment/create_conda_env.sh
conda activate cryo-modelling-env
```

Then launch JupyterLab from the repository root:

```bash
jupyter-lab
```

Environment notes, remote Jupyter instructions, and dependency caveats are documented in [docs/environment_setup.md](docs/environment_setup.md).

## Large Files

Some saved models and intermediate datasets are too large for Git. If you want to rerun notebooks without retraining everything from scratch, download the supplemental files from the shared Google Drive folder:

<https://drive.google.com/drive/folders/1WVHR4b4g1M4xdOUghbwNKrABafRz-YaQ?usp=sharing>

Those files should be unpacked into the corresponding experiment directories under `experiments/`.

## Repository Layout

- `experiments/`: the four main notebook-based modeling workflows.
- `figures/`: paper figure sources, generation scripts, and checked-in outputs.
- `scripts/`: environment helpers and small notebook-maintenance utilities.
- `docs/`: setup documentation and static reference images.
- `environment.yml`: conda environment definition used by the helper scripts.

More detailed folder guides live in:

- [experiments/README.md](experiments/README.md)
- [figures/README.md](figures/README.md)
- [scripts/README.md](scripts/README.md)
- [docs/README.md](docs/README.md)

## Experiments

The main experiment folders are:

- `experiments/model_predict_cavity_claw_RouteMeander_eigenmode`
- `experiments/model_predict_coupler_NCap_cap_matrix`
- `experiments/model_predict_qubit_TransmonCross_cap_matrix`
- `experiments/model_predict_qubit-TransmonCross-Hamiltonian_params`

Most workflows follow the same rough notebook sequence:

- `ml_00_*`: data inspection and preprocessing.
- `ml_01_*`: baseline training.
- `ml_02_*`: result inspection and export.
- `ml_03_*`: hyperparameter-search analysis.
- `ml_10_*` and `ml_20_*`: surrogate or defined-loss variants.
- `validation_*`: Ansys or downstream validation studies.

Each experiment directory also contains local `parameters*.py` configuration files plus generated CSV and plot outputs that stay next to the notebooks that produced them.

Within each experiment, non-notebook artifacts are now grouped into:

- `metadata/`: column-name files, saved schema/config JSON, and other lightweight descriptors.
- `results/training/`: history curves and test-loss summaries.
- `results/predictions/`: prediction tables and reconstruction outputs.
- `results/validation/`: sweep exports, candidate lists, and Ansys-comparison tables.
- `results/runtime/` and `results/cache/`: benchmarking and cached arrays where applicable.

## Paper Figures

Paper figure sources now live under `figures/paper/`:

- `source_materials/`: manually curated source material required by some figure generators.
- `outputs/`: checked-in generated SVG/PDF figures.
- `generate_*.py`: figure-generation scripts.

The desired end-to-end flow of the project is summarized below.

![Desired Flow](docs/images/desired_flow.png)

## Contact

Questions or comments can be sent to Olivia Seidel at `olivias@fnal.gov`.
