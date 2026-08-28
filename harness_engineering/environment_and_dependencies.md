# Environment and dependencies

## Why this file exists

This file keeps environment and version rules out of the root agent guide.

Its sources are `environment.yml`, `environment-eaf-gpu.yml`, and `docs/environment_setup.md`.

Read this file when changing dependencies, environment files, setup commands, Python versions, TensorFlow versions, GPU support, or environment documentation.

Review these rules when the environment files or supported computing platforms change. Remove details that are no longer present in those sources.

These rules remain active while the named environment files are the project source of truth. Replace them if the repository adopts a different environment system.

## Normal environment

Use `environment.yml` for normal work and activate `cryo-modelling-env`.

Keep these current requirements intact.

* Use the Python 3.10 series.
* Keep TensorFlow at 2.20.0.
* Keep TensorFlow Datasets at 4.8.3.
* Keep widgetsnbextension at 3.6.0.

Python 3.10.13 and TensorFlow 2.20.0 were recorded in a successful run. Treat Python 3.10.13 as known working evidence rather than an exact repository pin.

## GPU environment

Use `environment-eaf-gpu.yml` only for the Fermilab GPU workflow and activate `cryo-modelling-env-gpu`.

Keep TensorFlow with CUDA support at 2.20.0 in this environment.

The recorded GPU run used an NVIDIA A100 80GB PCIe host device with a 4g.40gb MIG allocation. This is known working evidence rather than a universal hardware requirement.

The TensorFlow CUDA extra does not independently pin the CUDA toolkit, cuDNN, drivers, or transitive packages.

## Packages without exact pins

The following normal environment packages currently have no exact release pin.

* NumPy
* SciPy
* Node.js
* IPython
* JupyterLab
* JupyterLab Git
* jsonschema with format support
* pandas
* joblib
* pydot
* Graphviz
* scikit learn
* Matplotlib
* seaborn
* ReportLab
* svglib
* CairoSVG
* PyMuPDF
* webcolors
* Keras Tuner

Do not invent versions for these packages.

The standalone Keras version was not recorded in older metadata. The external validation tools are also not pinned.

## Dependency changes

Treat the environment files as the source of truth.

Do not add, remove, upgrade, or downgrade dependencies unless the task requires an environment change.

Review the relevant environment file and `docs/environment_setup.md` together.

Update the setup guide when commands, environment names, required tools, or known working platform details change.

Explain why a dependency change is needed and verify that the environment still resolves.

Do not make GPU hardware a requirement for work that can be verified with the normal environment.
