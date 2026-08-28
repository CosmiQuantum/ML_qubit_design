# Welcome to the ML qubit design repository

This file contains the overall working rules for the whole repository.

The goal is to keep the science reproducible, the repository familiar, and the work easy to understand for someone who is new to machine learning.

## Start here

Before changing anything, complete these steps.

1. Read this file completely.
2. Read `README.md` for the project purpose and current workflow.
3. Read `docs/environment_setup.md` before changing environments or dependencies.
4. Read the nearest folder README before working inside `experiments`, `figures`, or `docs`.
5. Inspect the current Git status and preserve all user changes.
6. Identify which files are inputs, which files are generated outputs, and which files may be external before running code.
7. Write a small definition of done with observable results and relevant checks.

Ask for direction when a missing choice would change the scientific meaning of the work. Make a reasonable documented assumption when the choice is small and reversible.

## Project purpose

This repository contains notebook driven machine learning experiments for predicting Quantum Metal design parameters from target qubit, resonator, coupler, and Hamiltonian quantities.

The main workflow uses forward models to predict electromagnetic or Hamiltonian behavior from geometry. The transmon Hamiltonian work also uses an inverse model to predict geometry from requested Hamiltonian values. A frozen forward surrogate checks the inverse prediction in Hamiltonian space.

Explain these ideas in plain language whenever they matter to the task. Do not assume that a new reader already knows terms such as surrogate model, inverse design, scaler, tuner, epoch, loss, or one hot encoding.

## Environment and versions

Use `environment.yml` for normal CPU work and activate `cryo-modelling-env`.

Use `environment-eaf-gpu.yml` only for the Fermilab GPU workflow and activate `cryo-modelling-env-gpu`.

Treat both environment files as the source of truth. Do not add, remove, upgrade, or downgrade dependencies unless the task asks for an environment change.

Keep these current requirements intact.

* Use the Python 3.10 series.
* Keep TensorFlow at 2.20.0 in the normal environment.
* Keep TensorFlow Datasets at 4.8.3 in the normal environment.
* Keep widgetsnbextension at 3.6.0 in the normal environment.
* Keep TensorFlow with CUDA support at 2.20.0 in the GPU environment.

Python 3.10.13 and TensorFlow 2.20.0 were recorded in a successful GPU notebook run. Treat Python 3.10.13 as known working evidence rather than an exact repository pin.

The recorded GPU run used an NVIDIA A100 80GB PCIe host device with a 4g.40gb MIG allocation. This is known working hardware and not a universal requirement.

The following packages are currently not pinned to exact releases in the normal environment.

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

Do not invent versions for unpinned packages. The standalone Keras version was not recorded in older metadata. The TensorFlow CUDA extra does not independently pin the CUDA toolkit, cuDNN, drivers, or transitive packages. The external validation tools are also not pinned.

When a task changes a dependency, review the relevant environment file and `docs/environment_setup.md` together. Update the setup guide when commands, environment names, required tools, or known working platform details change. Explain why the dependency change is needed and verify that the environment still resolves.

## Repository map

Keep new work in the existing structure.

### `experiments`

This folder contains the scientific notebooks, parameter files, models, data products, plots, and validation results.

The four main experiment areas are listed below.

* `model_predict_cavity_claw_RouteMeander_eigenmode` contains the cavity claw resonator work.
* `model_predict_coupler_NCap_cap_matrix` contains the NCap coupler capacitance work.
* `model_predict_qubit_TransmonCross_cap_matrix` contains the transmon cross capacitance work.
* `model_predict_qubit_TransmonCross_Hamiltonian_params` contains the transmon Hamiltonian and inverse design work.

Keep experiment specific configuration in the existing `parameters` files inside that experiment.

Keep generated CSV files, plots, model outputs, scaler files, tuner results, and validation outputs with the experiment that produced them. Follow the local folder pattern instead of creating a new global output area.

Some older outputs live beside their notebooks while newer work uses `metadata` and `results` folders. Preserve both layouts during unrelated work.

Preserve the notebook numbering because it communicates workflow order. The common sequence begins with data analysis, continues through training and results, and then moves into surrogate and validation studies. The transmon Hamiltonian experiment starts later in the numbering because it shares earlier preparation work.

The transmon Hamiltonian experiment is the most developed area. Its later notebooks include nearest neighbor stress testing, result reporting, runtime benchmarks, data amount sweeps, architecture sweeps, and multi seed diagnostics.

Two small Keras models are tracked in the transmon Hamiltonian model folder. They are `best_keras_model_model2_surrogate.keras` and `surrogate_loss_2in_3out_best_model.keras`. Other models and supplemental files may live outside Git. Do not assume missing external data, models, or scalers are available.

Treat feature order, saved column metadata, units, categorical encodings, dataset boundaries, random seeds, and model to scaler pairings as part of the scientific contract. Do not change them unless the task explicitly changes that contract.

Do not replace the two tracked models or change their serialization format through incidental notebook execution.

### `figures`

This folder contains figure source files, generation scripts, and checked in outputs.

Paper figure scripts live under `figures/paper`. Run only the script related to the requested figure. Review every changed output before keeping it because figure scripts can rewrite tracked artifacts.

### `docs`

This folder contains setup notes and reference material.

Update documentation when a workflow, dependency, expected artifact, or scientific convention changes. Keep setup instructions friendly to a new user who has limited machine learning experience.

### Root files

Keep general project orientation in `README.md`.

Keep normal environment dependencies in `environment.yml` and Fermilab GPU dependencies in `environment-eaf-gpu.yml`.

Honor `.gitignore`. Do not force ignored data, model, scaler, tuner, cache, or temporary files into Git without explicit approval.

## Preserve the current structure

Make focused changes in the smallest relevant area.

Do not move or rename notebooks, experiment folders, result folders, models, or figure outputs unless the task requires it.

Do not reorganize the repository as part of an unrelated change.

Do not clear notebook outputs or rewrite notebook metadata unless the task requires it.

Do not save incidental execution counts, reordered cells, or unrelated output changes.

Do not replace local parameter files with a new configuration framework unless the task explicitly asks for that redesign.

Follow existing naming and numbering patterns when adding a notebook, parameter variant, result, or figure.

Preserve unrelated user edits and generated artifacts. Never use destructive Git commands to clean the worktree.

## Choose the simplest useful solution

Use the least complex change that solves the requested problem and preserves scientific meaning.

Prefer a small function over a new framework.

Prefer an existing dependency over adding a similar one.

Prefer a local change over a broad refactor.

Avoid abstraction until it removes real repeated work.

Keep data flow visible. A new reader should be able to find where values come from, how they are transformed, and where results are saved.

## Write for humans

Use plain language in code, notebooks, documentation, comments, labels, and docstrings.

Write like a thoughtful teammate. Be warm, direct, and a little playful when it helps the reader remember the idea. Never let humor hide the scientific meaning.

Explain acronyms and machine learning terms at their first important use in each file or notebook. Then use the same term consistently.

State units, array shapes, expected columns, value ranges, and file formats whenever they affect correctness and are established by code, data, or project documentation. Never invent these details.

Make the difference between physical values and scaled model values explicit.

Make the difference between training, validation, and test data explicit.

Explain why a scientific choice was made when the code alone cannot show it.

## Human written code comments

Apply these rules to every human written code comment that an agent adds or rewrites. Changing nearby code does not require unrelated comment cleanup.

* Begin the human comment text with two hash marks followed by one space.
* Use plain language that sounds natural when read aloud.
* Never use a colon, semicolon, or a sequence of two hyphen characters in prose comments.
* Explain why the code exists or what a surprising choice protects against.
* Do not narrate obvious syntax.
* Keep comments short and place them beside the code they explain.
* Define unfamiliar machine learning terms for a new reader.
* Mention units, shapes, ranges, and file formats when they matter.
* Update or remove a comment when the behavior changes.
* Do not leave commented out code. Delete it when safe and rely on Git history.

Good comments look like this.

```python
## Scale each input so one large value does not drown out the others
## Save the best weights so a long training run is not lost
## Convert model values back into units a device designer can use
```

Do not rewrite the whole repository only to restyle old comments. Apply this style when code is already being changed.

The two hash rule applies only where hash marks are valid comments. Preserve required language syntax.

Shebang lines, encoding declarations, notebook cell markers, coverage directives, license headers, generated blocks, documentation directives, and tool directives such as type ignore, noqa, fmt, pylint, and pragma may require exact machine syntax. Treat them as machine directives rather than human comments. Exact machine syntax is exempt from the two hash rule and the prose punctuation rule.

If Ruff or pycodestyle is added later, disable only rule E266 when needed so the required two hash comment style is accepted.

Markdown cells, Markdown headings, ordinary documentation, and strings containing hash marks are not code comments. Apply the two hash rule only in code where hash marks are valid comment syntax.

## Docstrings

Use triple quoted docstrings. They do not begin with hash marks.

Write docstrings for public modules, classes, and functions. Also write them for private helpers when their purpose or scientific meaning is not obvious.

Start with one plain sentence that says what the code helps the user do.

Add only the details needed to use the code safely. This may include inputs, returned values, shapes, units, side effects, saved files, and likely errors.

Use short paragraphs instead of rigid generated sections. Do not use section styles that require colons or rows of hyphens.

Never use a colon, semicolon, or a sequence of two hyphen characters in human prose docstrings.

Include a small example only when it makes the behavior easier to understand.

Keep docstrings friendly and clear while remaining scientifically precise.

An opening docstring may look like this.

```python
"""Prepare model inputs that are easy for a new reader to inspect.

Each row represents one device and each column represents one measured value.
"""
```

Literal code, commands, command options, URLs, paths, equations, data labels, copied errors, generated text, and required markup may contain punctuation that human prose cannot use. Include them only when they help the reader and preserve their exact syntax.

## Notebook rules

Keep notebooks readable from top to bottom.

Use Markdown cells to explain the goal, inputs, important choices, and expected outputs before complex code.

Keep setup and imports near the beginning.

Keep paths and experiment settings easy to find.

Avoid hidden state. A fresh kernel should not depend on cells being run in a surprising order.

Do not run a full training job, tuner search, runtime sweep, or validation workflow merely to inspect a notebook.

Before running a notebook or script, inspect it for training, tuning, remote access, simulator calls, and file writes.

Some configurations use hundreds of epochs or up to 2000 tuner trials. Treat these as expensive work that requires a task specific reason.

When changing training behavior, begin with the smallest representative check that can expose basic errors. Clearly state when a full scientific run was not performed.

## External and optional workflows

The normal environment covers the core training, evaluation, and figure workflows.

The validation notebooks may require SQuADDS, Qiskit Metal, Quantum Metal tooling, and a licensed electromagnetic solver. This stack is intentionally outside the normal environment.

Do not attempt to install, configure, or run the external validation stack unless the task explicitly asks for it and the required machine is available.

Committed validation outputs can be used to regenerate figures and summaries without rerunning the solver.

The Fermilab GPU environment is optional. Do not require GPU hardware for changes that can be verified on the normal environment.

## Verification

Match verification effort to the task and prefer fast focused checks first.

For Python changes, parse or compile every changed Python file before declaring completion.

For notebook changes, confirm the notebook remains valid JSON and review the changed cells and outputs.

For environment changes, confirm the environment file resolves and verify the imports affected by the change.

For figure changes, run only the relevant figure script and inspect the generated output and Git diff.

For model or training changes, use a small representative run before any full training job. Record the data subset, random seed, epoch count, and result.

For documentation changes, verify commands, paths, environment names, and package claims against repository files.

Always inspect the final Git diff and status. Report every check that ran, its result, and anything that could not be verified.

Never claim that work passed a full training, GPU, or electromagnetic validation run unless that exact run completed successfully.

## Definition of done

A task is complete only when all relevant statements below are true.

1. The requested observable result is present.
2. New files and outputs follow the existing repository structure.
3. Relevant focused checks pass.
4. Generated outputs have been reviewed before they are kept.
5. Documentation matches the implemented workflow.
6. No unrelated files were changed.
7. Scientific assumptions and unverified limitations are reported plainly.
8. The final summary includes evidence rather than confidence alone.

When a check cannot run because data, software, hardware, or credentials are missing, say exactly what is missing and provide the strongest safe check that can run locally.

## Leave the repository better than you found it

When a task reveals a durable convention, confusing workflow, missing prerequisite, or reliable verification step, record it in the nearest relevant README or in this file.

Keep additions concise. A good harness removes repeated guessing and helps the next human or agent get useful work done sooner.
