# Scientific workflows

## Why this file exists

This file protects the scientific meaning of data, models, notebooks, figures, and validation workflows.

Its sources are the repository READMEs, experiment organization, environment documentation, and tracked scientific artifacts.

Read this file when changing data preparation, features, targets, models, scalers, notebooks, training, tuning, figures, runtime studies, or external validation.

Review these rules when a scientific workflow is deliberately redesigned. Remove safeguards only when their underlying workflow no longer exists.

These rules remain active while the named data, model, notebook, figure, and validation workflows exist. Replace them when those workflows are deliberately retired or redesigned.

## Project model flow

This repository uses forward models to predict electromagnetic or Hamiltonian behavior from geometry.

The transmon Hamiltonian work also uses an inverse model to predict geometry from requested Hamiltonian values.

A frozen forward surrogate checks the inverse prediction in Hamiltonian space.

Explain these ideas in plain language whenever they matter to a task. Do not assume a new reader knows terms such as surrogate model, inverse design, scaler, tuner, epoch, loss, or one hot encoding.

## Scientific contracts

Treat feature order, saved column metadata, units, categorical encodings, dataset boundaries, random seeds, and model to scaler pairings as part of the scientific contract.

Do not change a scientific contract unless the task explicitly requires that change.

Keep data flow visible. A new reader should be able to find where values come from, how they are transformed, and where results are saved.

Name the scaling method and expected range when known. Explain how values return to physical units when the code supports it.

Do not mix training, validation, and test data.

Do not assume external models, data, scalers, or solver software are available.

Do not replace tracked models or change their serialization format through incidental notebook execution.

## Notebook work

Keep notebooks readable from top to bottom.

Use Markdown cells to explain the goal, inputs, important choices, and expected outputs before complex code.

Keep setup and imports near the beginning. Keep paths and experiment settings easy to find.

Avoid hidden state. A fresh kernel should not depend on cells being run in a surprising order.

Before running a notebook or script, inspect it for training, tuning, remote access, simulator calls, and file writes.

Do not run all cells, clear outputs, rewrite metadata, or save incidental execution counts during an unrelated edit.

## Training and tuning

Do not run a full training job, tuner search, runtime sweep, architecture sweep, multi seed study, or validation workflow merely to inspect code.

Some configurations use hundreds of epochs or up to 2000 tuner trials. Treat these as expensive work that requires a task specific reason.

When changing training behavior, begin with the smallest representative check that can expose basic errors.

Record the data subset, random seed, epoch count, and result for a representative training check.

Clearly state when a full scientific run was not performed.

## Figures

Run only the figure script related to the requested output.

Review every changed figure and generated artifact before keeping it because figure scripts can rewrite tracked outputs.

Use committed validation outputs to regenerate figures and summaries when a new solver run is unnecessary.

## External validation

The normal environment covers the core training, evaluation, and figure workflows.

Validation notebooks may require SQuADDS, Qiskit Metal, Quantum Metal tooling, and a licensed electromagnetic solver.

Do not install, configure, or run that external stack unless the task explicitly requires it and the appropriate machine is available.

Never claim electromagnetic validation passed unless that exact workflow completed successfully.

The Fermilab GPU workflow is optional. Do not require it for changes that can be verified with the normal environment.
