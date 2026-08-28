# Harness engineering

This folder contains focused task instructions for humans and coding agents.

The root `AGENTS.md` stays short and routes readers here only when a topic applies to their current work.

## Core root files

The tutorial recommends beginning with four discoverable root files.

* `AGENTS.md` provides the project overview, startup path, global constraints, and topic routes.
* `init.sh` prepares and checks the working environment.
* `progress.md` preserves verified state between sessions without using a tool specific filename.
* `feature_list.json` tracks work, verification steps, and evidence.

All four root files exist today. Read `progress.md` and `feature_list.json` before selecting work. Update both before ending a session.

## Focused instruction files

### `environment_and_dependencies.md`

Read this when changing dependencies, environment files, setup commands, Python, TensorFlow, or GPU support.

It records real version pins, known working evidence, unpinned packages, and dependency update rules.

### `repository_structure.md`

Read this when adding, moving, renaming, or saving notebooks, data, models, figures, results, documentation, or generated artifacts.

It maps the experiment folders and protects existing artifact layouts, notebook numbering, tracked models, and ignored outputs.

### `writing_for_humans.md`

Read this when adding or rewriting comments, docstrings, notebook prose, documentation, labels, or machine learning explanations.

It contains the plain language, two hash comment, punctuation, docstring, and beginner friendly writing rules.

### `scientific_workflows.md`

Read this when changing data preparation, features, targets, models, scalers, notebooks, training, tuning, figures, runtime studies, or external validation.

It protects scientific contracts and explains safe notebook, training, figure, GPU, and solver workflows.

### `verification.md`

Read this when changing code, notebooks, environments, figures, models, training behavior, or documentation claims, and whenever the correct checks are unclear.

It defines focused verification, evidence, the definition of done, and clean handoff expectations.

## Adding future rules

Add a rule to the root `AGENTS.md` only when it is nonnegotiable and applies to nearly every task.

Put task specific guidance in the narrowest matching file in this folder.

Create a new topic file only when none of the current files is a natural home.

Keep topic files around 50 to 150 lines. Add one route to the root `AGENTS.md` that says when the new file must be read.

Give each durable instruction a source, an applicability condition, and a review or removal condition.

Do not duplicate rules across files. Update or remove stale guidance as the repository changes.

Update this README whenever a focused instruction file is added, removed, renamed, or given a new purpose.
