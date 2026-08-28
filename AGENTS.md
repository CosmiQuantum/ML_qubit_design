# ML qubit design agent guide

## Project overview

This repository contains notebook driven machine learning experiments for predicting Quantum Metal design parameters from target qubit, resonator, coupler, and Hamiltonian quantities.

Keep the science reproducible, preserve the existing repository shape, and make every change understandable to a human with limited machine learning knowledge.

## Start here

Complete these steps before changing anything.

1. Read this file completely.
2. Read `progress.md` for the latest verified state, blocker, and next action.
3. Read `DECISIONS.md` for important decisions and their reasons.
4. Read `feature_list.json` and select the unfinished feature with the lowest priority number. If the feature list is empty, add the first real requested feature before starting work.
5. Keep no more than one feature marked `in_progress`.
6. Read `README.md` for the project purpose and workflow.
7. Run `bash init.sh` and stop if initialization fails.
8. Inspect the current Git status and preserve all user changes.
9. Identify the smallest task that produces the requested result.
10. Read every topic document below that applies to the task.
11. Read the nearest README for each area the task will change.
12. Define observable completion criteria and relevant checks.

## Topic documents

Read only the documents that apply to the current task.

* [Environment and dependencies](harness_engineering/environment_and_dependencies.md) when changing dependencies, environment files, setup commands, Python, TensorFlow, or GPU support.
* [Repository structure](harness_engineering/repository_structure.md) when adding, moving, renaming, or saving notebooks, data, models, figures, results, documentation, or generated artifacts.
* [Writing for humans](harness_engineering/writing_for_humans.md) when adding or rewriting code comments, docstrings, notebook prose, documentation, labels, or machine learning explanations.
* [Scientific workflows](harness_engineering/scientific_workflows.md) when changing data preparation, features, targets, models, scalers, notebooks, training, tuning, figures, runtime studies, or external validation.
* [Verification and completion](harness_engineering/verification.md) when changing code, notebooks, environments, figures, models, training behavior, or documentation claims, and whenever the correct checks are unclear.

Read more than one topic document when a task crosses several areas.

## Global hard constraints

These rules apply to every task.

1. Preserve user changes and unrelated generated artifacts.
2. Never use destructive Git commands to clean the worktree.
3. Use the simplest focused change that solves the requested problem.
4. Do not reorganize unrelated files or introduce a new framework without a task specific reason.
5. Ask for direction when a missing choice would change scientific meaning.
6. Do not change dependencies before reading the environment topic document.
7. Do not move artifacts before reading the structure topic document.
8. Do not run training, tuning, benchmarks, GPU work, or external validation before reading the scientific workflow topic document.
9. Do not change scientific data contracts unless the task explicitly requires that change.
10. Do not claim completion without relevant verification evidence.
11. Report missing data, software, hardware, credentials, and unverified limitations plainly.
12. Before running code, identify its inputs, generated outputs, external files, and expensive operations.

## Keep instructions focused

Treat this file as a router rather than an encyclopedia. Keep it between 50 and 200 lines.

Add a rule here only when it is nonnegotiable and applies to nearly every task.

Put task specific guidance in the narrowest matching document under `harness_engineering`.

Create a new topic document only when no current document is a natural home. Keep topic documents around 50 to 150 lines.

When adding a topic document, add one routing line here that says exactly when it must be read. Also update `harness_engineering/README.md`.

Give every durable instruction a source, an applicability condition, and a review or removal condition.

For example, a dependency rule may name `environment.yml` as its source, apply during environment changes, and require review when that file changes.

Keep one source of truth. Do not copy the same rule into several files.

Audit instructions regularly. Remove obsolete, redundant, or contradictory guidance.

Convert repeated failures into mechanical checks when practical instead of collecting permanent historical warnings.

## Definition of done

A task is complete only when the requested observable result is present, applicable checks pass, generated outputs are reviewed, documentation matches the work, and no unrelated files changed.

The final report must list evidence, assumptions, and anything that could not be verified.

## End each session

Update `DECISIONS.md` only when the session made an important decision that future sessions need to preserve.

Update `feature_list.json` with the true status and evidence for the selected feature.

Update `progress.md` with completed work, verification, evidence, commits, files changed, risks, blockers, and the next best step.

Keep unfinished or blocked work visible. Do not mark work passing without evidence.

Commit a clean checkpoint when it is safe and authorized.
