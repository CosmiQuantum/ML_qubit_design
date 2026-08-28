# Verification and completion

## Why this file exists

This file defines how to gather evidence and decide when work is complete.

Its sources are the repository workflows and the tutorial principle that completion must be supported by verification.

Read this file when changing code, notebooks, environments, figures, models, training behavior, or documentation claims, and whenever the correct checks are unclear.

Review these rules when the repository gains new tests, validation commands, or continuous integration. Replace manual checks with stronger mechanical checks when they become available.

These rules remain active while the listed checks reflect the available project tooling. Replace them when stronger standard commands or continuous integration become the source of truth.

## Verification approach

Match verification effort to the task and prefer fast focused checks first.

Run `bash init.sh` before implementation. Stop and repair the baseline when initialization fails.

For Python changes, parse or compile every changed Python file before declaring completion.

For notebook changes, confirm the notebook remains valid JSON and review changed cells, outputs, metadata, and execution counts.

For environment changes, confirm the environment resolves and verify imports affected by the change.

For figure changes, run only the relevant figure script and inspect the generated output and Git diff.

For model or training changes, use a small representative run before any full training job. Record the data subset, random seed, epoch count, and result.

For documentation changes, verify commands, paths, environment names, package claims, and workflow descriptions against repository files.

Always inspect the final Git diff and status.

Report every check that ran, its result, and anything that could not be verified.

Never claim that work passed a full training, GPU, figure, or electromagnetic validation run unless that exact run completed successfully.

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

## Clean handoff

Leave the repository in a state where another human or agent can continue without guessing.

Record durable conventions, missing prerequisites, and reliable verification steps in the narrowest relevant repository document.

Do not leave half finished work or unverified claims undocumented.
