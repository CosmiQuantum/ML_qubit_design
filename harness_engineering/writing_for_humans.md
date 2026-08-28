# Writing for humans

## Why this file exists

This file keeps code comments, docstrings, notebook prose, and beginner friendly writing consistent.

Its source is the durable writing preference established for this repository.

Read this file when adding or rewriting code comments, docstrings, notebook prose, documentation, labels, or machine learning explanations.

Review these rules when the project adopts a formatter, linter, or documentation standard. Remove exceptions when their related tool is no longer used.

These rules remain active while they reflect the repository owner preference. Replace them when a new durable writing standard is explicitly adopted.

## General writing

Use plain language that sounds like a thoughtful teammate.

Keep explanations warm, direct, and a little playful when that helps memory. Never let humor hide scientific meaning or make an error unclear.

Explain acronyms and machine learning terms at their first important use in each file or notebook. Then use the same term consistently.

State units, array shapes, expected columns, value ranges, and file formats when they affect correctness and are established by code, data, or project documentation.

Never invent shapes, units, ranges, scaling behavior, or scientific claims.

Make the difference between physical values and scaled model values explicit.

Make the difference between training, validation, and test data explicit.

Explain why a scientific choice was made when the code alone cannot show it.

## Human written comments

Apply these rules to each human written code comment that an agent adds or rewrites. Changing nearby code does not require unrelated comment cleanup.

* Begin the human comment text with two hash marks followed by one space.
* Use plain language that sounds natural when read aloud.
* Never use a colon, semicolon, or a sequence of two hyphen characters in human prose comments.
* Explain why the code exists or what a surprising choice protects against.
* Do not narrate obvious syntax.
* Keep comments short and place them beside the code they explain.
* Define unfamiliar machine learning terms for a new reader.
* Mention units, shapes, ranges, and file formats when they matter.
* Update or remove a comment when behavior changes.
* Do not leave commented out code. Delete it when safe and rely on Git history.

Good comments look like this.

```python
## Scale each input so one large value does not drown out the others
## Save the best weights so a long training run is not lost
## Convert model values back into units a device designer can use
```

Do not rewrite the whole repository only to restyle old comments.

The two hash rule applies only in code where hash marks are valid comment syntax.

Shebang lines, encoding declarations, notebook cell markers, coverage directives, license headers, generated blocks, documentation directives, and tool directives may require exact machine syntax.

Type ignore, noqa, fmt, pylint, and pragma directives are machine instructions rather than human comments.

Exact machine syntax is exempt from the two hash rule and the prose punctuation rule.

If Ruff or pycodestyle is added later, disable only rule E266 when needed so the two hash style is accepted.

Markdown cells, Markdown headings, ordinary documentation, and strings containing hash marks are not code comments.

## Docstrings

Use triple quoted docstrings. They do not begin with hash marks.

Write docstrings for public modules, classes, and functions. Also write them for private helpers when their purpose or scientific meaning is not obvious.

Start with one plain sentence that says what the code helps the user do.

Add only details needed to use the code safely. This may include inputs, returned values, shapes, units, side effects, saved files, and likely errors.

Use short paragraphs instead of rigid generated sections.

Never use a colon, semicolon, or a sequence of two hyphen characters in human prose docstrings.

Include a small example only when it makes behavior easier to understand.

Keep docstrings friendly and clear while remaining scientifically precise.

```python
"""Prepare model inputs that are easy for a new reader to inspect.

Each row represents one device and each column represents one measured value.
"""
```

Literal code, commands, command options, URLs, paths, equations, data labels, copied errors, generated text, and required markup may keep their exact punctuation.
