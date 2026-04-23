#!/usr/bin/env python3
"""
Patch all notebook plotting cells to standardize font sizes
to the 9-12pt range with sans-serif (Helvetica/Arial) fonts.

Rules:
  - Plot titles and axis labels: fontweight='normal' (unbolded)
  - fontsize=7 → fontsize=9  (minimum 9pt)
  - Legend fontsize: minimum 9pt
  - Diagram cells (in ml_22 architecture diagram): keep bold
  - Font family: sans-serif with Helvetica/Arial (already set in rcParams)
"""

import json
import os
import re
import sys
import glob

def patch_source_lines(source_lines, notebook_path):
    """Patch a list of source lines from a notebook cell."""
    patched = []
    for line in source_lines:
        new_line = line

        # 1. Replace fontsize=7 with fontsize=9 (minimum size)
        new_line = re.sub(r'fontsize\s*=\s*7\b', 'fontsize=9', new_line)

        # 2. For set_title, set_xlabel, set_ylabel, suptitle —
        #    ensure fontweight='normal' is present (unless it's a diagram block)
        #    We detect diagram blocks by checking for FancyBboxPatch or BLOCK_STYLE
        #    which are markers of the architecture diagram cell.
        #    For regular plot cells, we add fontweight='normal' if not present.

        # Handle set_title with fontsize but no fontweight
        if re.search(r'\.(set_title|suptitle)\(', new_line):
            if 'fontsize' in new_line and 'fontweight' not in new_line:
                # Add fontweight='normal' after fontsize=...
                new_line = re.sub(
                    r"(fontsize\s*=\s*\d+\.?\d*)",
                    r"\1, fontweight='normal'",
                    new_line
                )

        # Handle set_ylabel / set_xlabel with fontsize but no fontweight
        if re.search(r'\.(set_ylabel|set_xlabel)\(', new_line):
            if 'fontsize' in new_line and 'fontweight' not in new_line:
                new_line = re.sub(
                    r"(fontsize\s*=\s*\d+\.?\d*)",
                    r"\1, fontweight='normal'",
                    new_line
                )

        patched.append(new_line)
    return patched


def is_diagram_cell(source_lines):
    """Check if a cell is a diagram/architecture visualization cell.
    These should keep bold text."""
    source_text = ''.join(source_lines)
    # Architecture diagram cells use FancyBboxPatch or BLOCK_STYLE
    diagram_markers = [
        'FancyBboxPatch',
        'BLOCK_STYLE',
        'draw_block',
        'arrow_style',
        'Paper-themed model',
    ]
    for marker in diagram_markers:
        if marker in source_text:
            return True
    return False


def patch_notebook(notebook_path):
    """Patch a single notebook file."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue

        source = cell.get('source', [])
        if not source:
            continue

        # Skip diagram cells (keep bold)
        if is_diagram_cell(source):
            print(f"  [SKIP diagram] {notebook_path}")
            continue

        # Check if this cell has any plotting code
        source_text = ''.join(source)
        if not any(keyword in source_text for keyword in [
            'fontsize', 'set_title', 'set_xlabel', 'set_ylabel',
            'suptitle', 'plt.rcParams'
        ]):
            continue

        new_source = patch_source_lines(source, notebook_path)
        if new_source != source:
            cell['source'] = new_source
            modified = True
            print(f"  [PATCHED] cell in {notebook_path}")

    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"  Saved: {notebook_path}")
    else:
        print(f"  No changes needed: {notebook_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Find all notebooks in the model directories
    notebook_patterns = [
        'model_predict_qubit-TransmonCross-Hamiltonian_params/ml_22_print_results_surrogate_defined_loss.ipynb',
        'model_predict_qubit-TransmonCross-Hamiltonian_params/ml_14_surrogate_stress_test.ipynb',
        'model_predict_qubit-TransmonCross-Hamiltonian_params/ml_11_train_keras_surrogate.ipynb',
        'model_predict_qubit-TransmonCross-Hamiltonian_params/validation_21_data_analysis.ipynb',
        'model_predict_cavity_claw_RouteMeander_eigenmode/ml_22_print_results_surrogate_defined_loss.ipynb',
        'model_predict_cavity_claw_RouteMeander_eigenmode/ml_11_train_keras_surrogate.ipynb',
        'model_predict_coupler_NCap_cap_matrix/ml_11_train_keras_surrogate.ipynb',
        'model_predict_coupler_NCap_cap_matrix/ml_21_train_keras_surrogate_defined_loss.ipynb',
        'model_predict_qubit_TransmonCross_cap_matrix/ml_11_train_keras_surrogate.ipynb',
    ]

    for pattern in notebook_patterns:
        full_path = os.path.join(base_dir, pattern)
        if os.path.exists(full_path):
            print(f"Processing: {pattern}")
            patch_notebook(full_path)
        else:
            print(f"Not found: {pattern}")


if __name__ == '__main__':
    main()
