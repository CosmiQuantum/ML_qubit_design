#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="cryo-modelling-env"
INSTALL_CMD="conda env create -f environment.yml"
VERIFY_CMD="conda run -n ${ENV_NAME} python -c \"list(map(__import__, ['cairosvg', 'fitz', 'IPython', 'joblib', 'jsonschema', 'keras_tuner', 'matplotlib', 'numpy', 'pandas', 'pydot', 'reportlab', 'scipy', 'seaborn', 'sklearn', 'svglib', 'tensorflow', 'tensorflow_datasets', 'webcolors']))\""
START_CMD="conda run -n ${ENV_NAME} jupyter-lab"
RUN_START_COMMAND="${RUN_START_COMMAND-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "ML qubit design initialization"
echo "Working directory"
pwd
echo

if [[ ! -f "environment.yml" ]]
then
  echo "environment.yml was not found in the working directory"
  exit 1
fi

if ! command -v conda >/dev/null 2>&1
then
  echo "Conda is required before this repository can be initialized"
  echo "Install Conda or Miniconda and run this file again"
  exit 1
fi

echo "Preparing dependencies"
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"
then
  echo "The ${ENV_NAME} environment already exists"
else
  echo "Creating ${ENV_NAME} from environment.yml"
  eval "$INSTALL_CMD"
fi
echo

echo "Running baseline verification"
eval "$VERIFY_CMD"
echo "Baseline verification passed"
echo

if [[ "$RUN_START_COMMAND" == "1" ]]
then
  echo "Starting JupyterLab"
  exec bash -lc "$START_CMD"
fi

echo "Initialization complete"
echo "Start the project with"
echo "$START_CMD"
