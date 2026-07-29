#!/bin/bash

#SBATCH --account=p32999
#SBATCH --partition=gengpu
#SBATCH --constraint=quest12
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=ml22-ml30-runtime
#
# ---- fill these in ----
#SBATCH --cpus-per-task=8
#SBATCH --mem=8GB
#SBATCH --time=00:08:00
# -----------------------
#
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Quest 12 GPU job for ml_22 + ml_30.
# Submit from the experiment directory:
#   mkdir -p logs
#   sbatch run_ml_22_30_quest12_gpu.sh

set -eo pipefail

echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results/runtime results/predictions results/validation

# ---- activate your Quest env ----
# module purge
source /hpc/software/mamba/24.3.0/etc/profile.d/conda.sh
conda activate cryo-modelling-env
# ---------------------------------

# Prefer conda's libstdc++ over the system one (fixes GLIBCXX_3.4.29 / NumPy import).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "Python: $(command -v python)"
echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
python -c "import tensorflow as tf; print('TF', tf.__version__); print('built_with_cuda', tf.test.is_built_with_cuda()); print('GPUs', tf.config.list_physical_devices('GPU'))"

echo "===== ml_22 ====="
python ml_22_print_results_surrogate_defined_loss.py

echo "===== ml_30 ====="
python ml_30_runtime_batch_benchmark.py \
  --combined-model "${SLURM_SUBMIT_DIR}/model/surrogate_loss_2in_3out_best_model.keras"

echo "Job end: $(date)"
