#!/bin/bash
#SBATCH -J evalcond-check
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
PY="${VENV:-$SCRATCH/.venv-jax-fb}/bin/python"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl WANDB_MODE=disabled PYTHONPATH="$PWD"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
$PY -m pytest tests/impls/test_eval_cond_route.py tests/impls/test_log_utils.py -q -p no:cacheprovider
