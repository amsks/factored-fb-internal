#!/usr/bin/env bash
#SBATCH --job-name=matrix
#SBATCH --array=0-119
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/matrix_%a.out
# Goal-vs-z ablation matrix as a slurm array (set TIER=1|2|3 in the environment to restrict). One array index = one line of
#   JOBS_ONLY=1 bash scripts/launch_ablation_matrix.sh
# Adapt the partition/account lines to the cluster; set REPO/PY for its layout
# (see docs/specs/2026-09-03-cluster-run-matrix.md "Cluster ops"). Idempotent: rerunning the
# array skips finished runs and --auto_resume continues partial ones.
set -euo pipefail
export REPO="${REPO:-$HOME/git/Factored-FB-seam}"
export PY="${PY:-$REPO/.venv-jax/bin/python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl
cd "$REPO"; mkdir -p logs
mapfile -t CMDS < <(JOBS_ONLY=1 bash scripts/launch_ablation_matrix.sh)
[ "${SLURM_ARRAY_TASK_ID}" -lt "${#CMDS[@]}" ] || { echo "index ${SLURM_ARRAY_TASK_ID} beyond ${#CMDS[@]} pending jobs (already done)"; exit 0; }
echo ">>> ${CMDS[$SLURM_ARRAY_TASK_ID]}"
eval "${CMDS[$SLURM_ARRAY_TASK_ID]}"
