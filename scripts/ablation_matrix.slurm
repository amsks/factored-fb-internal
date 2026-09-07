#!/usr/bin/env bash
#SBATCH --job-name=matrix
#SBATCH --array=0-119
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/matrix_%a.out
# Goal-vs-z ablation matrix as a slurm array. The job list is FROZEN at submit time:
#   TIER=1 JOBS_ONLY=1 bash scripts/launch_ablation_matrix.sh > jobs_tier1.cmds
#   sbatch --array=0-$(( $(wc -l < jobs_tier1.cmds) - 1 )) --export=ALL,JOBS_FILE=jobs_tier1.cmds scripts/ablation_matrix.slurm
# (Regenerating the list inside each array task shifts indices as earlier runs finish and
# skips the tail -- the 2026-09-07 cluster finding. Never regenerate inside the task.)
set -euo pipefail
export REPO="${REPO:-$HOME/git/Factored-FB-seam}"
export PY="${PY:-$REPO/.venv-jax/bin/python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl
cd "$REPO"; mkdir -p logs
: "${JOBS_FILE:?set JOBS_FILE to a frozen command list (see header)}"
mapfile -t CMDS < "$JOBS_FILE"
[ "${SLURM_ARRAY_TASK_ID}" -lt "${#CMDS[@]}" ] || { echo "index ${SLURM_ARRAY_TASK_ID} beyond ${#CMDS[@]} jobs"; exit 0; }
echo ">>> ${CMDS[$SLURM_ARRAY_TASK_ID]}"
eval "${CMDS[$SLURM_ARRAY_TASK_ID]}"
