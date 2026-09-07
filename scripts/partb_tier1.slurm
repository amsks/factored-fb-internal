#!/usr/bin/env bash
#SBATCH --job-name=partb1
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/partb1_%a.out
# Part B Tier 1 as a slurm array. One array index = one line of
#   JOBS_ONLY=1 bash scripts/launch_partb_tier1.sh
# Adapt the partition/account lines to the cluster; set REPO/PY for its layout
# (see docs/specs/2026-09-03-cluster-run-matrix.md "Cluster ops"). Idempotent: rerunning the
# array skips finished runs and --auto_resume continues partial ones.
set -euo pipefail
export REPO="${REPO:-$HOME/git/Factored-FB-seam}"
export PY="${PY:-$REPO/.venv-jax/bin/python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl
cd "$REPO"; mkdir -p logs
# Frozen list (see scripts/ablation_matrix.slurm header): JOBS_ONLY=1 bash scripts/launch_partb_tier1.sh > jobs_partb.cmds
: "${JOBS_FILE:?set JOBS_FILE to a frozen command list}"
mapfile -t CMDS < "$JOBS_FILE"
[ "${SLURM_ARRAY_TASK_ID}" -lt "${#CMDS[@]}" ] || { echo "index ${SLURM_ARRAY_TASK_ID} beyond ${#CMDS[@]} pending jobs (already done)"; exit 0; }
echo ">>> ${CMDS[$SLURM_ARRAY_TASK_ID]}"
eval "${CMDS[$SLURM_ARRAY_TASK_ID]}"
