#!/usr/bin/env bash
#SBATCH --job-name=gciql-visual
#SBATCH --array=0-9
#SBATCH --partition=ai,tnt
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=/bigwork/nhwpmoha/Factored-FB/logs/slurm/gciql_visual_%A_%a.out
#SBATCH --error=/bigwork/nhwpmoha/Factored-FB/logs/slurm/gciql_visual_%A_%a.err

# scripts/slurm/gciql_visual_sweep.sh — GCIQL visual sweep (visual-cube-single-play-v0).
# Submit with: sbatch scripts/slurm/gciql_visual_sweep.sh
# Seeds 0-9 run in parallel. Each is a 500k-step JAX run with IMPALA encoder.
#
# Prerequisites:
#   make install-jax-bigwork   (once — installs JAX venv on Lustre)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

export MUJOCO_GL=egl
export WANDB_MODE=offline

source "$REPO/.venv/bin/activate"

SEED=$SLURM_ARRAY_TASK_ID

echo "[gciql-visual] node=$(hostname) seed=${SEED} gpu=${CUDA_VISIBLE_DEVICES:-unset}"

python run_gciql.py \
    --config-name cube_single_visual \
    storage=lustre_bigwork \
    seed="${SEED}" \
    wandb_mode=offline \
    run_group=factored-fb-gciql-visual \
    "hydra.run.dir=/bigwork/nhwpmoha/Factored-FB/logs/slurm/hydra_gciql_visual_s${SEED}"

echo "[gciql-visual] seed=${SEED} DONE"
