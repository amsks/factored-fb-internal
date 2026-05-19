#!/usr/bin/env bash
#SBATCH --job-name=gciql-state
#SBATCH --array=0-9
#SBATCH --partition=ai,tnt
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/bigwork/nhwpmoha/Factored-FB/logs/slurm/gciql_state_%A_%a.out
#SBATCH --error=/bigwork/nhwpmoha/Factored-FB/logs/slurm/gciql_state_%A_%a.err

# scripts/slurm/gciql_state_sweep.sh — GCIQL state sweep (cube-single-play-v0).
# Submit with: sbatch scripts/slurm/gciql_state_sweep.sh
# Seeds 0-9 run in parallel as a Slurm array. Each is a 1M-step JAX run.
#
# Prerequisites:
#   make install-jax-bigwork   (once — installs JAX venv on Lustre)
#   (OGBench datasets downloaded by run_gciql.py via ~/.ogbench symlink)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

export MUJOCO_GL=egl
export WANDB_MODE=offline

source "$REPO/.venv/bin/activate"

SEED=$SLURM_ARRAY_TASK_ID

echo "[gciql-state] node=$(hostname) seed=${SEED} gpu=${CUDA_VISIBLE_DEVICES:-unset}"

python run_gciql.py \
    --config-name cube_single_state \
    storage=lustre_bigwork \
    seed="${SEED}" \
    wandb_mode=offline \
    run_group=factored-fb-gciql-state \
    "hydra.run.dir=/bigwork/nhwpmoha/Factored-FB/logs/slurm/hydra_gciql_state_s${SEED}"

echo "[gciql-state] seed=${SEED} DONE"
