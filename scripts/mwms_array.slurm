#!/bin/bash
# Slurm ARRAY wrapper for scripts/launch_mwms_baselines.sh (MODE=slurm): task N reads line N of
# $JOBS_FILE -- one run per line in scripts/vista_train.slurm's env-var contract
# (CRITIC/ACTOR/DOMAIN/SEED/OVERRIDE/TRAIN_STEPS/EVAL_INTERVAL/SAVE_INTERVAL/EVAL_EPISODES/
# NUM_VIDEO/AUTO_RESUME/SAVE_DIR/FB_DATA_ROOT/FB_HF_EXPORT/VENV) -- exports it and hands off to
# vista_train.slurm's body, so the environment (venv, EGL, Vulkan ICD, XLA memory fraction,
# offline wandb, USR1 checkpoint-on-walltime) is the one every other campaign used.
#
#   sbatch -A <alloc> --array=0-<N-1>%20 --export=ALL,JOBS_FILE=<abs path>,CODE_DIR=<repo> scripts/mwms_array.slurm
#
#SBATCH -J mwms
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o logs/%x.o%A_%a
#SBATCH -e logs/%x.e%A_%a
#SBATCH --signal=USR1@600
set -euo pipefail
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
: "${JOBS_FILE:?JOBS_FILE (written by launch_mwms_baselines.sh MODE=slurm) must be exported}"
line="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOBS_FILE")"
[ -n "$line" ] || { echo "no job line $SLURM_ARRAY_TASK_ID in $JOBS_FILE" >&2; exit 2; }
echo "== array task $SLURM_ARRAY_TASK_ID: $line =="
eval "export $line"
exec bash scripts/vista_train.slurm
