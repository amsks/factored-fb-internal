#!/bin/bash
# Train one run on TACC Vista's GH200 queue (after scripts/vista_setup.sh).
#
#   CRITIC=fb ACTOR=ddpgbc DOMAIN=mw-bin-picking SEED=1 sbatch -A <alloc> scripts/vista_train.slurm
#
#SBATCH -J fb-train              # job name
#SBATCH -p gh                    # Vista GPU queue: 1x GH200 per node (use gh-dev for <2h tests)
#SBATCH -N 1                     # one node
#SBATCH -n 1                     # one task
#SBATCH -t 12:00:00              # walltime (gh max 48h; gh-dev max 2h)
#SBATCH -o logs/%x.o%j           # stdout -> logs/<jobname>.o<jobid>
#SBATCH -e logs/%x.e%j           # stderr
# SIGUSR1 10 min before walltime: impls/main.py saves a checkpoint and exits
#SBATCH --signal=USR1@600
# #SBATCH -A your_allocation     # set via `sbatch -A ...` or uncomment and edit

set -euo pipefail
mkdir -p "$SLURM_SUBMIT_DIR/logs"
# Run from a frozen copy of the tree when CODE_DIR is set
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
echo "== code: $(pwd) =="

# Environment
VENV="${VENV:-$SCRATCH/.venv-jax-fb}"
PY="$VENV/bin/python"

# Run settings (override on the sbatch command line)
CRITIC="${CRITIC:-fb}"
ACTOR="${ACTOR:-flowbc}"
DOMAIN="${DOMAIN:-mw-bin-picking}"
SEED="${SEED:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-300000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25000}"     # 12 eval points over 300k (default 100k is for 1e6)
SAVE_INTERVAL="${SAVE_INTERVAL:-50000}"     # 7 checkpoints: post-hoc video/probes need these
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SAVE_DIR="${SAVE_DIR:-$SCRATCH/factored-fb-exp}"
# AUTO_RESUME=1 resumes from the latest checkpoint in the run directory
AUTO_RESUME="${AUTO_RESUME:-0}"

# Data roots
export FB_HF_EXPORT="${FB_HF_EXPORT:-$SCRATCH/fb-data/hf_export}"
export FB_DATA_ROOT="${FB_DATA_ROOT:-$SCRATCH/fb-data/expert}"
# ExoRL datasets and relabelled-reward cache
export EXORL_DATA_ROOT="${EXORL_DATA_ROOT:-$SCRATCH/exorl_data/exorl/datasets}"
export EXORL_CACHE_ROOT="${EXORL_CACHE_ROOT:-$SCRATCH/exorl_data/relabel_cache}"

# GH200 has one visible GPU per node.
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl                       # headless Metaworld/MuJoCo (main.py also sets this)
# Self-owned Vulkan ICD for SAPIEN; some gh nodes lack the system one
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$SCRATCH/vulkan/nvidia_icd.json}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false # don't grab all HBM up front
# Leave GPU memory for the SAPIEN renderer
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}"
# Offline by default; an unreachable wandb server hangs an online init
export WANDB_MODE="${WANDB_MODE:-offline}"
# Keep wandb runs under SAVE_DIR
export WANDB_DIR="$SAVE_DIR"
mkdir -p "$WANDB_DIR"

echo "== host: $(hostname)  job: ${SLURM_JOB_ID:-none} =="
echo "== run: $CRITIC x $ACTOR on $DOMAIN seed $SEED for $TRAIN_STEPS steps =="
$PY -c 'import jax; print("jax", jax.__version__, "devices:", jax.devices())'

# OVERRIDE: space-separated dotlist config overrides
OVERRIDE="${OVERRIDE:-}"
ov_flags=()
for kv in $OVERRIDE; do ov_flags+=(--override "$kv"); done
# `if`, not `&&`: a failed test would exit under set -e
if [ "$AUTO_RESUME" = "1" ]; then ov_flags+=(--auto_resume); fi

# EXTRA_FLAGS: extra impls.main flags, word-split at the callsite
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# Train
srun $PY -m impls.main \
    --critic "$CRITIC" --actor "$ACTOR" --domain "$DOMAIN" \
    --seed "$SEED" --train_steps "$TRAIN_STEPS" \
    --eval_interval "$EVAL_INTERVAL" --save_interval "$SAVE_INTERVAL" \
    --eval_episodes "$EVAL_EPISODES" \
    "${ov_flags[@]}" \
    $EXTRA_FLAGS \
    --save_dir "$SAVE_DIR"

echo "== done. checkpoints/logs under $SAVE_DIR =="
echo "   sync wandb later from a login node:  wandb sync $SAVE_DIR/wandb/offline-run-*"
