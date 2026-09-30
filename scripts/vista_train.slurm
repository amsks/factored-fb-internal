#!/bin/bash
# Train one (critic x actor x domain x seed) run on TACC Vista's Grace-Hopper (GH200) queue.
#
# Submit from the repo root, after running scripts/vista_setup.sh once:
#
#   sbatch -A ASC26008 scripts/vista_train.slurm
#   CRITIC=gciql ACTOR=ddpgbc DOMAIN=mw-bin-picking SEED=1 sbatch -A ASC26008 scripts/vista_train.slurm
#
# Find your allocation name with:  sacctmgr -pns show assoc user=$USER format=Account
#
#SBATCH -J fb-train              # job name
#SBATCH -p gh                    # Vista GPU queue: 1x GH200 per node (use gh-dev for <2h tests)
#SBATCH -N 1                     # one node
#SBATCH -n 1                     # one task
#SBATCH -t 12:00:00              # walltime (gh max 48h; gh-dev max 2h)
#SBATCH -o logs/%x.o%j           # stdout -> logs/<jobname>.o<jobid>
#SBATCH -e logs/%x.e%j           # stderr
# 10 min before the walltime kill, SIGUSR1 is sent to the job STEPS (the srun'd python;
# no B: prefix means steps, not this batch shell). impls/main.py catches it between
# training steps, writes params_<step>.pkl, and exits cleanly, so a resubmission with
# AUTO_RESUME=1 continues from there instead of losing everything since the last
# SAVE_INTERVAL checkpoint. The 600 s margin has to cover at most one training step plus
# one eval pass plus the pickle write. If a run is killed harder than that (node crash,
# SIGKILL), the loss window falls back to SAVE_INTERVAL steps.
#SBATCH --signal=USR1@600
# #SBATCH -A your_allocation     # set via `sbatch -A ...` or uncomment and edit

set -euo pipefail
mkdir -p "$SLURM_SUBMIT_DIR/logs"
# Run from a FROZEN copy of the tree when one is given (scripts/launch_matrix.sh makes it).
# A 36-job matrix drains over hours; without this, jobs that start after an edit would run
# different code than jobs that started before it, and the grid would not be comparable.
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
echo "== code: $(pwd) =="

# ── Environment ─────────────────────────────────────────────────────────────
VENV="${VENV:-$SCRATCH/.venv-jax-fb}"
PY="$VENV/bin/python"

# Knobs (override on the sbatch command line, e.g. CRITIC=isfb ACTOR=fql sbatch ...)
CRITIC="${CRITIC:-fb}"
ACTOR="${ACTOR:-flowbc}"
DOMAIN="${DOMAIN:-mw-bin-picking}"
SEED="${SEED:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-300000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25000}"     # 12 eval points over 300k (default 100k is for 1e6)
SAVE_INTERVAL="${SAVE_INTERVAL:-50000}"     # 7 checkpoints: post-hoc video/probes need these
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SAVE_DIR="${SAVE_DIR:-$SCRATCH/factored-fb-exp}"
# AUTO_RESUME=1 passes --auto_resume: continue from the latest params_*.pkl in this run's
# own directory (fresh start if none). Safe to set unconditionally on resubmission -- the
# run dir is derived from CRITIC/ACTOR/DOMAIN/SEED/SAVE_DIR, so the same sbatch line that
# started a run resumes it. Default 0 keeps prior launches bit-identical.
AUTO_RESUME="${AUTO_RESUME:-0}"

# Data roots. Old bin-picking/button-press domains interpolate ${oc.env:FB_HF_EXPORT}; the
# regime-selectable domains (mw-pick-place/push, ms-*) interpolate ${oc.env:FB_DATA_ROOT},
# which the campaign launcher sets per data_type (expert / play-noise / play-resample).
export FB_HF_EXPORT="${FB_HF_EXPORT:-$SCRATCH/fb-data/hf_export}"
export FB_DATA_ROOT="${FB_DATA_ROOT:-$SCRATCH/fb-data/expert}"
# ExoRL (dmc-*) roots. Without these, env/exorl_env.py falls back to ~/exorl_data, which does
# not exist on this box, and the job dies in the loader after the node has been allocated.
# EXORL_CACHE_ROOT holds the relabelled-reward cache -- one .npy per (domain, algo, task,
# episode set), so the first job of a cell pays the mujoco relabelling cost and the rest hit
# the cache.
export EXORL_DATA_ROOT="${EXORL_DATA_ROOT:-$SCRATCH/exorl_data/exorl/datasets}"
export EXORL_CACHE_ROOT="${EXORL_CACHE_ROOT:-$SCRATCH/exorl_data/relabel_cache}"

# GH200 has one visible GPU per node.
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl                       # headless Metaworld/MuJoCo (main.py also sets this)
# SAPIEN (ManiSkill) Vulkan. The system NVIDIA ICD is MISSING on some Vista gh nodes (present
# on c642-052, absent on c642-011/031), so SAPIEN gets ErrorIncompatibleDriver there. Ship a
# self-owned ICD pointing at the driver lib (libGLX_nvidia.so.0), which every gh node has.
# Verified: env create + step + success + render all OK on a fresh node (job 879508).
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$SCRATCH/vulkan/nvidia_icd.json}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false # don't grab all HBM up front
# 0.6 not 0.9: ManiSkill's SAPIEN renderer needs GPU memory of its own in the same process,
# so JAX must not claim nearly all of HBM. GH200 has 96 GB, so 0.6 is ample for these agents.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}"
# Vista's gh compute nodes DO have outbound internet: the vista_smoke.slurm probe reached
# api.wandb.ai and completed an authenticated wandb.init from c642-071 (job 876172,
# 2026-07-29). So `WANDB_MODE=online sbatch ...` gives a live dashboard. The default stays
# offline because that degrades safely on a node that cannot route out — an unreachable
# api.wandb.ai makes an online init hang, which would burn the walltime doing nothing.
export WANDB_MODE="${WANDB_MODE:-offline}"
# wandb writes runs to $WANDB_DIR/wandb/. Pin it to SAVE_DIR: the default is cwd-relative, and
# cwd is CODE_DIR (the frozen snapshot), so offline runs would scatter one directory per matrix
# launch and the `wandb sync` hint at the bottom of this script would find nothing.
export WANDB_DIR="$SAVE_DIR"
mkdir -p "$WANDB_DIR"

echo "== host: $(hostname)  job: ${SLURM_JOB_ID:-none} =="
echo "== run: $CRITIC x $ACTOR on $DOMAIN seed $SEED for $TRAIN_STEPS steps =="
$PY -c 'import jax; print("jax", jax.__version__, "devices:", jax.devices())'

# OVERRIDE: space-separated dotlist config overrides (e.g. "value_latent_dim=50"), each passed
# as its own --override. Used by the metric-probe matrix to sweep latent dim / discount.
OVERRIDE="${OVERRIDE:-}"
ov_flags=()
for kv in $OVERRIDE; do ov_flags+=(--override "$kv"); done
# `if`, not `[ ... ] &&`: this script runs under set -e, and a failed test at the end of
# an AND list would kill the job before srun ever ran.
if [ "$AUTO_RESUME" = "1" ]; then ov_flags+=(--auto_resume); fi

# EXTRA_FLAGS: verbatim additional impls.main flags that are NOT config overrides
# Intentionally unquoted at the callsite so it word-splits into separate flags.
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# ── Train ───────────────────────────────────────────────────────────────────
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
