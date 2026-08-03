#!/bin/bash
# Smoke-test the phenomenon.tex kill-test code paths before burning full 500k runs:
#   1. static test suite for the touched seams (gciql q_bound, GCBilinearPsiValue psi_layer_norm)
#   2. 200-step train of gciql with q_bound=75 (Pred. mech.1 arm)
#   3. 200-step train of iql_bilinear d50 with psi_layer_norm=false (Pred. mech.2 arm)
#   4. rollout_probe collect (1 ep/task) + score on the existing biql-d50 seed-0 checkpoint
#      (Pred. mech.3 driver)
#
#   sbatch -A ASC26008 scripts/phen_smoke.slurm
#
#SBATCH -J phen-smoke
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

set -uo pipefail
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
echo "== code: $(pwd) =="

VENV="${VENV:-$SCRATCH/.venv-ms}"
PY="$VENV/bin/python"
RUNS="${RUNS:-$SCRATCH/factored-fb-metric-probe}"
SMOKE="$SCRATCH/phen-smoke-$SLURM_JOB_ID"
mkdir -p "$SMOKE"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline
export WANDB_DIR="$SMOKE"

FAIL=0
step () { echo "== [$1] =="; shift; "$@" || { echo "!! FAILED: $*"; FAIL=1; }; }

step "tests" $PY -m pytest -x -q \
    tests/impls/test_gcbilinear.py tests/impls/test_gciql_critic_static.py \
    tests/impls/test_iql_bilinear_static.py tests/impls/test_iql_bilinear_seam.py \
    tests/impls/test_config_utils.py tests/impls/test_main_config.py

step "gciql-qb75-200steps" $PY -m impls.main \
    --critic gciql --actor flowbc --domain cube-single --seed 0 \
    --override q_bound=75.0 --train_steps 200 --eval_interval 1000000 \
    --eval_episodes 1 --num_video 0 --save_dir "$SMOKE/gciql-qb75"

step "biql-nopsiln-200steps" $PY -m impls.main \
    --critic iql_bilinear --actor flowbc --domain cube-single --seed 0 \
    --override value_latent_dim=50 --override psi_layer_norm=false \
    --train_steps 200 --eval_interval 1000000 \
    --eval_episodes 1 --num_video 0 --save_dir "$SMOKE/biql-nopsiln"

RDIR="$RUNS/biql-d50/cube-single-play-v0_iql_bilinear_flowbc_seed_0"
step "rollout-collect" $PY -m eval.rollout_probe --mode collect \
    --critic iql_bilinear --actor flowbc --domain cube-single \
    --override value_latent_dim=50 --seed 0 \
    --restore_path "$RDIR" --restore_epoch 500000 \
    --episodes 1 --out "$SMOKE/rollouts_smoke.npz"

step "rollout-score-self" $PY -m eval.rollout_probe --mode score \
    --critic iql_bilinear --actor flowbc --domain cube-single \
    --override value_latent_dim=50 --seed 0 \
    --restore_path "$RDIR" --restore_epoch 500000 \
    --rollouts "$SMOKE/rollouts_smoke.npz" --out "$SMOKE/p2_smoke_self"

GDIR="$RUNS/gciql/cube-single-play-v0_gciql_flowbc_seed_0"
step "rollout-score-cross" $PY -m eval.rollout_probe --mode score \
    --critic gciql --actor flowbc --domain cube-single --seed 0 \
    --restore_path "$GDIR" --restore_epoch 500000 \
    --rollouts "$SMOKE/rollouts_smoke.npz" --out "$SMOKE/p2_smoke_cross"

echo "== SMOKE DONE (FAIL=$FAIL) out=$SMOKE =="
exit $FAIL
