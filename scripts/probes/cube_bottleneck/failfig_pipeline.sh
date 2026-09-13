#!/usr/bin/env bash
# Full pipeline for one env: rollouts (fig A) -> banks -> predictions (fig B, C).
# usage: failfig_pipeline.sh <env> <gpu>
set -euo pipefail
ENV=$1; GPU=$2
cd /home/adityamohan/git/Factored-FB-seam
export CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl WANDB_MODE=offline
PY=/dev/shm/.venv-jax/bin/python
S=scripts/probes/cube_bottleneck/failure_figs.py
L=logs/failfig; mkdir -p $L
$PY $S rollouts --env $ENV --episodes 20 2>&1 | grep -v "Warn\|gym.logger" > $L/${ENV}_rollouts.log
$PY $S banks --env $ENV 2>&1 | grep -v "Warn\|gym.logger" > $L/${ENV}_banks.log
# figure B: both methods at 1M, all three banks; figure C: carrying bank at earlier checkpoints
$PY $S predict --env $ENV --critics fb_tdj       --epoch 1000000 2>&1 | grep -v "Warn\|gym.logger" > $L/${ENV}_predict_fb_1M.log &
$PY $S predict --env $ENV --critics density_fb_z --epoch 1000000 2>&1 | grep -v "Warn\|gym.logger" > $L/${ENV}_predict_density_1M.log &
( for E in 250000 500000 750000; do
    $PY $S predict --env $ENV --critics fb_tdj --epoch $E --banks carrying 2>&1 | grep -v "Warn\|gym.logger" > $L/${ENV}_predict_fb_${E}.log
  done ) &
( for E in 250000 500000 750000; do
    $PY $S predict --env $ENV --critics density_fb_z --epoch $E --banks carrying 2>&1 | grep -v "Warn\|gym.logger" > $L/${ENV}_predict_density_${E}.log || true
  done ) &
wait
echo "PIPELINE DONE $ENV" >> $L/${ENV}_rollouts.log
