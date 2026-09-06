#!/usr/bin/env bash
# Test 2 Pass A (the shared p_secure ground truth), sliced across states.
# EVERY worker must drive the SAME arm's actor -- p_secure is one fixed quantity that all
# arms are scored against -- so the critic/seed are pinned here, not varied.
set -euo pipefail
ENV_NAME="${1:-cube-single-play-v0}"
SESSION="${2:-cbgt}"
WIDTH="${WIDTH:-8}"
NSTATES="${NSTATES:-64}"
GT_CRITIC="${GT_CRITIC:-fb_ref}"
GT_SEED="${GT_SEED:-0}"
REPO="$HOME/git/Factored-FB-seam"
PY=/dev/shm/.venv-jax/bin/python

tmux has-session -t "=$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n idle
for ((s=0; s<NSTATES; s+=WIDTH)); do
  e=$((s+WIDTH)); (( e > NSTATES )) && e=$NSTATES
  win="gt-${s}-${e}"
  tmux kill-window -t "=$SESSION:=$win" 2>/dev/null || true
  tmux new-window -d -t "=$SESSION" -n "$win" \
    "cd $REPO && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
     XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu MUJOCO_GL=egl \
     $PY -m scripts.probes.cube_bottleneck.measure_probe --env $ENV_NAME \
       --critics $GT_CRITIC --seeds $GT_SEED --state-start $s --state-end $e \
     2>&1 | tee -a $REPO/analysis/cube_bottleneck/$ENV_NAME/log_gt_${s}_${e}.txt; \
     echo EXIT=\$?; sleep 86400"
done
echo "launched $(( (NSTATES + WIDTH - 1) / WIDTH )) Pass A workers in '$SESSION'"
