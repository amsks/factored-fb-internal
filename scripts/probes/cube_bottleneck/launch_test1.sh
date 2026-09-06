#!/usr/bin/env bash
# Test 1 (held-start eval), one tmux window per arm.
# Arms run as separate processes and write per-arm CSVs, so there is no shared-file race.
# CPU jax: the bottleneck is MuJoCo stepping, not linear algebra, and it keeps the probe
# off the GPUs entirely. Four BLAS threads per worker (spec section 4, "Cost").
set -euo pipefail

ENV_NAME="${1:-cube-single-play-v0}"
SESSION="${2:-cbprobe}"
EPISODES="${EPISODES:-50}"
EPISODES_H2="${EPISODES_H2:-10}"
REPO="$HOME/git/Factored-FB-seam"
PY=/dev/shm/.venv-jax/bin/python

case "$ENV_NAME" in
  cube-single-play-v0) ARMS="fb_ref:0 fb_ref:1 fb_ref:2 density_fb_state:0 density_fb_state:1 density_fb_state:2 density_fb:0 density_fb:1 density_fb:2 fb:0" ;;
  cube-double-play-v0) ARMS="fb_ref:0 density_fb_state:0 density_fb_state:1 density_fb_state:2 density_fb:0 density_fb:1 density_fb:2 fb:0" ;;
  *) echo "unknown env $ENV_NAME" >&2; exit 1 ;;
esac

tmux has-session -t "=$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n idle

for arm in $ARMS; do
  critic="${arm%%:*}"; seed="${arm##*:}"
  win="${ENV_NAME%%-play-v0}-${critic}-s${seed}"
  tmux kill-window -t "=$SESSION:=$win" 2>/dev/null || true
  tmux new-window -d -t "=$SESSION" -n "$win" \
    "cd $REPO && \
     OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
     XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu MUJOCO_GL=egl \
     $PY -m scripts.probes.cube_bottleneck.held_start_eval \
       --env $ENV_NAME --critics $critic --seeds $seed \
       --episodes $EPISODES --episodes-h2 $EPISODES_H2 \
     2>&1 | tee -a $REPO/analysis/cube_bottleneck/$ENV_NAME/log_${critic}_s${seed}.txt; \
     echo EXIT=\$?; sleep 86400"
done
echo "launched $(echo $ARMS | wc -w) windows in tmux session '$SESSION' for $ENV_NAME"
tmux list-windows -t "=$SESSION"
