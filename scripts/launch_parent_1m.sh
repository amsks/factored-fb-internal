#!/bin/bash
# Parent-critic 1M controls: 6 x 1M cube-single runs, 3 seeds per parent critic,
# one GPU per arm, one tmux window per arm. Modeled on launch_density_1m.sh.
#
#   bash scripts/launch_parent_1m.sh
#
# Arms (all --actor flowbc --domain cube_single):
#   rldp_s0 GPU0  rldp_s1 GPU1  rldp_s2 GPU2   (critic rldp, seeds 0/1/2)
#   psm_s0  GPU3  psm_s1  GPU4  psm_s2  GPU5   (critic psm,  seeds 0/1/2)
#
# There is NO wandb-mode flag; setup_wandb() reads WANDB_MODE from the
# environment, so offline mode is set via env var.
set -euo pipefail

REPO=/home/adityamohan/git/Factored-FB-seam
PY=/dev/shm/.venv-jax/bin/python
LOGDIR=/dev/shm/p1m_logs
SESSION=p1m

STEPS=1000000
EVAL_EVERY=100000
SAVE_EVERY=500000

# Exact-name guard: tmux has-session prefix-matches, so '=' pins the exact name.
tmux has-session -t "=${SESSION}" 2>/dev/null && { echo "session exists"; exit 1; }

[ -x "$PY" ] || { echo "missing jax venv python at $PY"; exit 1; }
mkdir -p "$LOGDIR"

# arm-name  critic  seed  gpu
ARMS=(
  "rldp_s0  rldp   0     0"
  "rldp_s1  rldp   1     1"
  "rldp_s2  rldp   2     2"
  "psm_s0   psm    0     3"
  "psm_s1   psm    1     4"
  "psm_s2   psm    2     5"
)

first=1
for arm in "${ARMS[@]}"; do
  read -r name critic seed gpu <<< "$arm"
  cmd="cd $REPO && \
CUDA_VISIBLE_DEVICES=$gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_MODE=offline \
$PY -m impls.main \
  --critic $critic \
  --actor flowbc \
  --domain cube_single \
  --seed $seed \
  --train_steps $STEPS \
  --eval_interval $EVAL_EVERY \
  --save_interval $SAVE_EVERY \
  2>&1 | tee $LOGDIR/$name.log"
  if [ "$first" -eq 1 ]; then
    tmux new-session -d -s "$SESSION" -n "$name" "$cmd"
    first=0
  else
    tmux new-window -t "=${SESSION}" -n "$name" "$cmd"
  fi
done

echo "launched tmux session '$SESSION' with 6 windows (rldp_s0..2 / psm_s0..2)"
echo "logs: $LOGDIR/<arm>.log   attach: tmux attach -t =$SESSION"
