#!/bin/bash
# Density-critic 1M finals: 6 x 1M cube-single runs, 3 seeds per density critic,
# one GPU per arm, one tmux window per arm. Modeled on launch_density_sanity.sh.
#
#   bash scripts/launch_density_1m.sh
#
# Arms (all --actor flowbc --domain cube_single):
#   rldpd_s0 GPU0  rldpd_s1 GPU1  rldpd_s2 GPU2   (critic rldp_density, seeds 0/1/2)
#   psmd_s0  GPU3  psmd_s1  GPU5  psmd_s2  GPU6   (critic psm_density,  seeds 0/1/2)
# GPU4 is skipped: it holds the running reference job (tmux session 'dref').
#
# There is NO wandb-mode flag; setup_wandb() reads WANDB_MODE from the
# environment, so offline mode is set via env var.
set -euo pipefail

REPO=/home/adityamohan/git/Factored-FB-seam
PY=/dev/shm/.venv-jax/bin/python
LOGDIR=/dev/shm/d1m_logs
SESSION=d1m

STEPS=1000000
EVAL_EVERY=100000
SAVE_EVERY=500000

# Exact-name guard: tmux has-session prefix-matches, so '=' pins the exact name.
tmux has-session -t "=${SESSION}" 2>/dev/null && { echo "session exists"; exit 1; }

[ -x "$PY" ] || { echo "missing jax venv python at $PY"; exit 1; }
mkdir -p "$LOGDIR"

# arm-name   critic         seed  gpu
ARMS=(
  "rldpd_s0  rldp_density   0     0"
  "rldpd_s1  rldp_density   1     1"
  "rldpd_s2  rldp_density   2     2"
  "psmd_s0   psm_density    0     3"
  "psmd_s1   psm_density    1     5"
  "psmd_s2   psm_density    2     6"
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

echo "launched tmux session '$SESSION' with 6 windows (rldpd_s0..2 / psmd_s0..2)"
echo "logs: $LOGDIR/<arm>.log   attach: tmux attach -t =$SESSION"
