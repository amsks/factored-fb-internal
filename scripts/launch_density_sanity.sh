#!/bin/bash
# Density-critic sanity screen: 4 x 100k cube-single runs comparing the two new
# density critics against their parents, one GPU per arm, one tmux window per arm.
#
#   bash scripts/launch_density_sanity.sh
#
# Arms (all --actor flowbc --domain cube_single, seed 0):
#   rldp   GPU0    rldp_density  GPU1    psm  GPU2    psm_density  GPU3
#
# Flags verified against impls/main.py (2026-08-24): --critic --actor --domain
# --seed --train_steps --eval_interval --save_interval. There is NO wandb-mode
# flag; setup_wandb() (impls/utils/log_utils.py:73) reads WANDB_MODE from the
# environment, so offline mode is set via env var. eval_interval=25000 gives
# evals at steps 1, 25k, 50k, 75k, 100k (main.py also evals at i==1).
set -euo pipefail

REPO=/home/adityamohan/git/Factored-FB-seam
PY=/dev/shm/.venv-jax/bin/python
LOGDIR=/dev/shm/dsanity_logs
SESSION=dsanity

STEPS=100000
EVAL_EVERY=25000
SAVE_EVERY=100000
SEED=0

# Exact-name guard: tmux has-session prefix-matches, so '=' pins the exact name.
tmux has-session -t "=${SESSION}" 2>/dev/null && { echo "session exists"; exit 1; }

[ -x "$PY" ] || { echo "missing jax venv python at $PY"; exit 1; }
mkdir -p "$LOGDIR"

# arm-name  critic         gpu
ARMS=(
  "rldp   rldp          0"
  "rldpd  rldp_density  1"
  "psm    psm           2"
  "psmd   psm_density   3"
)

first=1
for arm in "${ARMS[@]}"; do
  read -r name critic gpu <<< "$arm"
  cmd="cd $REPO && \
CUDA_VISIBLE_DEVICES=$gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_MODE=offline \
$PY -m impls.main \
  --critic $critic \
  --actor flowbc \
  --domain cube_single \
  --seed $SEED \
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

echo "launched tmux session '$SESSION' with 4 windows (rldp/rldpd/psm/psmd)"
echo "logs: $LOGDIR/<arm>.log   attach: tmux attach -t =$SESSION"
