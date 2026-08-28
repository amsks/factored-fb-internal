#!/bin/bash
# 4-arm object-coordinate (oc) screen: 500k cube-single, seed 0, one GPU per
# arm, one tmux window per arm. Modeled on launch_diag_screen.sh.
#
#   bash scripts/launch_oc_screen.sh
#   GPUS="4 5 6 7" bash scripts/launch_oc_screen.sh   # override GPU assignment
#
# Arms (all --actor flowbc --domain cube_single --seed 0), default GPUs 0-3:
#   rldp_oc    GPU0  critic rldp_oc            (backward_input: hit_spec)
#   rldpd_oc   GPU1  critic rldp_density_oc
#   psm_oc     GPU2  critic psm_oc
#   psmd_oc    GPU3  critic psm_density_oc
#
# There is NO wandb-mode flag; setup_wandb() reads WANDB_MODE from the environment.
set -euo pipefail

REPO=/home/adityamohan/git/Factored-FB-seam
PY=/dev/shm/.venv-jax/bin/python
LOGDIR=/dev/shm/oc_logs
SESSION=oc

STEPS=500000
EVAL_EVERY=50000
SAVE_EVERY=250000

# GPU assignment, overridable: GPUS="4 5 6 7" bash scripts/launch_oc_screen.sh
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3}"
[ "${#GPU_LIST[@]}" -eq 4 ] || { echo "GPUS must list exactly 4 GPUs"; exit 1; }

# Exact-name guard: tmux has-session prefix-matches, so '=' pins the exact name.
tmux has-session -t "=${SESSION}" 2>/dev/null && { echo "session exists"; exit 1; }

[ -x "$PY" ] || { echo "missing jax venv python at $PY"; exit 1; }
mkdir -p "$LOGDIR"

# arm-name   critic
ARMS=(
  "rldp_oc   rldp_oc"
  "rldpd_oc  rldp_density_oc"
  "psm_oc    psm_oc"
  "psmd_oc   psm_density_oc"
)

first=1
for i in "${!ARMS[@]}"; do
  read -r name critic <<< "${ARMS[$i]}"
  gpu="${GPU_LIST[$i]}"
  cmd="cd $REPO && \
CUDA_VISIBLE_DEVICES=$gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_MODE=offline \
$PY -m impls.main \
  --critic $critic \
  --actor flowbc \
  --domain cube_single \
  --seed 0 \
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

echo "launched tmux session '$SESSION' with 4 windows on GPUs ${GPU_LIST[*]}"
echo "logs: $LOGDIR/<arm>.log   attach: tmux attach -t =$SESSION"
