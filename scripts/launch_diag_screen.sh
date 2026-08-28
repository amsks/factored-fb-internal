#!/bin/bash
# 6-arm diagnosis screen: 500k cube-single, seed 0, one GPU per arm, one tmux
# window per arm. Modeled on launch_density_1m.sh. GPU 7 is BUSY (probeiface).
#
#   bash scripts/launch_diag_screen.sh
#
# Arms (all --actor flowbc --domain cube_single --seed 0):
#   rldp_fix   GPU0  critic rldp              (bug-repair rerun: fixed WindowedDataset)
#   rldpd_fix  GPU1  critic rldp_density      (bug-repair rerun)
#   rldpd_gk   GPU2  critic rldp_density_gk   (Exp A: density_fb task interface)
#   densfb_z   GPU3  critic density_fb_z      (Exp B: spec-30 centered dual reward index)
#   psm_rep    GPU4  critic psm_repair        (torch sweep-winning cube config)
#   psmd_rep   GPU5  critic psm_density_repair
#
# There is NO wandb-mode flag; setup_wandb() reads WANDB_MODE from the environment.
set -euo pipefail

REPO=/home/adityamohan/git/Factored-FB-seam
PY=/dev/shm/.venv-jax/bin/python
LOGDIR=/dev/shm/diag_logs
SESSION=diag

STEPS=500000
EVAL_EVERY=50000
SAVE_EVERY=250000

# Exact-name guard: tmux has-session prefix-matches, so '=' pins the exact name.
tmux has-session -t "=${SESSION}" 2>/dev/null && { echo "session exists"; exit 1; }

[ -x "$PY" ] || { echo "missing jax venv python at $PY"; exit 1; }
mkdir -p "$LOGDIR"

# arm-name   critic              gpu
ARMS=(
  "rldp_fix   rldp                0"
  "rldpd_fix  rldp_density        1"
  "rldpd_gk   rldp_density_gk     2"
  "densfb_z   density_fb_z        3"
  "psm_rep    psm_repair          4"
  "psmd_rep   psm_density_repair  5"
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

echo "launched tmux session '$SESSION' with 6 windows"
echo "logs: $LOGDIR/<arm>.log   attach: tmux attach -t =$SESSION"
