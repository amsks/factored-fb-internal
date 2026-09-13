#!/usr/bin/env bash
# Density-RLDP levers on cube-single at ortho_coef=1: train_b / object-coordinate input / regression index / all three.
# 4 configs x 3 seeds = 12 runs, tdjtab protocol. tmux session `levers`, one window per GPU, seed 0 first.
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
BASE="XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python -m impls.main --actor flowbc --auto_resume --train_steps 1000000 --eval_interval 100000 --save_interval 250000 --domain cube_single --save_dir exp/levers --override ortho_coef=1 --override measure_density_hard=true --override 'measure_hit_spec=\"19:22:0.4\"' --override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10"
CRITICS=(rldp_density_trainb rldp_density_oc rldp_density_reg rldp_density_full)
GPUS=(0 1 2 3 4 5 6 7)
declare -a Q; for g in "${GPUS[@]}"; do Q+=(""); done
# priority: the non-object-coordinate variants first (trainb, reg: 6 runs on GPUs 0-5), then oc and full queued behind them
i=0
for c in rldp_density_trainb rldp_density_reg; do for s in 0 1 2; do
  gi=$((i % 6)); gpu=${GPUS[$gi]}
  Q[$gi]+="CUDA_VISIBLE_DEVICES=$gpu $BASE --critic $c --seed $s > logs/levers/${c}_s$s.log 2>&1; "
  i=$((i+1))
done; done
for c in rldp_density_oc rldp_density_full; do for s in 0 1 2; do
  gi=$((i % 6)); gpu=${GPUS[$gi]}
  Q[$gi]+="CUDA_VISIBLE_DEVICES=$gpu $BASE --critic $c --seed $s > logs/levers/${c}_s$s.log 2>&1; "
  i=$((i+1))
done; done
tmux new-session -d -s levers -n init
for gi in "${!GPUS[@]}"; do tmux new-window -t levers -n "gpu${GPUS[$gi]}" "cd $HOME/git/Factored-FB-seam; ${Q[$gi]} echo DONE; sleep 999999"; done
tmux kill-window -t levers:init || true
echo "jobs: $i"
