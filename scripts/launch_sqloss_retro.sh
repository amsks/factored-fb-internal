#!/usr/bin/env bash
# Density-RLDP (regression tasks, ortho 1) and Density-HILP (regression tasks) with squared error
# on the softmax probabilities instead of cross-entropy, cube-single, 3 seeds each.
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
BASE="XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python -m impls.main --actor flowbc --auto_resume --train_steps 1000000 --eval_interval 100000 --save_interval 250000 --domain cube_single --save_dir exp/sqloss_retro --override measure_density_hard=true --override 'measure_hit_spec=\"19:22:0.4\"' --override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10"
declare -A ARM=( [rldp_density_reg_sq]="--critic rldp_density_reg --override ortho_coef=1 --override rldp_density_loss=sq" \
                 [hilp_density_reg_sq]="--critic hilp_density_reg --override hilp_density_loss=sq" )
GPUS=(6 7 2)
declare -a Q; for g in "${GPUS[@]}"; do Q+=(""); done
i=0
for a in rldp_density_reg_sq hilp_density_reg_sq; do for s in 0 1 2; do
  gi=$((i % 3)); gpu=${GPUS[$gi]}
  Q[$gi]+="CUDA_VISIBLE_DEVICES=$gpu $BASE ${ARM[$a]} --seed $s > logs/sqloss_retro/${a}_s$s.log 2>&1; "
  i=$((i+1))
done; done
tmux new-session -d -s sqloss_retro -n init
for gi in "${!GPUS[@]}"; do tmux new-window -t sqloss_retro -n "gpu${GPUS[$gi]}" "cd $HOME/git/Factored-FB-seam; ${Q[$gi]} echo DONE; sleep 999999"; done
tmux kill-window -t sqloss_retro:init || true
echo "jobs: $i"
