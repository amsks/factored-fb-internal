#!/usr/bin/env bash
# Density-HILP regression-task lever on cube-single (+ object-coordinate variant), and the
# plain-baseline object-coordinate controls (rldp@ortho1 + hit_spec, hilp_tdjc + hit_spec).
# Priority: non-object-coordinate runs first. 4 configs x 3 seeds = 12 runs, tdjtab protocol.
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
BASE="XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python -m impls.main --actor flowbc --auto_resume --train_steps 1000000 --eval_interval 100000 --save_interval 250000 --domain cube_single --save_dir exp/hlevers --override measure_density_hard=true --override 'measure_hit_spec=\"19:22:0.4\"' --override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10"
declare -A EXTRA=( [hilp_density_reg]="" [hilp_density_reg_oc]="" \
                   [rldp_oc_ortho1]="--critic rldp --override ortho_coef=1 --override backward_input=hit_spec" \
                   [hilp_tdjc_oc]="--critic hilp_tdjc --override backward_input=hit_spec" )
GPUS=(2 3 4 5 6 7)
declare -a Q; for g in "${GPUS[@]}"; do Q+=(""); done
i=0
for c in hilp_density_reg rldp_oc_ortho1 hilp_density_reg_oc hilp_tdjc_oc; do for s in 0 1 2; do
  gi=$((i % 6)); gpu=${GPUS[$gi]}
  if [ -n "${EXTRA[$c]}" ]; then crit="${EXTRA[$c]}"; else crit="--critic $c"; fi
  Q[$gi]+="CUDA_VISIBLE_DEVICES=$gpu $BASE $crit --seed $s > logs/hlevers/${c}_s$s.log 2>&1; "
  i=$((i+1))
done; done
tmux new-session -d -s hlevers -n init
for gi in "${!GPUS[@]}"; do tmux new-window -t hlevers -n "gpu${GPUS[$gi]}" "cd $HOME/git/Factored-FB-seam; ${Q[$gi]} echo DONE; sleep 999999"; done
tmux kill-window -t hlevers:init || true
echo "jobs: $i"
