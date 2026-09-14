#!/usr/bin/env bash
# ExoRL walker, TD-JEPA protocol: the OBJECTIVE-ONLY density retrofits (feature-index tasks, no
# hindsight regression) -- rldp_density_tdj at ortho 1 and hilp_density_tdj at discount 0.98 --
# 3 seeds each, same actor/eval settings as exp/walker_20260914.
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
ACT="--override actor.motivo_archi=true --override 'actor.hidden_dims=[1024]' --override actor.alpha=0.0 --override actor.q_noise_std=0.2 --override actor.q_noise_clip=0.3 --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3 --override actor.q_normalize=false"
BASE="FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python -m impls.main --actor ddpgbc --auto_resume --train_steps 500000 --eval_interval 100000 --save_interval 100000 --domain dmc-walker --save_dir exp/walker_feat --override discount=0.98 $ACT"
declare -A ARM=( [rldp_density_tdj]="--critic rldp_density_tdj --override rldp_phase1_steps=2000000 --override 'measure_hit_spec=\"0:24:15.9\"' --override ortho_coef=1" \
                 [hilp_density_tdj]="--critic hilp_density_tdj" )
GPUS=(3 4)
declare -a Q; for g in "${GPUS[@]}"; do Q+=(""); done
i=0
for s in 0 1 2; do for a in rldp_density_tdj hilp_density_tdj; do
  gi=$((i % 2)); gpu=${GPUS[$gi]}
  Q[$gi]+="CUDA_VISIBLE_DEVICES=$gpu $BASE ${ARM[$a]} --seed $s > logs/walker_feat/${a}_s$s.log 2>&1; "
  i=$((i+1))
done; done
tmux new-session -d -s walker_feat -n init
for gi in "${!GPUS[@]}"; do tmux new-window -t walker_feat -n "gpu${GPUS[$gi]}" "cd $HOME/git/Factored-FB-seam; ${Q[$gi]} echo DONE; sleep 999999"; done
tmux kill-window -t walker_feat:init || true
echo "jobs: $i"
