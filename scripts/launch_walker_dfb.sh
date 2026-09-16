#!/usr/bin/env bash
# ExoRL walker under the TD-JEPA protocol for OUR OWN pair: plain FB and Density-FB
# (density_fb_z, the shipped reward-sample task index). The ExoRL block so far only has the
# RLDP/HILP retrofits; the paper's claim ("the density objective does not cost performance
# off manipulation") needs FB vs Density-FB itself. Same actor/eval settings as
# exp/walker_20260914 and exp/walker_feat so the four sets are comparable.
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
mkdir -p logs/walker_dfb
ACT="--override actor.motivo_archi=true --override 'actor.hidden_dims=[1024]' --override actor.alpha=0.0 --override actor.q_noise_std=0.2 --override actor.q_noise_clip=0.3 --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3 --override actor.q_normalize=false"
BASE="FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python -m impls.main --actor ddpgbc --auto_resume --train_steps 500000 --eval_interval 100000 --save_interval 100000 --domain dmc-walker --save_dir exp/walker_dfb --override discount=0.98 $ACT"
declare -A ARM=( [fb_tdj]="--critic fb_tdj" \
                 [density_fb_z]="--critic density_fb_z --override 'measure_hit_spec=\"0:24:15.9\"'" )
GPUS=(0 2 6)
declare -a Q; for g in "${GPUS[@]}"; do Q+=(""); done
i=0
for s in 0 1 2; do for a in fb_tdj; do
  gi=$((i % ${#GPUS[@]})); gpu=${GPUS[$gi]}
  Q[$gi]+="CUDA_VISIBLE_DEVICES=$gpu $BASE ${ARM[$a]} --seed $s > logs/walker_dfb/${a}_s$s.log 2>&1; "
  i=$((i+1))
done; done
tmux new-session -d -s walker_fbtdj -n init
for gi in "${!GPUS[@]}"; do tmux new-window -t walker_dfb -n "gpu${GPUS[$gi]}" "cd $HOME/git/Factored-FB-seam; ${Q[$gi]} echo DONE; sleep 999999"; done
tmux kill-window -t walker_fbtdj:init || true
echo "jobs: $i"
