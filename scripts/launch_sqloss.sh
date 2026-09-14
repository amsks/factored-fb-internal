#!/usr/bin/env bash
# Density-FB with squared error on the softmax probabilities instead of cross-entropy
# (measure_density_loss=sq), cube-single + cube-double, 3 seeds, campaign protocol otherwise.
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
BASE="XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python -m impls.main --actor flowbc --auto_resume --train_steps 1000000 --eval_interval 100000 --save_interval 250000 --critic density_fb_z --save_dir exp/sqloss --override measure_density_loss=sq --override measure_density_z_actor_reward=latent_centered --override measure_density_hard=true --override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10"
declare -A HIT=( [cube_single]='19:22:0.4' [cube_double]='19:22:0.4,28:31:0.4' )
GPUS=(0 1 4 5)
declare -a Q; for g in "${GPUS[@]}"; do Q+=(""); done
i=0
for dom in cube_single cube_double; do for s in 0 1 2; do
  gi=$((i % 4)); gpu=${GPUS[$gi]}
  Q[$gi]+="CUDA_VISIBLE_DEVICES=$gpu $BASE --domain $dom --seed $s --override 'measure_hit_spec=\"${HIT[$dom]}\"' > logs/sqloss/${dom}_s$s.log 2>&1; "
  i=$((i+1))
done; done
tmux new-session -d -s sqloss -n init
for gi in "${!GPUS[@]}"; do tmux new-window -t sqloss -n "gpu${GPUS[$gi]}" "cd $HOME/git/Factored-FB-seam; ${Q[$gi]} echo DONE; sleep 999999"; done
tmux kill-window -t sqloss:init || true
echo "jobs: $i"
