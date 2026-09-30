#!/usr/bin/env bash
# Ben's actor-readout ablation (2026-09-24): Density-FB (density_fb_z, cube-single) with the
# actor ascending FB's linear readout F(s,a,z)^T z instead of the reward averaged under the
# predicted successor distribution. Same critic training; fresh same-code baseline alongside.
# 2 arms x 3 seeds, 500k steps, eval every 100k (the table estimator = mean of the last 3 evals).
#   bash scripts/launch_fz_ablation.sh          # tmux session fz_ablation, GPUs 0-5
set -euo pipefail
REPO="${REPO:-$HOME/git/Factored-FB-seam}"
PY="${PY:-/dev/shm/.venv-jax/bin/python}"
cd "$REPO"
mkdir -p logs/fz_ablation
ENVV="XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl"
BASE="$PY -m impls.main --actor flowbc --auto_resume --train_steps 500000 --eval_interval 100000 \
--save_interval 250000 --critic density_fb_z --domain cube_single \
--override measure_density_z_actor_reward=latent_centered --override measure_density_hard=true \
--override 'measure_hit_spec=\"19:22:0.4\"' --override eval_relabel_size=10000 \
--override eval_relabel_shift=1.0 --override eval_relabel_episodes=10"
tmux new-session -d -s fz_ablation -n idle 2>/dev/null || true
gpu=0
for arm in softmax linear_fz; do
  for seed in 0 1 2; do
    cmd="CUDA_VISIBLE_DEVICES=$gpu $ENVV $BASE --seed $seed --save_dir exp/fz_ablation/$arm \
--override measure_density_actor_value=$arm > logs/fz_ablation/${arm}_s${seed}.log 2>&1"
    tmux new-window -t fz_ablation -n "${arm}_s${seed}" "cd $REPO && $cmd"
    echo "gpu $gpu: $arm seed $seed"
    gpu=$((gpu + 1))
  done
done
