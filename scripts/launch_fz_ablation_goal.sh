#!/usr/bin/env bash
# F^T z actor ablation on the GOAL-CONDITIONED Density-FB (density_fb, the model behind the
# OGBench cells of tab:benchmarks): the actor ascends F(s,a,g)^T B(g) instead of the exact goal
# reward averaged under the predicted successor distribution. Same critic; fresh baseline alongside.
# 2 arms x 3 seeds, cube-single, 500k steps, eval every 100k (table estimator = mean of last 3 evals).
set -euo pipefail
REPO="${REPO:-$HOME/git/Factored-FB-seam}"
PY="${PY:-/dev/shm/.venv-jax/bin/python}"
cd "$REPO"; mkdir -p logs/fz_ablation_goal
ENVV="XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl"
BASE="$PY -m impls.main --actor flowbc --auto_resume --train_steps 500000 --eval_interval 100000 \
--save_interval 250000 --critic density_fb --domain cube_single"
tmux new-session -d -s fz_ablation_goal -n idle 2>/dev/null || true
gpu=0
for arm in softmax linear_fz; do
  for seed in 0 1 2; do
    cmd="CUDA_VISIBLE_DEVICES=$gpu $ENVV $BASE --seed $seed --save_dir exp/fz_ablation_goal/$arm \
--override measure_density_actor_value=$arm > logs/fz_ablation_goal/${arm}_s${seed}.log 2>&1"
    tmux new-window -t fz_ablation_goal -n "${arm}_s${seed}" "cd $REPO && $cmd"
    echo "gpu $gpu: $arm seed $seed"; gpu=$((gpu + 1))
  done
done
