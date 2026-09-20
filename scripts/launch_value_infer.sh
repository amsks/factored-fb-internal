#!/usr/bin/env bash
# Model-based task-vector inference (scripts/exorl_value_infer.py) on the round-3 cheetah
# checkpoints, one process per (seed, task), sequential on one GPU. Results append to
# exp/zsearch/value_infer.jsonl, logs in exp/zsearch/logs/.
#   usage: bash scripts/launch_value_infer.sh <gpu> "<seed> <task>" ["<seed> <task>" ...]
#   e.g.   tmux new -d -s vi6 'bash scripts/launch_value_infer.sh 6 "2 walk" "2 run"'
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
GPU=$1; shift
export FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages
export EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=$GPU
PY=/dev/shm/.venv-jax/bin/python
mkdir -p exp/zsearch/logs
for job in "$@"; do
  set -- $job; seed=$1; task=$2
  run=exp/exorl_sqt/dmc-cheetah/dmc-cheetah_density_fb_z_ddpgbc_seed_$seed
  echo "[launch_value_infer] gpu $GPU seed $seed task $task $(date -u +%H:%M:%SZ)"
  $PY -m scripts.exorl_value_infer --restore_path $run --restore_epoch 1000000 --seed $seed \
      --override exorl_task=$task --tag sqt_s${seed}_${task} --out_jsonl exp/zsearch/value_infer.jsonl \
      ${EXTRA:-} > exp/zsearch/logs/value_infer_s${seed}_${task}.log 2>&1 || echo "[launch_value_infer] FAILED seed $seed task $task"
  grep "VALUE_INFER" exp/zsearch/logs/value_infer_s${seed}_${task}.log || true
done
echo "[launch_value_infer] done $(date -u +%H:%M:%SZ)"
