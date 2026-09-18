#!/usr/bin/env bash
# Score the Phase-3 (support-calibrated radius) Density-FB ExoRL runs at 1M with the paper
# protocol: 10 episodes, 50k reward samples, td_jepa task set (walker 4th task = spin), the
# same scorer settings as exp/paper_eval_exorl_dfb_2M.jsonl. Usage: bash scripts/score_exorl_sup86.sh dmc-walker 0
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
DOM=$1; GPU=$2
export FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages
export EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl
CUDA_VISIBLE_DEVICES=$GPU /dev/shm/.venv-jax/bin/python scripts/exorl_paper_eval.py \
  --root exp/exorl_dfb_sup86 --critic density_fb_z --actor ddpgbc --domains "$DOM" --seeds 0 1 2 3 4 \
  --steps 1000000 --episodes 10 --inference-samples 50000 --task-set "${TASKSET:-tdjepa}" --jobs 4 \
  --out exp/paper_eval_exorl_sup86_1M.jsonl > "logs/exorl_score/sup86_${DOM}.log" 2>&1
echo "SCORED $DOM"
