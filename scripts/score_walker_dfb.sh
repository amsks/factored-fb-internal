#!/usr/bin/env bash
# Score exp/walker_dfb (FB at its tuned ortho 1000 = fb_tdj, and Density-FB = density_fb_z)
# under the same ExoRL protocol as exp/walker_20260914: 4 td_jepa tasks, 10 episodes,
# task vector inferred from 10,000 reward-relabelled transitions, 500k checkpoints.
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
mkdir -p logs/walker_dfb exp/walker_dfb/dmc-walker
for c in fb_tdj density_fb_z; do
  mkdir -p exp/walker_dfb/dmc-walker/$c
  for s in 0 1 2; do
    src="$PWD/exp/walker_dfb/dmc-walker_${c}_ddpgbc_seed_${s}"
    [ -d "$src" ] && ln -sfn "$src" "exp/walker_dfb/dmc-walker/$c/dmc-walker_${c}_ddpgbc_seed_${s}"
  done
done
export FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl
for c in fb_tdj density_fb_z; do
  CUDA_VISIBLE_DEVICES=${GPU:-3} /dev/shm/.venv-jax/bin/python scripts/exorl_paper_eval.py \
    --root exp/walker_dfb --critic $c --actor ddpgbc --domains dmc-walker --seeds 0 1 2 \
    --steps 500000 --episodes 10 --inference-samples 10000 --task-set tdjepa \
    --out exp/paper_eval_walker_dfb.jsonl > logs/walker_dfb/score_$c.log 2>&1 || echo "FAIL $c"
done
echo WALKER_DFB_SCORE_DONE
