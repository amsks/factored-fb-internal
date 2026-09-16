#!/usr/bin/env bash
# ExoRL walker at the configuration that already replicated Touati on this codebase:
# docs/2026-08-25-exorl-humenv-compendium.md "arch-shape" -- B hidden [256,256],
# measure_density_archi=motivo, measure_density_b_norm=true, lr 1e-4, 1M steps
# (walker density_fb 756.1 +- 45.6 against the published 760 +- 19).
# The 500k runs in exp/walker_dfb used the cube-tuned defaults and reached 553, which is why
# our ExoRL cells look weak. FB gets the same budget; note our FB has never replicated on
# ExoRL (its best published-protocol cell is 309 +- 190).
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
mkdir -p logs/walker_arch
ACT="--override actor.motivo_archi=true --override actor.hidden_dims=[1024] --override actor.alpha=0.0 --override actor.q_noise_std=0.2 --override actor.q_noise_clip=0.3 --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3 --override actor.q_normalize=false"
SHAPE="--override measure_density_b_hidden_dims=[256,256] --override measure_density_archi=motivo --override measure_density_b_norm=true --override lr=1e-4"
BASE="FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python -m impls.main --actor ddpgbc --auto_resume --train_steps 1000000 --eval_interval 250000 --save_interval 250000 --domain dmc-walker --save_dir exp/walker_arch --override discount=0.98 $ACT"
i=0
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$((i % 6)) bash -c "$BASE $SHAPE --critic density_fb_z --override 'measure_hit_spec=\"0:24:15.9\"' --seed $s > logs/walker_arch/dfb_s$s.log 2>&1" &
  i=$((i+1))
  CUDA_VISIBLE_DEVICES=$((i % 6)) bash -c "$BASE --override lr=1e-4 --critic fb --seed $s > logs/walker_arch/fb_s$s.log 2>&1" &
  i=$((i+1))
done
wait
echo WALKER_ARCH_DONE
