#!/usr/bin/env bash
# Density-FB on ExoRL walker after the SAME audit that fixed FB
# (docs/notes/2026-09-15-fb-reference-audit.md). Two deltas remained against the
# td_jepa/metamotivo reference FB configuration:
#   forward tower: value_hidden_dims [512,512,512] = width 512 x 3 trunk layers  ->  [1024] = 1024 x 1
#   target polyak: tau 0.005                                                     ->  0.001
# Everything else already matched (batch 1024, lr 1e-4, discount 0.98, z_dim 50,
# B 256x2 normalized, Motivo shape). 1M steps, 3 seeds, same actor and eval protocol as
# exp/walker_fbref so the two are directly comparable.
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
mkdir -p logs/walker_dfbref
REF="--override 'value_hidden_dims=[1024]' --override tau=0.001 --override 'measure_density_b_hidden_dims=[256,256]' --override measure_density_archi=motivo --override measure_density_b_norm=true --override lr=1e-4"
ACT="--override actor.motivo_archi=true --override 'actor.hidden_dims=[1024]' --override actor.alpha=0.0 --override actor.q_noise_std=0.2 --override actor.q_noise_clip=0.3 --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3 --override actor.q_normalize=false"
BASE="FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python -m impls.main --critic density_fb_z --actor ddpgbc --auto_resume --train_steps 1000000 --eval_interval 250000 --save_interval 250000 --domain dmc-walker --save_dir exp/walker_dfbref --override discount=0.98 --override 'measure_hit_spec=\"0:24:15.9\"'"
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$(( (s + 4) % 8 )) bash -c "$BASE $REF $ACT --seed $s > logs/walker_dfbref/dfbref_s$s.log 2>&1" &
done
wait
echo DFBREF_DONE
