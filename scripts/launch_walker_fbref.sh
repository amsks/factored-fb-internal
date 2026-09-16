#!/usr/bin/env bash
# FB on ExoRL walker at the REFERENCE configuration, transcribed from
# ~/git/baselines/td_jepa/scripts/train/proprio/launch_fb_dmc.py (the td_jepa/metamotivo
# launcher our FB was ported from). Every knob below differs from our fb.yaml default; the
# audit table is in docs/notes/2026-09-15-fb-reference-audit.md.
#   batch 1024 (ours 256) | L_dim 256 (50) | F 1024x1 (512x2) | B 256x2 (512x4)
#   left encoder 256x0 (512x4) | taus 0.001 (0.005) | ortho 1 | discount 0.98
# Budget: the reference trains 3M steps; this runs 1M first so it is directly comparable to
# our existing 1M cells, with --auto_resume so it can be extended to 3M in place.
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
mkdir -p logs/walker_fbref
REF="--override batch_size=1024 --override L_dim=256 --override 'forward={hidden_dim: 1024, hidden_layers: 1}' --override 'backward={hidden_dim: 256, hidden_layers: 2, norm: true}' --override 'left_encoder={hidden_dim: 256, hidden_layers: 0, norm: true, identity: false}' --override f_target_tau=0.001 --override b_target_tau=0.001 --override ortho_coef=1"
ACT="--override actor.motivo_archi=true --override 'actor.hidden_dims=[1024]' --override actor.alpha=0.0 --override actor.q_noise_std=0.2 --override actor.q_noise_clip=0.3 --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3 --override actor.q_normalize=false"
BASE="FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python -m impls.main --critic fb --actor ddpgbc --auto_resume --train_steps 1000000 --eval_interval 250000 --save_interval 250000 --domain dmc-walker --save_dir exp/walker_fbref --override discount=0.98"
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$(( (s + 3) % 8 )) bash -c "$BASE $REF $ACT --seed $s > logs/walker_fbref/fbref_s$s.log 2>&1" &
done
wait
echo FBREF_DONE
