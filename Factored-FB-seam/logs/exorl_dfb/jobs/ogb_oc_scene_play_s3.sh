#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export WANDB_MODE=offline
export MUJOCO_GL=egl
exec /dev/shm/.venv-jax/bin/python -m impls.main \
  --actor flowbc --auto_resume \
  --train_steps 1000000 --eval_interval 100000 --save_interval 250000 \
  --critic density_fb_z --domain scene_play --seed 3 --save_dir exp/tdjtab \
  --override measure_density_z_actor_reward=latent_centered \
  --override measure_density_hard=true \
  --override 'measure_hit_spec="19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6"' \
  --override eval_relabel_size=10000 \
  --override eval_relabel_shift=1.0 \
  --override eval_relabel_episodes=10
