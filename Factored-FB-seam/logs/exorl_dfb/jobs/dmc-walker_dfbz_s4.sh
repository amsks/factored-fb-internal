#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
export FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages
export EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets
export EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export WANDB_MODE=offline
export MUJOCO_GL=egl
exec /dev/shm/.venv-jax/bin/python -m impls.main \
  --critic density_fb_z --actor ddpgbc --domain dmc-walker --seed 4 \
  --auto_resume --train_steps 2000000 --eval_interval 250000 --save_interval 250000 \
  --eval_episodes 10 --num_video 0 --infer_cond_episodes 50000 --eval_on_inferred_cond \
  --save_dir exp/exorl_dfb/dmc-walker/density_fb_z \
  --override discount=0.98 --override 'measure_hit_spec="0:24:15.9"' \
  --override 'value_hidden_dims=[1024]' --override tau=0.001 \
  --override 'measure_density_b_hidden_dims=[256,256]' --override measure_density_archi=motivo \
  --override measure_density_b_norm=true --override lr=1e-4 \
  --override actor.motivo_archi=true --override 'actor.hidden_dims=[1024]' \
  --override actor.alpha=0.0 --override actor.q_noise_std=0.2 --override actor.q_noise_clip=0.3 \
  --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3 \
  --override actor.q_normalize=false
