#!/usr/bin/env bash
# The same arch-shape recipe on the OTHER ExoRL domains, so the question "does Density-FB
# match the published FB column across environments?" has an answer per environment rather
# than only on walker. Usage: launch_exorl_archshape.sh <domain> <hit_spec> <gpu0>
# hit specs are "0:obs_dim:eps" for that domain's observation layout.
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
DOM=$1; HIT=$2; G0=${3:-0}
mkdir -p logs/exorl_arch
ACT="--override actor.motivo_archi=true --override actor.hidden_dims=[1024] --override actor.alpha=0.0 --override actor.q_noise_std=0.2 --override actor.q_noise_clip=0.3 --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3 --override actor.q_normalize=false"
SHAPE="--override measure_density_b_hidden_dims=[256,256] --override measure_density_archi=motivo --override measure_density_b_norm=true --override lr=1e-4"
BASE="FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python -m impls.main --actor ddpgbc --auto_resume --train_steps 1000000 --eval_interval 250000 --save_interval 250000 --domain $DOM --save_dir exp/exorl_arch --override discount=0.98 $ACT"
i=0
for s in 0 1 2; do
  g=$(( (G0 + i) % 8 ))
  CUDA_VISIBLE_DEVICES=$g bash -c "$BASE $SHAPE --critic density_fb_z --override 'measure_hit_spec=\"$HIT\"' --seed $s > logs/exorl_arch/${DOM}_dfb_s$s.log 2>&1" &
  i=$((i+1))
done
wait
echo "ARCH_${DOM}_DONE"
