#!/usr/bin/env bash
# Round 2 (2026-09-18) after scripts/launch_exorl_disc.sh:
#   maze: lr_actor 1e-6 scored 942/685/799/646 = 768 (seeds 0,1) -> add seeds 2,3,4 (five-seed table).
#   cheetah: FB reference reached 366 at 1M in our pipeline, Density-FB 232 at 2M -> method gap.
#     Two single-lever probes against the Phase-1 recipe (exp/exorl_dfb), 1M, 3 + 2 seeds:
#     (d) lr_actor 3e-4 -> 1e-4 (every reference runs cheetah's actor at 1e-4; ours is the ddpgbc yaml default)
#     (e) actor reward latent_centered -> kernel (exact bump on hindsight rows; the cube fix, never run on ExoRL)
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
mkdir -p logs/exorl_disc
ENV="FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl"
PY=/dev/shm/.venv-jax/bin/python
ACT="--override actor.motivo_archi=true --override 'actor.hidden_dims=[1024]' --override actor.alpha=0.0 --override actor.q_noise_std=0.2 --override actor.q_noise_clip=0.3 --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3 --override actor.q_normalize=false"
COMMON="--auto_resume --train_steps 1000000 --eval_interval 250000 --save_interval 250000 --eval_episodes 10 --num_video 0 --infer_cond_episodes 50000 --eval_on_inferred_cond"
DFB="--override 'value_hidden_dims=[1024]' --override tau=0.001 --override 'measure_density_b_hidden_dims=[256,256]' --override measure_density_archi=motivo --override measure_density_b_norm=true --override lr=1e-4"
g=0
for s in 2 3 4; do
  CUDA_VISIBLE_DEVICES=$g bash -c "$ENV $PY -m impls.main --critic density_fb_z --actor ddpgbc $COMMON --domain dmc-point-mass-maze --save_dir exp/exorl_dfb_mazelr --override discount=0.99 --override 'measure_hit_spec=\"0:2:0.03\"' --override lr_actor=1e-6 $DFB $ACT --seed $s > logs/exorl_disc/dfbz_maze_lr1e-6_s$s.log 2>&1" &
  g=$((g+1))
done
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g bash -c "$ENV $PY -m impls.main --critic density_fb_z --actor ddpgbc $COMMON --domain dmc-cheetah --save_dir exp/exorl_dfb_cheetahlr --override discount=0.98 --override 'measure_hit_spec=\"0:17:7.5\"' --override lr_actor=1e-4 $DFB $ACT --seed $s > logs/exorl_disc/dfbz_cheetah_lr1e-4_s$s.log 2>&1" &
  g=$((g+1))
done
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=$g bash -c "$ENV $PY -m impls.main --critic density_fb_z --actor ddpgbc $COMMON --domain dmc-cheetah --save_dir exp/exorl_dfb_cheetahkernel --override discount=0.98 --override 'measure_hit_spec=\"0:17:7.5\"' --override measure_density_z_actor_reward=kernel $DFB $ACT --seed $s > logs/exorl_disc/dfbz_cheetah_kernel_s$s.log 2>&1" &
  g=$((g+1))
done
wait
echo EXORL_DISC2_DONE
