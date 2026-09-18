#!/usr/bin/env bash
# Discriminating runs for the ExoRL cheetah / point-mass gap (2026-09-17).
# (a) FB at the walker-validated REFERENCE config (scripts/launch_walker_fbref.sh) on cheetah, 3 seeds:
#     if this reaches published FB (~500 domain mean) the pipeline is fine on cheetah and the
#     Density-FB gap is the method; if it lands near Density-FB (~230) the pipeline is the gap.
# (b) FB reference on point-mass-maze at the reference maze settings shared by td_jepa
#     (launch_fb_dmc.py sweep_pointmass: discount 0.99, lr_actor 1e-6) and the RLDP paper's FB
#     baseline (rldp_train_dmc.py:69-83: discount 0.99, lr_actor = lr_b = 1e-6), 3 seeds.
# (c) Density-FB (density_fb_z, Phase-1 recipe) on point-mass-maze with the ONE reference
#     maze change we never applied, lr_actor 3e-4 -> 1e-6, 2 seeds. Tests whether the
#     saturated bang-bang / collapsed TD3 actor of the 2M maze runs is the actor lr.
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
mkdir -p logs/exorl_disc
ENV="FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl"
PY=/dev/shm/.venv-jax/bin/python
REF="--override batch_size=1024 --override L_dim=256 --override 'forward={hidden_dim: 1024, hidden_layers: 1}' --override 'backward={hidden_dim: 256, hidden_layers: 2, norm: true}' --override 'left_encoder={hidden_dim: 256, hidden_layers: 0, norm: true, identity: false}' --override f_target_tau=0.001 --override b_target_tau=0.001 --override ortho_coef=1"
ACT="--override actor.motivo_archi=true --override 'actor.hidden_dims=[1024]' --override actor.alpha=0.0 --override actor.q_noise_std=0.2 --override actor.q_noise_clip=0.3 --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3 --override actor.q_normalize=false"
COMMON="--auto_resume --train_steps 1000000 --eval_interval 250000 --save_interval 250000 --eval_episodes 10 --num_video 0"
DFB="--override 'measure_hit_spec=\"0:2:0.03\"' --override 'value_hidden_dims=[1024]' --override tau=0.001 --override 'measure_density_b_hidden_dims=[256,256]' --override measure_density_archi=motivo --override measure_density_b_norm=true --override lr=1e-4 --infer_cond_episodes 50000 --eval_on_inferred_cond"
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s bash -c "$ENV $PY -m impls.main --critic fb --actor ddpgbc $COMMON --domain dmc-cheetah --save_dir exp/exorl_fbref --override discount=0.98 $REF $ACT --seed $s > logs/exorl_disc/fbref_cheetah_s$s.log 2>&1" &
  CUDA_VISIBLE_DEVICES=$((s+3)) bash -c "$ENV $PY -m impls.main --critic fb --actor ddpgbc $COMMON --domain dmc-point-mass-maze --save_dir exp/exorl_fbref --override discount=0.99 --override lr_actor=1e-6 $REF $ACT --seed $s > logs/exorl_disc/fbref_maze_s$s.log 2>&1" &
done
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=$((s+6)) bash -c "$ENV $PY -m impls.main --critic density_fb_z --actor ddpgbc $COMMON --domain dmc-point-mass-maze --save_dir exp/exorl_dfb_mazelr --override discount=0.99 --override lr_actor=1e-6 $DFB $ACT --seed $s > logs/exorl_disc/dfbz_maze_lr1e-6_s$s.log 2>&1" &
done
wait
echo EXORL_DISC_DONE
