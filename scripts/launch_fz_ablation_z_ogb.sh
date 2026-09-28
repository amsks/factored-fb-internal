#!/usr/bin/env bash
# F^T z actor ablation on the REWARD-INFERENCE Density-FB (density_fb_z), cube-double + scene,
# 3 seeds x 500k. Flags = the tdjtab manifest's density_fb_z lines + measure_density_actor_value=linear_fz.
# Baselines: the existing density_fb_z runs (exp/tdjtab + hub), read at the same 300k-500k evals.
set -euo pipefail
REPO="${REPO:-$HOME/git/Factored-FB-seam}"
PY="${PY:-/dev/shm/.venv-jax/bin/python}"
cd "$REPO"; mkdir -p logs/fz_ablation
ENVV="XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl"
REL="--override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10"
declare -A SPEC=([cube_double]='"19:22:0.4,28:31:0.4"' [scene_play]='"19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6"')
tmux new-session -d -s fz_ablation_z -n idle 2>/dev/null || true
g=0
for dom in cube_double scene_play; do
  for s in 0 1 2; do
    cmd="CUDA_VISIBLE_DEVICES=$g $ENVV $PY -m impls.main --actor flowbc --auto_resume --train_steps 500000 \
--eval_interval 100000 --save_interval 250000 --critic density_fb_z --domain $dom --seed $s \
--save_dir exp/fz_ablation/linear_fz --override measure_density_z_actor_reward=latent_centered \
--override measure_density_hard=true --override 'measure_hit_spec=${SPEC[$dom]}' $REL \
--override measure_density_actor_value=linear_fz > logs/fz_ablation/z_${dom}_s$s.log 2>&1"
    tmux new-window -t fz_ablation_z -n "z_${dom}_s$s" "cd $REPO && $cmd"; echo "gpu $g: $dom seed $s"
    g=$((g + 1))
  done
done
