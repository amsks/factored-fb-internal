#!/usr/bin/env bash
# F^T z actor ablation on goal-conditioned Density-FB (density_fb), OGBench cube-double + scene,
# 3 seeds x 500k (arm linear_fz); plus the goal-reward baseline on scene seeds 0-2 at 1M, which the
# table's scene cell is missing (only seeds 3,4 exist). Flags copied from the table runs' config.json.
set -euo pipefail
REPO="${REPO:-$HOME/git/Factored-FB-seam}"
PY="${PY:-/dev/shm/.venv-jax/bin/python}"
cd "$REPO"; mkdir -p logs/fz_ablation_goal
ENVV="XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl"
REL="--override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10"
declare -A SPEC=([cube_double]='"19:22:0.4,28:31:0.4"' [scene_play]='"19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6"')
tmux new-session -d -s fz_ablation_ogb -n idle 2>/dev/null || true
launch() {  # gpu domain seed steps save_dir extra name
  local cmd="CUDA_VISIBLE_DEVICES=$1 $ENVV $PY -m impls.main --actor flowbc --auto_resume --train_steps $4 \
--eval_interval 100000 --save_interval 250000 --critic density_fb --domain $2 --seed $3 --save_dir $5 \
--override measure_density_hard=true --override 'measure_hit_spec=${SPEC[$2]}' $REL $6 > logs/fz_ablation_goal/$7.log 2>&1"
  tmux new-window -t fz_ablation_ogb -n "$7" "cd $REPO && $cmd"; echo "gpu $1: $7"
}
g=0
for dom in cube_double scene_play; do
  for s in 0 1 2; do
    launch $g $dom $s 500000 exp/fz_ablation_goal/linear_fz "--override measure_density_actor_value=linear_fz" "fz_${dom}_s$s"
    g=$((g + 1))
  done
done
launch 6 scene_play 0 1000000 exp/tdjtab "" "base_scene_play_s0"
launch 7 scene_play 1 1000000 exp/tdjtab "" "base_scene_play_s1"
launch 7 scene_play 2 1000000 exp/tdjtab "" "base_scene_play_s2"
