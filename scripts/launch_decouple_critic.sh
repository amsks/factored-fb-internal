#!/usr/bin/env bash
# Decouple-actor-z × {IQL critic, zHER/measure critic} on cube-single, flow actor.
# Fills the gap from phase2: fb_cube_decouple only ran measure+perm/0.5.
# These two complete the matrix with a clean one-variable comparison vs
# fb_state_zher / fb_state_iql (both traj/0.5, coupled). 1M x 3 seeds each.
set -euo pipefail
cd "$(dirname "$0")/.."
SESS=decpl2
LOGS=/dev/shm/decpl2_logs
DATA=/dev/shm/datasets
mkdir -p "$LOGS" /dev/shm/decpl2_out
tmux has-session -t "$SESS" 2>/dev/null && { echo "session $SESS exists; abort"; exit 1; }
tmux new-session -d -s "$SESS" -n ctl

fb() {  # tag gpu grp seed extra...
  local tag=$1 gpu=$2 grp=$3 seed=$4; shift 4
  local base="agent=fb_flowbc domain=cube_single ortho_coef=1000 lr_b=1e-4 device=cuda \
data_path=$DATA use_wandb=true wandb_entity=amsks save_eval_videos=false \
decouple_actor_z=true train_goal_mode=traj train_goal_ratio=0.5 num_train_steps=1000000"
  local cmd="CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python train.py $base $* \
seed=$seed wandb_group=$grp wandb_run_name=${grp}_s${seed} hydra.run.dir=/dev/shm/decpl2_out/$tag \
2>&1 | tee $LOGS/$tag.log"
  tmux new-window -t "$SESS" -n "$tag" "$cmd; echo EXIT=\$? >> $LOGS/$tag.log; exec bash"
}

# decouple + IQL critic  (the real gap)
fb diql_s0 0 fb_cube_decouple_iql 0 critic_mode=iql
fb diql_s1 1 fb_cube_decouple_iql 1 critic_mode=iql
fb diql_s2 2 fb_cube_decouple_iql 2 critic_mode=iql

# decouple + measure/zHER critic, traj/0.5  (clean one-variable vs fb_state_zher)
fb dzher_s0 3 fb_cube_decouple_zher_traj 0 critic_mode=measure
fb dzher_s1 4 fb_cube_decouple_zher_traj 1 critic_mode=measure
fb dzher_s2 5 fb_cube_decouple_zher_traj 2 critic_mode=measure

echo "launched 6 FB jobs into tmux $SESS (GPUs 0-5); logs $LOGS"
