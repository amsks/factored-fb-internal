#!/usr/bin/env bash
# z=g experiment (2026-06-25): FB with the goal used DIRECTLY as the task vector
# (fixed_b=identity => B=I, z=g, norm_z off), all goals on-trajectory HER
# (train_goal_mode=traj, train_goal_ratio=1.0). Two critic variants, on cube-single,
# 1M x 3 seeds, tmux `zg_cube`. Base hyperparams match the bridge-table rows
# fb_state_zher / fb_state_iqlcritic_fix exactly (ortho_coef/lr_b inert under identity).
#   zg_her  = zHER + z=g          (critic_mode=measure)   <- compare vs zHER row (~32 @1M)
#   zg_iql  = IQL-critic + z=g    (critic_mode=iql)        <- compare vs FB-IQL-critic row (~23.6 @1M)
# cube-only: FB-IQL success uses cube physics; that's also where the bridge anchors live.
set -euo pipefail
cd "$(dirname "$0")/.."
SESS=zg_cube
LOGS=/dev/shm/zg_cube_logs
DATA=/dev/shm/datasets
mkdir -p "$LOGS"
tmux has-session -t "$SESS" 2>/dev/null && { echo "session $SESS exists; abort"; exit 1; }
tmux new-session -d -s "$SESS" -n ctl

base="agent=fb_flowbc domain=cube_single ortho_coef=1000 lr_b=1e-4 device=cuda \
data_path=$DATA use_wandb=true wandb_entity=amsks save_eval_videos=false \
fixed_b=identity train_goal_mode=traj train_goal_ratio=1.0 num_train_steps=1000000"

run() {  # tag gpu grp seed extra...
  local tag=$1 gpu=$2 grp=$3 seed=$4; shift 4
  local dir=/dev/shm/zg_cube_out/$tag
  local cmd="CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python train.py $base $* \
seed=$seed wandb_group=$grp wandb_run_name=${grp}_s${seed} hydra.run.dir=$dir \
2>&1 | tee $LOGS/$tag.log"
  tmux new-window -t "$SESS" -n "$tag" "$cmd; echo EXIT=\$? >> $LOGS/$tag.log; exec bash"
}

# zg_iql = FB-IQL-critic + z=g  (GPUs 0,1,2)
run zgiql_s0 0 fb_cube_zg_iql 0 critic_mode=iql
run zgiql_s1 1 fb_cube_zg_iql 1 critic_mode=iql
run zgiql_s2 2 fb_cube_zg_iql 2 critic_mode=iql

# zg_her = zHER + z=g  (GPUs 5,6,7)  critic_mode=measure is the default
run zgher_s0 5 fb_cube_zg_her 0 critic_mode=measure
run zgher_s1 6 fb_cube_zg_her 1 critic_mode=measure
run zgher_s2 7 fb_cube_zg_her 2 critic_mode=measure

echo "launched 6 jobs into tmux $SESS; logs in $LOGS"
