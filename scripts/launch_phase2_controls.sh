#!/usr/bin/env bash
# Phase-2 (2026-06-25), tmux `phase2`. Two parts:
#  (A) RELAUNCH the stuck antmaze GCIQL sanity (g1/g2) with cheap eval.
#      g1 stock GCIQL+DDPGBC: eval_on_cpu=0 (GPU eval is fast for the gaussian actor and
#         is what OGBench uses for the ~71 number); the cpu-eval rule is flow-actor-only.
#      g2 flow-MLP GCIQL: keep eval_on_cpu=1 (flow rollout) but eval cheap.
#  (B) FB z=g HER confound controls on cube (1M x3), to explain zg_her (~19 vs zHER ~32):
#      her_lb     = HER + LEARNED B + ratio 1.0  (is identity B the problem?)
#      her_idnorm = HER + identity B + NORMALIZED z (does bounding z fix the measure blowup?)
#      iql_lb     = IQL + LEARNED B + ratio 1.0  (single-var control for zg_iql ~44 vs IQLcritic ~24)
set -euo pipefail
cd "$(dirname "$0")/.."
SESS=phase2
LOGS=/dev/shm/phase2_logs
DATA=/dev/shm/datasets
mkdir -p "$LOGS" /dev/shm/phase2_hydra
tmux has-session -t "$SESS" 2>/dev/null && { echo "session $SESS exists; abort"; exit 1; }
tmux new-session -d -s "$SESS" -n ctl

gciql() {  # tag gpu config run_group seed eval_on_cpu eval_eps eval_int extra...
  local tag=$1 gpu=$2 cfg=$3 grp=$4 seed=$5 eoc=$6 eeps=$7 eint=$8; shift 8
  local cmd="XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpu \
.venv/bin/python run_gciql.py --config-name $cfg env_name=antmaze-medium-navigate-v0 \
seed=$seed train_steps=500000 save_interval=500000 eval_on_cpu=$eoc eval_episodes=$eeps \
eval_interval=$eint run_group=$grp hydra.run.dir=/dev/shm/phase2_hydra/$tag $* \
2>&1 | tee $LOGS/$tag.log"
  tmux new-window -t "$SESS" -n "$tag" "$cmd; echo EXIT=\$? >> $LOGS/$tag.log; exec bash"
}

fb() {  # tag gpu grp seed extra...
  local tag=$1 gpu=$2 grp=$3 seed=$4; shift 4
  local base="agent=fb_flowbc domain=cube_single ortho_coef=1000 lr_b=1e-4 device=cuda \
data_path=$DATA use_wandb=true wandb_entity=amsks save_eval_videos=false \
train_goal_mode=traj train_goal_ratio=1.0 num_train_steps=1000000"
  local cmd="CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python train.py $base $* \
seed=$seed wandb_group=$grp wandb_run_name=${grp}_s${seed} hydra.run.dir=/dev/shm/phase2_out/$tag \
2>&1 | tee $LOGS/$tag.log"
  tmux new-window -t "$SESS" -n "$tag" "$cmd; echo EXIT=\$? >> $LOGS/$tag.log; exec bash"
}

# --- (A) GCIQL antmaze relaunch ---
# g1 stock GCIQL + DDPGBC : GPU eval (fast), 50 eps, eval @100k
gciql g1_s0 0 cube_single_state GCIQL-AntMaze-Medium-Stock-DDPGBC 0 0 50 100000 agent_overrides.alpha=0.3
gciql g1_s1 1 cube_single_state GCIQL-AntMaze-Medium-Stock-DDPGBC 1 0 50 100000 agent_overrides.alpha=0.3
gciql g1_s2 2 cube_single_state GCIQL-AntMaze-Medium-Stock-DDPGBC 2 0 50 100000 agent_overrides.alpha=0.3
# g2 flow-MLP GCIQL : cpu eval (flow), 10 eps, eval only @500k
gciql g2_s0 0 cube_single_state_gciql_flow GCIQL-AntMaze-Medium-Flow-MLP 0 1 10 500000
gciql g2_s1 1 cube_single_state_gciql_flow GCIQL-AntMaze-Medium-Flow-MLP 1 1 10 500000
gciql g2_s2 2 cube_single_state_gciql_flow GCIQL-AntMaze-Medium-Flow-MLP 2 1 10 500000

# --- (B) FB z=g HER controls (cube 1M) ---
fb herlb_s0 3 fb_cube_her_learnedB_r1 0 fixed_b=none         critic_mode=measure
fb herlb_s1 4 fb_cube_her_learnedB_r1 1 fixed_b=none         critic_mode=measure
fb herlb_s2 5 fb_cube_her_learnedB_r1 2 fixed_b=none         critic_mode=measure
fb hernrm_s0 3 fb_cube_her_idnorm_r1 0 fixed_b=identity_norm critic_mode=measure
fb hernrm_s1 4 fb_cube_her_idnorm_r1 1 fixed_b=identity_norm critic_mode=measure
fb hernrm_s2 5 fb_cube_her_idnorm_r1 2 fixed_b=identity_norm critic_mode=measure
fb iqllb_s0 6 fb_cube_iql_learnedB_r1 0 fixed_b=none         critic_mode=iql
fb iqllb_s1 7 fb_cube_iql_learnedB_r1 1 fixed_b=none         critic_mode=iql
fb iqllb_s2 6 fb_cube_iql_learnedB_r1 2 fixed_b=none         critic_mode=iql

echo "launched 6 GCIQL + 9 FB = 15 jobs into tmux $SESS; logs $LOGS"
