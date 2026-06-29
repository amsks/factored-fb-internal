#!/usr/bin/env bash
# Sanity checks (2026-06-25), all 500k, 3 seeds each, tmux session `sanity500`.
#   G1  stock GCIQL + DDPGBC          on antmaze-medium-navigate  (must reproduce ~71)
#   G2  GCIQL MLP-value + Flow actor  on antmaze-medium-navigate  (isolate bilinear vs MLP, flow held fixed)
#   G3  GCIQL Bilinear-d50 + Flow     on scene-play               (bridge cell on scene)
#   G4  FB FlowBC + Flow              on scene-play               (FB comparator on scene)
# GCIQL jobs packed 2/GPU with XLA_PYTHON_CLIENT_PREALLOCATE=false; FB (torch) 1/GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
SESS=sanity500
LOGS=/dev/shm/sanity_logs
HYDRA=/dev/shm/sanity_hydra
mkdir -p "$LOGS" "$HYDRA"
tmux has-session -t "$SESS" 2>/dev/null && { echo "session $SESS exists; abort"; exit 1; }
tmux new-session -d -s "$SESS" -n ctl

gciql() {  # tag gpu config env run_group seed extra...
  local tag=$1 gpu=$2 cfg=$3 env=$4 grp=$5 seed=$6; shift 6
  local cmd="XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpu \
.venv/bin/python run_gciql.py --config-name $cfg \
env_name=$env seed=$seed train_steps=500000 save_interval=500000 eval_interval=100000 \
run_group=$grp hydra.run.dir=$HYDRA/$tag $* \
2>&1 | tee $LOGS/$tag.log"
  tmux new-window -t "$SESS" -n "$tag" "$cmd; echo EXIT=\$? >> $LOGS/$tag.log; exec bash"
}

fb() {  # tag gpu seed
  local tag=$1 gpu=$2 seed=$3
  local cmd="CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python train.py domain=scene seed=$seed \
num_train_steps=500000 eval_every=100000 use_wandb=true wandb_group=fb_scene_500k \
wandb_entity=amsks data_path=/dev/shm/datasets device=cuda \
2>&1 | tee $LOGS/$tag.log"
  tmux new-window -t "$SESS" -n "$tag" "$cmd; echo EXIT=\$? >> $LOGS/$tag.log; exec bash"
}

# --- G1: stock GCIQL + DDPGBC on antmaze (alpha=0.3 per OGBench hyperparameters.sh) ---
gciql g1_s0 0 cube_single_state antmaze-medium-navigate-v0 GCIQL-AntMaze-Medium-Stock-DDPGBC 0 agent_overrides.alpha=0.3
gciql g1_s1 1 cube_single_state antmaze-medium-navigate-v0 GCIQL-AntMaze-Medium-Stock-DDPGBC 1 agent_overrides.alpha=0.3
gciql g1_s2 2 cube_single_state antmaze-medium-navigate-v0 GCIQL-AntMaze-Medium-Stock-DDPGBC 2 agent_overrides.alpha=0.3

# --- G2: GCIQL MLP-value + Flow actor on antmaze (bc_coeff=3.0 from base config, value_arch=mlp default) ---
gciql g2_s0 0 cube_single_state_gciql_flow antmaze-medium-navigate-v0 GCIQL-AntMaze-Medium-Flow-MLP 0
gciql g2_s1 1 cube_single_state_gciql_flow antmaze-medium-navigate-v0 GCIQL-AntMaze-Medium-Flow-MLP 1
gciql g2_s2 2 cube_single_state_gciql_flow antmaze-medium-navigate-v0 GCIQL-AntMaze-Medium-Flow-MLP 2

# --- G3: GCIQL Bilinear-d50 + Flow on scene ---
gciql g3_s0 3 cube_single_state_gciql_flow_bilinear_d50 scene-play-v0 GCIQL-Scene-Flow-Bilinear-d50 0
gciql g3_s1 3 cube_single_state_gciql_flow_bilinear_d50 scene-play-v0 GCIQL-Scene-Flow-Bilinear-d50 1
gciql g3_s2 4 cube_single_state_gciql_flow_bilinear_d50 scene-play-v0 GCIQL-Scene-Flow-Bilinear-d50 2

# --- G4: FB FlowBC on scene ---
fb g4_s0 5 0
fb g4_s1 6 1
fb g4_s2 7 2

echo "launched 12 jobs into tmux session $SESS; logs in $LOGS"
