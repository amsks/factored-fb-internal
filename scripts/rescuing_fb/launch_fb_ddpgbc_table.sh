#!/usr/bin/env bash
# Vanilla FB with the DDPG+BC actor on cube-double, scene and antmaze-medium -- the
# missing FB cells of the env x actor x method table.
#
# WHY: the table's FB column currently has flowbc on all four envs but ddpgbc only on
# cube-single (.353/3s). Without these, a DDPG+BC-only table has no FB baseline at all on
# the three envs that matter most, and the cube-double flowbc FB numbers we do hold are
# single-seed, which cannot carry a 95% CI either.
#
# `agent=fb` is the DDPG+BC actor; `agent=fb_flowbc` is the flow variant (the actor lives
# in the agent yaml, not a CLI knob). ortho_coef comes from each domain yaml (1.0), i.e.
# the "tuned FB" convention used by launch_missing_baselines.sh -- NOT the ortho1000
# reference reimplementation. Recorded here because FB run-group names have lied before.
#
# Datasets: /dev/shm/factored-fb/datasets holds cube/scene but NOT antmaze, and /dev/shm
# gets wiped -- antmaze therefore reads from ~/.ogbench/data. Both paths are passed
# explicitly per env rather than assumed.
#
# Cooperative concurrency; DRY_RUN short-circuits before the wait.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=$(pwd)/.venv/bin/python
OUT=${OUT:-/home/adityamohan/exp_fbddpg}; LOG=/dev/shm/fbddpg_logs
mkdir -p "$LOG" "$OUT"; DRY_RUN=${DRY_RUN:-0}; MAX=${MAX:-8}; STEPS=${STEPS:-500000}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
active() { local c; c=$(pgrep -fc 'train.py|main.py --env_name' 2>/dev/null); echo "${c:-0}"; }

# tag|domain|data_path
CELLS=(
  "fbd_cd|cube_double|/dev/shm/factored-fb/datasets"
  "fbd_sc|scene|/dev/shm/factored-fb/datasets"
  "fbd_am|antmaze_medium|/home/adityamohan/.ogbench/data"
)
n=0; launched=0
for c in "${CELLS[@]}"; do
  IFS='|' read -r base dom data <<<"$c"
  [ -d "$data" ] || { echo "[fbd] FATAL missing dataset dir $data"; exit 1; }
  for s in 0 1 2; do
    tag="${base}_s${s}"
    [ -d "$OUT/$tag" ] && { echo "[fbd] skip $tag (exists)"; continue; }
    if [ "$DRY_RUN" = "1" ]; then echo "      DRY: $tag domain=$dom data=$data"; continue; fi
    while [ "$(active)" -ge "$MAX" ]; do echo "[fbd] $(date +%H:%M:%S) active=$(active) waiting for $tag"; sleep 180; done
    g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1)); launched=$((launched+1))
    echo "[fbd] $(date +%H:%M:%S) launch $tag on GPU$g"
    CUDA_VISIBLE_DEVICES=$g "$PY" train.py agent=fb domain="$dom" \
      data_path="$data" num_train_steps=$STEPS save_every=100000 eval_every=100000 \
      seed="$s" use_wandb=true wandb_entity=amsks wandb_group="fbd_$base" \
      device=cuda hydra.run.dir=$OUT/$tag > "$LOG/$tag.log" 2>&1 &
    sleep 45
  done
done
wait; echo "[fbd] $(date +%H:%M:%S) launched=$launched"
