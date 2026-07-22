#!/usr/bin/env bash
# data_collection/run_collection_box2d.sh — collect 1000 expert demos for the 8
# validated Box2D tasks (TD-MPC2 trained from scratch; checkpoints in logs_rlmini).
# Runs concurrently across GPUs (5 faulty -> excluded). success_only=true => clean
# demos only. Detached so it survives terminal/session boundaries.
#
#   setsid nohup bash data_collection/run_collection_box2d.sh >/raid/tongle/rl/fb-data/collect_logs/box2d_driver.log 2>&1 & disown
set -uo pipefail
trap '' HUP

FB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEWT_PY="/raid/tongle/rl/newt/.venv/bin/python"
LOGS="/raid/tongle/rl/newt/tdmpc2/logs_rlmini"

NUM_EPISODES="${NUM_EPISODES:-1000}"
NUM_ENVS="${NUM_ENVS:-50}"
OUT="${OUT:-/raid/tongle/rl/fb-data/collected}"
SUCCESS_ONLY="${SUCCESS_ONLY:-true}"
GPUS=(0 1 2 3 4 6 7)                    # GPU 5 faulty -> excluded
LOGDIR="/raid/tongle/rl/fb-data/collect_logs"; mkdir -p "$LOGDIR"

# 8 validated Box2D tasks (4 loco + 4 manip), all with good policies.
TASKS=(center-ctrl goal-ctrl goal-ctrl-maze center-wall-navigation \
       goal-ctrl-target center-ctrl-target-hard center-ctrl-target center-wall)

resolve_ckpt() {                        # $1=task -> latest checkpoint (final name preferred)
    local t="$1" ck
    ck=$(ls -t "$LOGS/b2d-$t"/*/models/"b2d-$t.pt" 2>/dev/null | head -1)
    [ -z "$ck" ] && ck=$(ls -t "$LOGS/b2d-$t"/*/models/*.pt 2>/dev/null | head -1)
    echo "$ck"
}

i=0
for t in "${TASKS[@]}"; do
    gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
    ck="$(resolve_ckpt "$t")"
    if [ -z "$ck" ]; then echo "[box2d-collect] SKIP b2d-$t (no checkpoint)"; i=$((i+1)); continue; fi
    echo "[box2d-collect] b2d-$t -> GPU $gpu  ckpt=$ck  (log $LOGDIR/b2d-$t.log)"
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl nohup "$NEWT_PY" \
        "$FB_ROOT/data_collection/collect_expert.py" \
        task="b2d-$t" num_envs="$NUM_ENVS" +num_episodes="$NUM_EPISODES" \
        +success_only="$SUCCESS_ONLY" +ckpt="$ck" +out="$OUT" \
        > "$LOGDIR/b2d-$t.log" 2>&1 &
    i=$(( i + 1 ))
done
wait
echo "[box2d-collect] ALL DONE -> $OUT"
