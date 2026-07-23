#!/usr/bin/env bash
# data_collection/run_collection.sh — collect 1000 episodes for the 5 ManiSkill +
# 5 Metaworld tasks (newt TD-MPC2 experts), across GPUs. GPU 5 is faulty -> excluded.
#
#   bash data_collection/run_collection.sh
#
# Knobs (env vars): NUM_EPISODES (1000), NUM_ENVS (50), OUT, SUCCESS_ONLY (false).
set -uo pipefail
trap '' HUP

FB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEWT_PY="/raid/tongle/rl/newt/.venv/bin/python"
SNAP="/raid/tongle/rl/newt/checkpoints/models--nicklashansen--newt/snapshots/7eef11eb63c8ed53d61d739693d7140135ea0876"

NUM_EPISODES="${NUM_EPISODES:-1000}"
NUM_ENVS="${NUM_ENVS:-50}"
OUT="${OUT:-/raid/tongle/rl/fb-data/collected}"
SUCCESS_ONLY="${SUCCESS_ONLY:-false}"
GPUS=(0 1 2 3 4)                       # 5 good GPUs (GPU 5 faulty; 6,7 reserved)
LOGDIR="/raid/tongle/rl/fb-data/collect_logs"; mkdir -p "$LOGDIR"

MANISKILL=(ms-pick-cube ms-push-cube ms-stack-cube ms-poke-cube ms-lift-peg)
METAWORLD=(mw-assembly mw-basketball mw-bin-picking mw-box-close mw-button-press)

launch_group() {              # $@ = task names; one per GPU, run concurrently, then wait
    local -a tasks=("$@") i=0 t gpu
    for t in "${tasks[@]}"; do
        gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
        echo "[run_collection] $t -> GPU $gpu  (log $LOGDIR/$t.log)"
        CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl nohup "$NEWT_PY" \
            "$FB_ROOT/data_collection/collect_expert.py" \
            task="$t" num_envs="$NUM_ENVS" +num_episodes="$NUM_EPISODES" \
            +success_only="$SUCCESS_ONLY" +checkpoint_dir="$SNAP" +out="$OUT" \
            > "$LOGDIR/$t.log" 2>&1 &
        i=$(( i + 1 ))
    done
    wait
}

echo "[run_collection] ManiSkill group (5 tasks)"
launch_group "${MANISKILL[@]}"
echo "[run_collection] Metaworld group (5 tasks)"
launch_group "${METAWORLD[@]}"
echo "[run_collection] ALL DONE -> $OUT"
