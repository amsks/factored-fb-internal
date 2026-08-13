#!/usr/bin/env bash
# Warm-start experiment (2026-07-21): does the good scalar-Q basin SURVIVE the measure
# objective when you start INSIDE it? Cold-start sat at the floor for the whole offdiag
# sweep (w=0 included). Here each measure cell is warm-started from the SEED-MATCHED trained
# bilinear-d50 flow checkpoint (~0.90), critic+actor spliced trainable, then trains the
# measure. Same box-wide backfill gate as the main queue (MAX procs), so they co-schedule.
#
# Cells (per takeaway: first-passage is the only measure axis with a cold pulse, 0.23):
#   WarmFirstPassage : first-passage measure-only, warm      -> does warm lift 0.23 -> 0.90?
#   WarmCellW0       : M=1+(1-g)Q degenerate (offdiag 0), warm-> does the Q basin survive w=0?
#   WarmCellW025     : offdiag 0.25, warm                    -> warm sweep point
#
#   tmux new -d -s warmq 'DRY_RUN=0 scripts/rescuing_fb/warmstart_queue.sh 2>&1 | tee /dev/shm/warmq_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$(pwd)/.venv/bin/python}
DRY_RUN=${DRY_RUN:-1}
MAX=${MAX:-12}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
STEPS=${STEPS:-500000}
STAGGER=${STAGGER:-45}
OUTROOT=${OUTROOT:-/home/adityamohan/exp_queue}
GPUS=(${GPUS:-0 2 3 4 7})
LOGDIR=/dev/shm/queue_logs
mkdir -p "$LOGDIR" "$OUTROOT"
CKROOT=/home/adityamohan/bridge_ckpt_backup/gciql_outputs/factored-fb/GCIQL-State-Flow-Bilinear-d50
declare -A SRC=( [0]="$(ls -d $CKROOT/sd000_* | head -1)"
                 [1]="$(ls -d $CKROOT/sd001_* | head -1)"
                 [2]="$(ls -d $CKROOT/sd002_* | head -1)" )

# tag|config|seed|extra
JOBS=()
for s in 0 1 2; do
  warm="agent_overrides.warm_start_ckpt=${SRC[$s]} agent_overrides.warm_start_epoch=500000"
  JOBS+=("warm_fp_s${s}|cube_single_state_measure_firstpassage_d50|${s}|${warm} run_group=GCIQL-State-Flow-Measure-WarmFirstPassage-d50")
  JOBS+=("warm_cellw0_s${s}|cube_single_state_measure_cellw0_d50|${s}|${warm} run_group=GCIQL-State-Flow-Measure-WarmCellW0-d50")
  JOBS+=("warm_cellw025_s${s}|cube_single_state_measure_cellw0_d50|${s}|${warm} agent_overrides.measure_offdiag_weight=0.25 run_group=GCIQL-State-Flow-Measure-WarmCellW025-d50")
done

active() { pgrep -fc 'main.py --env_name' 2>/dev/null || echo 0; }
pick_gpu() {
  local elig; elig=$(printf '%s\n' "${GPUS[@]}" | paste -sd'|' -)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v e="^(${elig})$" '$1 ~ e {print $2, $1}' | sort -n | head -1 | awk '{print $2}'
}

echo "[warmq] ${#JOBS[@]} warm-start jobs, MAX=$MAX, eval_interval=$EVAL_INTERVAL -> $OUTROOT"
for s in 0 1 2; do [ -e "${SRC[$s]}/params_500000.pkl" ] || { echo "MISSING source seed $s: ${SRC[$s]}"; exit 1; }; done
if [ "$DRY_RUN" = "1" ]; then
  for j in "${JOBS[@]}"; do IFS='|' read -r t c se x <<<"$j"; echo "   $t ($c seed $se) [$x]"; done
  echo "[warmq] active=$(active)"; exit 0
fi
n=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed extra <<<"$j"
  while [ "$(active)" -ge "$MAX" ]; do echo "[warmq] $(date +%H:%M:%S) active=$(active) >= $MAX — wait $tag"; sleep 60; done
  g=$(pick_gpu); [ -z "$g" ] && g=${GPUS[0]}; n=$((n+1))
  echo "[warmq] $(date +%H:%M:%S) launch #$n/${#JOBS[@]} $tag on GPU$g"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" train_steps="$STEPS" \
      eval_interval="$EVAL_INTERVAL" output_root="$OUTROOT" exp_name="$tag" $extra \
      >"$LOGDIR/${tag}.log" 2>&1 &
  sleep "$STAGGER"
done
echo "[warmq] all dispatched; waiting..."; wait; echo "[warmq] DONE"
