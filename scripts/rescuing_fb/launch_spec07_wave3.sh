#!/usr/bin/env bash
# Spec 07 WAVE 3 (2026-07-23): relaunch the 4 wave-2 jobs that segfaulted during the
# first CPU eval under 12-job contention (20 s/it). Same cells, idle box, MAX=6.
# zctl = the single-difference z-actor attribution control for the TPActor failure.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec07}
LOG=/dev/shm/spec07_logs; mkdir -p "$LOG" "$OUTROOT"
MAX=${MAX:-6}
GPUS=(0 2 3 4 7)

active() { pgrep -fc 'main.py --env_name' 2>/dev/null || echo 0; }
pick_gpu() {
  local elig; elig=$(printf '%s\n' "${GPUS[@]}" | paste -sd'|' -)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v e="^(${elig})$" '$1 ~ e {print $2, $1}' | sort -n | head -1 | awk '{print $2}'
}

JOBS=(
  "zctl_s0|cube_single_state_measure_firstpassage_noortho_d50|0|+agent_overrides.actor_cond=z +agent_overrides.actor_z_from_critic=true run_group=GCIQL-State-Flow-Measure-FP-NoOrtho-ZActor-d50"
  "zctl_s1|cube_single_state_measure_firstpassage_noortho_d50|1|+agent_overrides.actor_cond=z +agent_overrides.actor_z_from_critic=true run_group=GCIQL-State-Flow-Measure-FP-NoOrtho-ZActor-d50"
  "w03no_s0|cube_single_state_measure_cellw0_noortho_d50|0|agent_overrides.measure_offdiag_weight=0.3 +agent_overrides.measure_scale_norm=true run_group=GCIQL-State-Flow-Measure-CellW030-NoOrtho-d50"
  "w01no_s0|cube_single_state_measure_cellw0_noortho_d50|0|agent_overrides.measure_offdiag_weight=0.1 +agent_overrides.measure_scale_norm=true run_group=GCIQL-State-Flow-Measure-CellW010-NoOrtho-d50"
)

n=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed extra <<<"$j"
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[w3] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 60
  done
  g=$(pick_gpu); [ -z "$g" ] && g=${GPUS[0]}
  n=$((n+1))
  echo "[w3] $(date +%H:%M:%S) launch #$n/${#JOBS[@]} $tag on GPU$g ${extra:+[$extra]}"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      output_root="$OUTROOT" exp_name="$tag" $extra \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[w3] all wave-3 jobs finished"
