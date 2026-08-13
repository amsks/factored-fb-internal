#!/usr/bin/env bash
# Spec 08 E1 (2026-07-24): PHASE-2 ball-visitation on the SEPARATED (z,x) form.
# Warm-start from the FP-ZX phase-1 checkpoints (0.904/0.960) and train the non-absorbing
# visitation TD on the frozen backward map (measure_stopgrad_psi), z and x independent.
# The config cube_single_state_measure_ballvisit_zx_d50.yaml bakes in the ZX form + the
# three phase-2 switches; this launcher only supplies the per-seed warm-start ckpt.
# Payoff eval (separate): direct-z on region/linear/mixture (spec 08 E1).
# Spec: PAPER/rescuing-fb/specs/08-two-phase-next-experiments.md
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec08}
LOG=/dev/shm/spec08_logs; mkdir -p "$LOG" "$OUTROOT"
MAX=${MAX:-6}
GPUS=(0 2 3 4 7)                         # 1/5/6 = other track, never
ZXRUNS=/home/adityamohan/exp_spec07/factored-fb/GCIQL-State-Flow-Measure-FP-ZX-d50
CFG=cube_single_state_measure_ballvisit_zx_d50

active() { pgrep -fc 'main.py --env_name' 2>/dev/null || echo 0; }
pick_gpu() {
  local elig; elig=$(printf '%s\n' "${GPUS[@]}" | paste -sd'|' -)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v e="^(${elig})$" '$1 ~ e {print $2, $1}' | sort -n | head -1 | awk '{print $2}'
}

WS="+agent_overrides.warm_start_epoch=500000"
JOBS=(
  "visz_s0|0|+agent_overrides.warm_start_ckpt=$ZXRUNS/zx_s0 $WS"
  "visz_s1|1|+agent_overrides.warm_start_ckpt=$ZXRUNS/zx_s1 $WS"
)

n=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag seed extra <<<"$j"
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[e1] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 60
  done
  g=$(pick_gpu); [ -z "$g" ] && g=${GPUS[0]}
  n=$((n+1))
  echo "[e1] $(date +%H:%M:%S) launch #$n/${#JOBS[@]} $tag on GPU$g [$extra]"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$CFG" seed="$seed" \
      output_root="$OUTROOT" exp_name="$tag" $extra \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[e1] all E1 visitation-zx jobs finished"
