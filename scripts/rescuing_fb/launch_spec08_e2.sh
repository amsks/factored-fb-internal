#!/usr/bin/env bash
# Spec 08 E2 (2026-07-24): frozen-B / FROZEN-ACTOR ball-visitation. Warm-start the full net
# from the FP-NoOrtho 0.967 phase-1 ckpt (per seed), freeze psi AND freeze the actor; train
# only phi by the non-absorbing visitation TD. Fixed-policy visitation evaluation -> isolates
# the policy-measure feedback loop vs the frozen/updated BallVisit-FrozenPsi cell.
# Spec: PAPER/rescuing-fb/specs/08-two-phase-next-experiments.md (E2)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec08}
LOG=/dev/shm/spec08_logs; mkdir -p "$LOG" "$OUTROOT"
MAX=${MAX:-6}
GPUS=(0 2 3 4 7)                         # 1/5/6 = other track, never
FPRUNS=/home/adityamohan/exp_spec06/factored-fb/GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50
CFG=cube_single_state_measure_ballvisit_frozenactor_d50

active() { pgrep -fc 'main.py --env_name' 2>/dev/null || echo 0; }
pick_gpu() {
  local elig; elig=$(printf '%s\n' "${GPUS[@]}" | paste -sd'|' -)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v e="^(${elig})$" '$1 ~ e {print $2, $1}' | sort -n | head -1 | awk '{print $2}'
}

WS="+agent_overrides.warm_start_epoch=500000"
JOBS=(
  "faz_s0|0|+agent_overrides.warm_start_ckpt=$FPRUNS/m_firstpassage_noortho_s0 $WS"
  "faz_s1|1|+agent_overrides.warm_start_ckpt=$FPRUNS/m_firstpassage_noortho_s1 $WS"
)

n=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag seed extra <<<"$j"
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[e2] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 60
  done
  g=$(pick_gpu); [ -z "$g" ] && g=${GPUS[0]}
  n=$((n+1))
  echo "[e2] $(date +%H:%M:%S) launch #$n/${#JOBS[@]} $tag on GPU$g [$extra]"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$CFG" seed="$seed" \
      output_root="$OUTROOT" exp_name="$tag" $extra \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[e2] all E2 frozen-actor jobs finished"
