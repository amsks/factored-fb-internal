#!/usr/bin/env bash
# Spec 08 E6 (2026-07-24): policy-latent (z) coverage sweep on the ZX phase-1 form, cold start.
# Arms (measure_z_random_frac): 0.0 goal-only | 0.5 = existing zx_s0/s1 (reused) | anneal (E6 knob).
# ARM=zfrac0 launches the goal-only arm (config-only, runnable now); ARM=anneal launches the
# annealed arm (needs measure_z_random_frac_anneal knob). Discriminating eval: direct-z region+linear.
# Spec: PAPER/rescuing-fb/specs/08-two-phase-next-experiments.md (E6)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec08}
LOG=/dev/shm/spec08_logs; mkdir -p "$LOG" "$OUTROOT"
MAX=${MAX:-6}
GPUS=(0 2 3 4 7)                         # 1/5/6 = other track, never
CFG=cube_single_state_measure_fp_zx_d50
ARM=${ARM:-zfrac0}

active() { pgrep -fc 'main.py --env_name' 2>/dev/null || echo 0; }
pick_gpu() {
  local elig; elig=$(printf '%s\n' "${GPUS[@]}" | paste -sd'|' -)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v e="^(${elig})$" '$1 ~ e {print $2, $1}' | sort -n | head -1 | awk '{print $2}'
}

case "$ARM" in
  zfrac0)
    OV="agent_overrides.measure_z_random_frac=0.0 run_group=GCIQL-State-Flow-Measure-FP-ZX-zfrac0-d50"
    JOBS=("zf0_s0|0|$OV" "zf0_s1|1|$OV") ;;
  anneal)
    OV="+agent_overrides.measure_z_random_frac_anneal=true run_group=GCIQL-State-Flow-Measure-FP-ZX-zfracAnneal-d50"
    JOBS=("zfan_s0|0|$OV" "zfan_s1|1|$OV") ;;
  *) echo "unknown ARM=$ARM (zfrac0|anneal)"; exit 1 ;;
esac

n=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag seed extra <<<"$j"
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[e6:$ARM] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 60
  done
  g=$(pick_gpu); [ -z "$g" ] && g=${GPUS[0]}
  n=$((n+1))
  echo "[e6:$ARM] $(date +%H:%M:%S) launch #$n/${#JOBS[@]} $tag on GPU$g [$extra]"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$CFG" seed="$seed" \
      output_root="$OUTROOT" exp_name="$tag" $extra \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[e6:$ARM] all jobs finished"
