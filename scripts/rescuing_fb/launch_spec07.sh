#!/usr/bin/env bash
# Spec 07 launch (PAPER/rescuing-fb/specs/07-fp-capability-and-renewal-measure.md):
# item 1 TPActor 3 seeds + item 4 CriticMix linearity-conflict probe 1 seed.
# GPUs 1/5/6 = other track (never). GPU0 left to the last queue straggler.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec07}
LOG=/dev/shm/spec07_logs; mkdir -p "$LOG" "$OUTROOT"

JOBS=(
  "tpactor_s0|cube_single_state_measure_fp_noortho_tpactor_d50|0|2"
  "tpactor_s1|cube_single_state_measure_fp_noortho_tpactor_d50|1|3"
  "tpactor_s2|cube_single_state_measure_fp_noortho_tpactor_d50|2|4"
  "criticmix_s0|cube_single_state_measure_fp_noortho_tpactor_criticmix_d50|0|7"
)
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed gpu <<<"$j"
  echo "[spec07] $(date +%H:%M:%S) launching $tag ($cfg seed $seed) on GPU$gpu"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpu \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      output_root="$OUTROOT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[spec07] all training jobs finished"
