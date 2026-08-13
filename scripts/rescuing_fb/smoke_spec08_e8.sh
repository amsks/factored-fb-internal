#!/usr/bin/env bash
# Spec 08 E8 -- smoke test every NEW env config before committing GPU-days to it.
#
# Six of the nine environments have never been run in this repo (cube-triple,
# cube-quadruple, puzzle-3x3/4x4/4x5/4x6), and each brings a new observation width and a
# new hit-spec conjunction width (up to 24 ANDed blocks on puzzle-4x6). This runs a few
# hundred gradient steps per config with eval disabled, which exercises: dataset load,
# env construction, agent build, the assert-lattice, _parse_hit_spec, and the measure loss
# at the real observation shape.
#
# Usage:  scripts/rescuing_fb/smoke_spec08_e8.sh 2>&1 | tee /dev/shm/spec08_logs/e8_smoke.log
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
LOG=/dev/shm/spec08_logs/smoke; mkdir -p "$LOG"
OUT=/dev/shm/e8_smoke; mkdir -p "$OUT"
STEPS=${STEPS:-300}
GPUS=(2 3 4 5 6 7)          # 0 and 1 are held by the running E7 cube-double jobs
PAR=${PAR:-6}

CFGS=(
  cube_triple_state_measure_firstpassage_noortho_d50
  cube_quadruple_state_measure_firstpassage_noortho_d50
  puzzle_3x3_state_measure_firstpassage_noortho_d50
  puzzle_4x4_state_measure_firstpassage_noortho_d50
  puzzle_4x5_state_measure_firstpassage_noortho_d50
  puzzle_4x6_state_measure_firstpassage_noortho_d50
  cube_triple_state_gciql_flow_bilinear_d50
  cube_quadruple_state_gciql_flow_bilinear_d50
  scene_state_gciql_flow_bilinear_d50
  puzzle_3x3_state_gciql_flow_bilinear_d50
  puzzle_4x4_state_gciql_flow_bilinear_d50
  puzzle_4x5_state_gciql_flow_bilinear_d50
  puzzle_4x6_state_gciql_flow_bilinear_d50
)

n=0
for cfg in "${CFGS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 5; done
  g=${GPUS[$(( n % ${#GPUS[@]} ))]}
  n=$((n+1))
  echo "[smoke] $(date +%H:%M:%S) $cfg on GPU$g"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed=0 \
      train_steps=$STEPS log_interval=100 eval_interval=1000000 \
      save_interval=1000000 wandb_mode=offline \
      output_root="$OUT" exp_name="smoke_$cfg" \
      > "$LOG/$cfg.log" 2>&1 &
  sleep 3
done
wait

echo
echo "================ SMOKE RESULTS ================"
fail=0
for cfg in "${CFGS[@]}"; do
  # a run that reached the step loop prints tqdm progress; a build failure prints a traceback
  if grep -qE "Traceback|AssertionError|Error" "$LOG/$cfg.log" 2>/dev/null; then
    echo "FAIL  $cfg"
    grep -E "AssertionError|Error:" "$LOG/$cfg.log" | tail -2 | sed 's/^/        /'
    fail=1
  elif tr '\r' '\n' < "$LOG/$cfg.log" | grep -qE "$STEPS/$STEPS"; then
    echo "OK    $cfg"
  else
    echo "UNSURE $cfg  (no completion marker; inspect $LOG/$cfg.log)"
    fail=1
  fi
done
echo "==============================================="
exit $fail
