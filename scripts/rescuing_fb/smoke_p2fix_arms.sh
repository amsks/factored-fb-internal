#!/usr/bin/env bash
# Ledger §9.16 arms -- smoke test the six new phase-2 configs (freeze_actor /
# density-matched eps / hindsight test points, on cube-double + scene) for a few
# hundred gradient steps each before committing GPU-hours. Exercises the
# assert-lattice (freeze_actor optimizer mask, tpd='hindsight' composition with a
# conjunctive hit spec), _parse_hit_spec on the widened eps strings, and the
# warm-start splice at the real observation widths.
#
# Usage:  scripts/rescuing_fb/smoke_p2fix_arms.sh 2>&1 | tee /dev/shm/spec08_logs/p2fix_smoke.log
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
LOG=/dev/shm/spec08_logs/p2fix_smoke; mkdir -p "$LOG"
OUT=/dev/shm/p2fix_smoke; mkdir -p "$OUT"
STEPS=${STEPS:-300}
GPUS=(0 1 2 3 4 5)
FP=/home/adityamohan/exp_spec08/factored-fb

declare -A WARM=(
  [cube_double]=$FP/GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d50/cd_fp_s0
  [scene]=$FP/GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d50/sc_fp_s0
)

CFGS=(
  cube_double_state_measure_ballvisit_frzactor_d50
  scene_state_measure_ballvisit_frzactor_d50
  cube_double_state_measure_ballvisit_epsdm_d50
  scene_state_measure_ballvisit_epsdm_d50
  cube_double_state_measure_ballvisit_hs_d50
  scene_state_measure_ballvisit_hs_d50
)

n=0
for cfg in "${CFGS[@]}"; do
  env=${cfg%%_state_*}
  g=${GPUS[$((n % ${#GPUS[@]}))]}
  echo "[smoke] $cfg on GPU$g (warm<-${WARM[$env]})"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" \
      train_steps=$STEPS log_interval=100 eval_interval=1000000 save_interval=1000000 \
      "+agent_overrides.warm_start_ckpt=${WARM[$env]}" "+agent_overrides.warm_start_epoch=500000" \
      output_root=$OUT exp_name="smoke_$cfg" wandb_mode=disabled \
      > "$LOG/$cfg.log" 2>&1 &
  n=$((n+1))
done
wait
echo "== smoke results =="
fail=0
for cfg in "${CFGS[@]}"; do
  if grep -q "step $((STEPS-STEPS%100))" "$LOG/$cfg.log" 2>/dev/null || tail -5 "$LOG/$cfg.log" | grep -qi "done\|finished\|total_time"; then
    echo "OK   $cfg"
  else
    echo "FAIL $cfg  (tail below)"; tail -15 "$LOG/$cfg.log"; fail=1
  fi
done
exit $fail
