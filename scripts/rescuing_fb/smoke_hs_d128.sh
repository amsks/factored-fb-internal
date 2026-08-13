#!/usr/bin/env bash
# Ledger §9.16c -- smoke the six new configs before committing GPU-hours.
#
# Two families:
#   hindsight scale-out  puzzle-3x3 + cube-single(ZX), both WARM-STARTED -- these exercise the
#                        real risk: tpd='hindsight' composed with a 9-block puzzle hit spec,
#                        and composed with the ZX knobs (fb_forward_z / actor_cond=z /
#                        actor_z_from_critic) whose param tree the splice must match.
#   d128 rank campaign   cube-double + scene, phase 1 (cold by design) and phase 2 (smoked
#                        COLD here -- the d128 phase-1 checkpoint it will warm-start from does
#                        not exist yet). The phase-2 splice is therefore NOT covered by this
#                        script; launch_d128_2stage.sh re-smokes each stage-2 cell against the
#                        real d128 checkpoint and aborts on failure before the 500k run.
#
# Usage:  scripts/rescuing_fb/smoke_hs_d128.sh 2>&1 | tee /dev/shm/spec08_logs/hs_d128_smoke.log
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
LOG=/dev/shm/spec08_logs/hs_d128_smoke; mkdir -p "$LOG"
OUT=/dev/shm/hs_d128_smoke; mkdir -p "$OUT"
STEPS=${STEPS:-300}
GPUS=(0 1 2 3 4 5)
FP=/home/adityamohan/exp_spec08/factored-fb
SP7=/home/adityamohan/exp_spec07/factored-fb

# cfg|warm-start run dir ("-" = cold)
CELLS=(
  "puzzle_3x3_state_measure_ballvisit_hs_d50|$FP/GCIQL-Puzzle3x3-Flow-Measure-FirstPassage-NoOrtho-d50/p1_p3_s0"
  "cube_single_state_measure_ballvisit_hs_zx_d50|$SP7/GCIQL-State-Flow-Measure-FP-ZX-d50/zx_s0"
  "cube_double_state_measure_firstpassage_noortho_d128|-"
  "scene_state_measure_firstpassage_noortho_d128|-"
  "cube_double_state_measure_ballvisit_d128|-"
  "scene_state_measure_ballvisit_d128|-"
)

n=0
for cell in "${CELLS[@]}"; do
  IFS='|' read -r cfg warm <<<"$cell"
  g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1))
  WARMARGS=()
  if [ "$warm" != "-" ]; then
    if [ ! -f "$warm/params_500000.pkl" ]; then
      echo "[smoke] FATAL $cfg -- warm-start ckpt missing: $warm/params_500000.pkl"; exit 1
    fi
    WARMARGS=("+agent_overrides.warm_start_ckpt=$warm" "+agent_overrides.warm_start_epoch=500000")
    echo "[smoke] $cfg on GPU$g (warm<-$warm)"
  else
    echo "[smoke] $cfg on GPU$g (cold)"
  fi
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" \
      train_steps=$STEPS log_interval=100 eval_interval=1000000 save_interval=1000000 \
      "${WARMARGS[@]}" \
      output_root=$OUT exp_name="smoke_$cfg" wandb_mode=disabled \
      > "$LOG/$cfg.log" 2>&1 &
done
wait

echo "== smoke results =="
fail=0
for cell in "${CELLS[@]}"; do
  cfg=${cell%%|*}
  if grep -q "step $((STEPS-STEPS%100))" "$LOG/$cfg.log" 2>/dev/null || tail -5 "$LOG/$cfg.log" | grep -qi "done\|finished\|total_time"; then
    echo "OK   $cfg"
  else
    echo "FAIL $cfg  (tail below)"; tail -20 "$LOG/$cfg.log"; fail=1
  fi
done
exit $fail
