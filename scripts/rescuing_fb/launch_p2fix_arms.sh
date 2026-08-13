#!/usr/bin/env bash
# Ledger §9.16 arms -- phase-2 fixes on the two envs where phase 2 regresses
# (cube-double −0.19, scene −0.14). Mechanism (§9.16): the visitation fixed point
# collapses to the hindsight dwell constant because off-diag grounding at eps=0.4 is
# 28x/4.5x sparser than cube-single; actor_q_center then amplifies the flat Q's residual
# noise into fast actor drift (first 50-100k steps).
#
# Arms (2 seeds each, warm-started from the matching E7 phase-1 ckpt, 500k):
#   frzactor  freeze_actor=true            -- regression erased by construction (E2 recipe)
#   epsdm     density-matched training eps -- cube-double 0.96 (1.21%), scene x1.9 (1.29%)
#   hs        measure_testpoint_dist=hindsight -- sparsity-independent grounding
# Pre-registered reads: ledger §9.16b.
#
# Run in tmux:
#   tmux new -d -s p2fix 'scripts/rescuing_fb/launch_p2fix_arms.sh 2>&1 | tee -a /dev/shm/spec08_logs/p2fix_dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_p2fix_arms.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec08}
LOG=/dev/shm/spec08_logs/p2fix; mkdir -p "$LOG" "$OUTROOT"
DRY_RUN=${DRY_RUN:-0}
MAX=${MAX:-12}
GPUS=(0 1 2 3 4 5 6 7)
EPOCH=500000
FP=$OUTROOT/factored-fb

active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }

# tag|config|seed|phase-1 warm-start run dir
JOBS=(
  "p2f_cd_s0|cube_double_state_measure_ballvisit_frzactor_d50|0|$FP/GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d50/cd_fp_s0"
  "p2f_cd_s1|cube_double_state_measure_ballvisit_frzactor_d50|1|$FP/GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d50/cd_fp_s1"
  "p2f_sc_s0|scene_state_measure_ballvisit_frzactor_d50|0|$FP/GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d50/sc_fp_s0"
  "p2f_sc_s1|scene_state_measure_ballvisit_frzactor_d50|1|$FP/GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d50/sc_fp_s1"

  "p2e_cd_s0|cube_double_state_measure_ballvisit_epsdm_d50|0|$FP/GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d50/cd_fp_s0"
  "p2e_cd_s1|cube_double_state_measure_ballvisit_epsdm_d50|1|$FP/GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d50/cd_fp_s1"
  "p2e_sc_s0|scene_state_measure_ballvisit_epsdm_d50|0|$FP/GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d50/sc_fp_s0"
  "p2e_sc_s1|scene_state_measure_ballvisit_epsdm_d50|1|$FP/GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d50/sc_fp_s1"

  "p2h_cd_s0|cube_double_state_measure_ballvisit_hs_d50|0|$FP/GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d50/cd_fp_s0"
  "p2h_cd_s1|cube_double_state_measure_ballvisit_hs_d50|1|$FP/GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d50/cd_fp_s1"
  "p2h_sc_s0|scene_state_measure_ballvisit_hs_d50|0|$FP/GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d50/sc_fp_s0"
  "p2h_sc_s1|scene_state_measure_ballvisit_hs_d50|1|$FP/GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d50/sc_fp_s1"
)

echo "[p2fix] $(date +%H:%M:%S) ${#JOBS[@]} cells (3 arms x cube-double/scene x 2 seeds), MAX=$MAX"
n=0; launched=0; done_already=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed src <<<"$j"
  if compgen -G "$FP/*/$tag" > /dev/null; then
    echo "[p2fix] skip $tag - output dir already exists"; done_already=$((done_already+1)); continue
  fi
  if [ ! -f "$src/params_$EPOCH.pkl" ]; then
    echo "[p2fix] FATAL $tag - phase-1 ckpt missing: $src/params_$EPOCH.pkl"; exit 1
  fi
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[p2fix] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 60
  done
  g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1)); launched=$((launched+1))
  echo "[p2fix] $(date +%H:%M:%S) launch $tag cfg=$cfg seed=$seed on GPU$g  warm<-$src"
  if [ "$DRY_RUN" = "1" ]; then
    echo "      DRY: --config-name $cfg seed=$seed"; continue
  fi
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      "+agent_overrides.warm_start_ckpt=$src" "+agent_overrides.warm_start_epoch=$EPOCH" \
      output_root="$OUTROOT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[p2fix] $(date +%H:%M:%S) launched=$launched already_present=$done_already"
