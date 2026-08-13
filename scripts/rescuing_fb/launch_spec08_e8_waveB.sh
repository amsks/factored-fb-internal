#!/usr/bin/env bash
# Spec 08 E8 -- WAVE B: PHASE 2 (ball-visitation) of the two-phase curriculum, on every
# environment whose phase-1 checkpoint exists. 3 seeds, 500k steps.
#
# Phase 2 re-trains the forward tower phi into a visitation measure on the FROZEN phase-1
# goal embedding psi: non-absorbing backup (revisits keep counting), measure_stopgrad_psi,
# centered actor normalizer, every module warm-started from the matching phase-1 run.
# On cube-single this held the phase-1 level (0.952/0.928) with the critic scale plateauing
# at ~24 against a Monte-Carlo visitation mass of 24.7.
#
# This script is DEPENDENT on wave A: it refuses to launch any cell whose phase-1
# params_500000.pkl is not on disk, and reports what it skipped. Re-run it as wave-A cells
# land -- already-launched tags are skipped, so it is safe to run repeatedly.
#
# Run in tmux:
#   tmux new -d -s s08e8b 'scripts/rescuing_fb/launch_spec08_e8_waveB.sh 2>&1 | tee -a /dev/shm/spec08_logs/e8b_dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_spec08_e8_waveB.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec08}
SPEC06=${SPEC06:-/home/adityamohan/exp_spec06}
LOG=/dev/shm/spec08_logs/e8b; mkdir -p "$LOG" "$OUTROOT"
DRY_RUN=${DRY_RUN:-0}
MAX=${MAX:-14}
GPUS=(0 1 2 3 4 5 6 7)
EPOCH=500000

active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }
pick_gpu() { echo "${GPUS[$(( $1 % ${#GPUS[@]} ))]}"; }

# tag|phase2-config|seed|phase-1 run dir (the warm-start source)
# cube-single's phase-1 winner lives in exp_spec06 under its own run names; every other
# env's phase-1 comes from this campaign's E7/E8 waves in exp_spec08.
FP=$OUTROOT/factored-fb
JOBS=(
  # cube-single phase 2 seeds 0 and 1 already exist and are the note's headline numbers
  # (0.952/0.928): exp_spec07/factored-fb/GCIQL-State-Flow-Measure-BallVisit-FrozenPsi-d50/
  # vis_s0, vis_s1. Only seed 2 is missing, so only seed 2 is launched here.
  "p2_cs_s2|cube_single_state_measure_ballvisit_d50|2|$SPEC06/factored-fb/GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50/m_firstpassage_noortho_s2"

  "p2_cd_s0|cube_double_state_measure_ballvisit_d50|0|$FP/GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d50/cd_fp_s0"
  "p2_cd_s1|cube_double_state_measure_ballvisit_d50|1|$FP/GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d50/cd_fp_s1"
  "p2_cd_s2|cube_double_state_measure_ballvisit_d50|2|$FP/GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d50/p1_cd_s2"

  "p2_sc_s0|scene_state_measure_ballvisit_d50|0|$FP/GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d50/sc_fp_s0"
  "p2_sc_s1|scene_state_measure_ballvisit_d50|1|$FP/GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d50/sc_fp_s1"
  "p2_sc_s2|scene_state_measure_ballvisit_d50|2|$FP/GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d50/p1_sc_s2"

  "p2_p3_s0|puzzle_3x3_state_measure_ballvisit_d50|0|$FP/GCIQL-Puzzle3x3-Flow-Measure-FirstPassage-NoOrtho-d50/p1_p3_s0"
  "p2_p3_s1|puzzle_3x3_state_measure_ballvisit_d50|1|$FP/GCIQL-Puzzle3x3-Flow-Measure-FirstPassage-NoOrtho-d50/p1_p3_s1"
  "p2_p3_s2|puzzle_3x3_state_measure_ballvisit_d50|2|$FP/GCIQL-Puzzle3x3-Flow-Measure-FirstPassage-NoOrtho-d50/p1_p3_s2"

#  "p2_ct_s0|cube_triple_state_measure_ballvisit_d50|0|$FP/GCIQL-CubeTriple-Flow-Measure-FirstPassage-NoOrtho-d50/p1_ct_s0"
#  "p2_ct_s1|cube_triple_state_measure_ballvisit_d50|1|$FP/GCIQL-CubeTriple-Flow-Measure-FirstPassage-NoOrtho-d50/p1_ct_s1"
#  "p2_ct_s2|cube_triple_state_measure_ballvisit_d50|2|$FP/GCIQL-CubeTriple-Flow-Measure-FirstPassage-NoOrtho-d50/p1_ct_s2"

#  "p2_p44_s0|puzzle_4x4_state_measure_ballvisit_d50|0|$FP/GCIQL-Puzzle4x4-Flow-Measure-FirstPassage-NoOrtho-d50/p1_p44_s0"
#  "p2_p44_s1|puzzle_4x4_state_measure_ballvisit_d50|1|$FP/GCIQL-Puzzle4x4-Flow-Measure-FirstPassage-NoOrtho-d50/p1_p44_s1"
#  "p2_p44_s2|puzzle_4x4_state_measure_ballvisit_d50|2|$FP/GCIQL-Puzzle4x4-Flow-Measure-FirstPassage-NoOrtho-d50/p1_p44_s2"

#  "p2_p45_s0|puzzle_4x5_state_measure_ballvisit_d50|0|$FP/GCIQL-Puzzle4x5-Flow-Measure-FirstPassage-NoOrtho-d50/p1_p45_s0"
#  "p2_p45_s1|puzzle_4x5_state_measure_ballvisit_d50|1|$FP/GCIQL-Puzzle4x5-Flow-Measure-FirstPassage-NoOrtho-d50/p1_p45_s1"
#  "p2_p45_s2|puzzle_4x5_state_measure_ballvisit_d50|2|$FP/GCIQL-Puzzle4x5-Flow-Measure-FirstPassage-NoOrtho-d50/p1_p45_s2"

#  "p2_p46_s0|puzzle_4x6_state_measure_ballvisit_d50|0|$FP/GCIQL-Puzzle4x6-Flow-Measure-FirstPassage-NoOrtho-d50/p1_p46_s0"
#  "p2_p46_s1|puzzle_4x6_state_measure_ballvisit_d50|1|$FP/GCIQL-Puzzle4x6-Flow-Measure-FirstPassage-NoOrtho-d50/p1_p46_s1"
#  "p2_p46_s2|puzzle_4x6_state_measure_ballvisit_d50|2|$FP/GCIQL-Puzzle4x6-Flow-Measure-FirstPassage-NoOrtho-d50/p1_p46_s2"

#  "p2_cq_s0|cube_quadruple_state_measure_ballvisit_d50|0|$FP/GCIQL-CubeQuad-Flow-Measure-FirstPassage-NoOrtho-d50/p1_cq_s0"
#  "p2_cq_s1|cube_quadruple_state_measure_ballvisit_d50|1|$FP/GCIQL-CubeQuad-Flow-Measure-FirstPassage-NoOrtho-d50/p1_cq_s1"
#  "p2_cq_s2|cube_quadruple_state_measure_ballvisit_d50|2|$FP/GCIQL-CubeQuad-Flow-Measure-FirstPassage-NoOrtho-d50/p1_cq_s2"
)

echo "[e8b] $(date +%H:%M:%S) ${#JOBS[@]} candidate cells (scope: cube-single/double, scene, puzzle-3x3), MAX=$MAX"
n=0; launched=0; waiting=0; done_already=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed src <<<"$j"

  # already launched or finished? (run dir carries the tag as exp_name)
  if compgen -G "$FP/*/$tag" > /dev/null; then
    echo "[e8b] skip $tag - output dir already exists"
    done_already=$((done_already+1)); continue
  fi
  # phase-1 checkpoint present?
  if [ ! -f "$src/params_$EPOCH.pkl" ]; then
    echo "[e8b] WAIT $tag - phase-1 ckpt missing: $src/params_$EPOCH.pkl"
    waiting=$((waiting+1)); continue
  fi

  while [ "$(active)" -ge "$MAX" ]; do
    echo "[e8b] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 60
  done
  g=$(pick_gpu "$n"); n=$((n+1)); launched=$((launched+1))
  echo "[e8b] $(date +%H:%M:%S) launch $tag cfg=$cfg seed=$seed on GPU$g  warm<-$src"
  if [ "$DRY_RUN" = "1" ]; then
    echo "      DRY: --config-name $cfg seed=$seed +agent_overrides.warm_start_ckpt=$src +agent_overrides.warm_start_epoch=$EPOCH"
    continue
  fi
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      "+agent_overrides.warm_start_ckpt=$src" "+agent_overrides.warm_start_epoch=$EPOCH" \
      output_root="$OUTROOT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[e8b] $(date +%H:%M:%S) launched=$launched  already_present=$done_already  awaiting_phase1=$waiting"
[ "$waiting" -gt 0 ] && echo "[e8b] re-run this script when more wave-A cells finish"
exit 0
