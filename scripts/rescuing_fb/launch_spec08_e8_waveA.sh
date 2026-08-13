#!/usr/bin/env bash
# Spec 08 E8 -- WAVE A: phase 1 + anchors for the two-phase curriculum across ALL NINE
# OGBench state-based manipulation environments, 3 seeds, 500k steps.
#
# E7 covered cube-double + scene (2 seeds). E8 completes the suite and tops every cell up
# to 3 seeds. The recipe is frozen at the cube-single winner (0.967); the ONLY per-env
# variable is `measure_hit_spec`, derived from each env's own success criterion and
# verified against privileged state (scripts/rescuing_fb/verify_hit_specs.py).
#
# Two cells per env in this wave:
#   <env>_state_measure_firstpassage_noortho_d50   phase 1, the reaching value
#   <env>_state_gciql_flow_bilinear_d50            GC-IQL anchor, the bar to clear
# Phase 2 warm-starts from this wave's checkpoints and is launched separately by
# launch_spec08_e8_waveB.sh once these finish.
#
# WHY BOTH CELLS PER ENV: the note's 0.90 bar is cube-single-only, and E7 showed the bar
# moves a lot (scene GC-IQL bilinear-d50 = 0.292). Running the anchor alongside is also
# what makes a FLOOR interpretable -- same data, same actor class, same goal event, so if
# both floor, the env is hard for goal-conditioned TD generally rather than for the
# measure objective specifically.
#
# REGION DENSITY (measure_region_density.py, off-diagonal positives per row at B=1024):
#   cube-single 13.90 | scene 2.85 | puzzle-3x3 2.04 | cube-double 0.46
#   puzzle-4x4 0.06 | cube-triple 0.03 | cube-quadruple 0.01 | puzzle-4x5 0.01 | puzzle-4x6 0.01
# The bottom five have essentially no off-diagonal signal -- only the 1024 diagonal
# entries are grounded per step. E7's lesson is that sparsity alone does not predict
# failure (scene's sparser conjunction beat its denser control 2x), but it has never been
# tested three orders of magnitude down. The anchors are the control for exactly this.
#
# ALREADY DONE / RUNNING, not relaunched here:
#   cube-single phase 1 s0,s1,s2 (exp_spec06, 0.967)   scene phase 1 s0,s1 (0.468/0.480)
#   cube-double phase 1 s0,s1 and anchor s0,s1 (in flight in tmux s08e7)
#
# Run in tmux:
#   tmux new -d -s s08e8a 'scripts/rescuing_fb/launch_spec08_e8_waveA.sh 2>&1 | tee /dev/shm/spec08_logs/e8a_dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_spec08_e8_waveA.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python          # run_gciql.py is orchestration; it spawns the jax child
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec08}
LOG=/dev/shm/spec08_logs/e8a; mkdir -p "$LOG" "$OUTROOT"
DRY_RUN=${DRY_RUN:-0}
# total concurrent training processes across the WHOLE box, including the 4 E7 jobs still
# running. Wave 1 ran 10 comfortably on this 8xL4 / 192-core host.
MAX=${MAX:-14}
GPUS=(0 1 2 3 4 5 6 7)

active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }
pick_gpu() { echo "${GPUS[$(( $1 % ${#GPUS[@]} ))]}"; }

# tag|config|seed|extra-overrides
# Ordered so the CHEAP, HIGH-INFORMATION envs land first: if something is systematically
# wrong, it shows on puzzle-3x3 / cube-triple long before the 1.9 GB cube-quadruple runs.
# SCOPE (2026-07-25): restricted to the FOUR envs for which a published vanilla-FB number
# exists, so every cell has an external reference point. These are exactly TD-JEPA's
# `sweep_cube` set (scripts/train/proprio/launch_fb_ogbench.py in facebookresearch/td_jepa):
#
#   env           FB (TD-JEPA, state)   GCIQL (OGBench paper, state)   region density
#   cube-single       49.6 +- 3.8              68 +- 6                   13.90/row
#   cube-double        2.6 +- 0.4              40 +- 5                    0.46/row
#   scene             12.8 +- 1.6              51 +- 4                    2.85/row
#   puzzle-3x3         4.8 +- 0.7              95 +- 1 (VERIFY)           2.04/row
#
# These four are also the four DENSEST regions in the suite, so the sparsity risk that
# dominated the 9-env plan is largely absent here (worst case cube-double at 30x).
#
# The cube-triple / cube-quadruple / puzzle-4x4 / 4x5 / 4x6 configs are written, smoke-
# tested and left in place; re-add their job lines below to extend the wave.
JOBS=(
  # --- phase 1: top-ups for envs E7 already started ---
  "p1_cd_s2|cube_double_state_measure_firstpassage_noortho_d50|2|"
  "p1_sc_s2|scene_state_measure_firstpassage_noortho_d50|2|"

  # --- phase 1: puzzle-3x3 (new env) ---
  "p1_p3_s0|puzzle_3x3_state_measure_firstpassage_noortho_d50|0|"
  "p1_p3_s1|puzzle_3x3_state_measure_firstpassage_noortho_d50|1|"
  "p1_p3_s2|puzzle_3x3_state_measure_firstpassage_noortho_d50|2|"

  # --- anchors: bilinear-d50 GC-IQL, rerun on all four envs for a matched comparison ---
  # cube-single's pinned anchor config is a 1M-step campaign cell; override to 500k so
  # every anchor in the table is measured at the same budget.
  "an_cs_s0|cube_single_state_gciql_flow_bilinear_d50|0|train_steps=500000 eval_interval=50000 save_interval=100000"
  "an_cs_s1|cube_single_state_gciql_flow_bilinear_d50|1|train_steps=500000 eval_interval=50000 save_interval=100000"
  "an_cs_s2|cube_single_state_gciql_flow_bilinear_d50|2|train_steps=500000 eval_interval=50000 save_interval=100000"
  "an_cd_s2|cube_double_state_gciql_flow_bilinear_d50|2|"
  "an_sc_s0|scene_state_gciql_flow_bilinear_d50|0|"
  "an_sc_s1|scene_state_gciql_flow_bilinear_d50|1|"
  "an_sc_s2|scene_state_gciql_flow_bilinear_d50|2|"
  "an_p3_s0|puzzle_3x3_state_gciql_flow_bilinear_d50|0|"
  "an_p3_s1|puzzle_3x3_state_gciql_flow_bilinear_d50|1|"
  "an_p3_s2|puzzle_3x3_state_gciql_flow_bilinear_d50|2|"
)

echo "[e8a] $(date +%H:%M:%S) ${#JOBS[@]} jobs, MAX=$MAX concurrent, out=$OUTROOT"
n=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed extra <<<"$j"
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[e8a] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 60
  done
  g=$(pick_gpu "$n")
  n=$((n+1))
  echo "[e8a] $(date +%H:%M:%S) launch #$n/${#JOBS[@]} $tag cfg=$cfg seed=$seed on GPU$g"
  if [ "$DRY_RUN" = "1" ]; then
    echo "      DRY: $PY run_gciql.py --config-name $cfg seed=$seed $extra output_root=$OUTROOT exp_name=$tag"
    continue
  fi
  # shellcheck disable=SC2086  # $extra is an intentional word-split override list
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" $extra \
      output_root="$OUTROOT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45     # jax co-init stagger (a simultaneous first-eval segfaulted an earlier wave)
done
wait
echo "[e8a] $(date +%H:%M:%S) ALL WAVE-A JOBS EXITED"
