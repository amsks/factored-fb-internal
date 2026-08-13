#!/usr/bin/env bash
# Spec 08 E7 -- WAVE 1: phase 1 of the two-phase curriculum on a SECOND and THIRD environment.
# Spec: PAPER/rescuing-fb/specs/08-two-phase-next-experiments.md (E7 "Scope: second environment").
# Note: PAPER/rescuing-fb/two-phase-successor-measure-note.tex (its lead caveat is
# "one environment so far (cube-single, state observations)").
#
# Cells (2 seeds each, 500k, cold start):
#   cd_fp     cube-double, goal region = BOTH cubes within 4cm   (env-faithful conjunction)
#   cd_fp1c   cube-double, goal region = cube 0 only             (sparsity control)
#   sc_fp     scene,       goal region = cube+2 buttons+drawer+window (env-faithful, 5-way)
#   sc_fpc    scene,       goal region = cube only               (sparsity control)
#   cd_anch   cube-double, GC-IQL bilinear-d50 flow              (the MISSING anchor)
#
# The controls exist because the conjunctive regions are much sparser than cube-single's.
# Measured off-diagonal positives per row at the training batch size B=1024:
#   cube-single 14.0 | cube-double 2-cube 0.51 | cube-double 1-cube 18.6
#                    | scene 5-way 3.07        | scene cube-only 61.9
# Without the controls a floor on the conjunctive cells would be uninterpretable.
#
# Run in tmux:
#   tmux new -d -s s08e7 'scripts/rescuing_fb/launch_spec08_e7_phase1.sh 2>&1 | tee /dev/shm/spec08_logs/e7_dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_spec08_e7_phase1.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python                  # run_gciql.py is orchestration only; it spawns
                                            # the jax child itself (configs/gciql/storage/shm.yaml)
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec08}
LOG=/dev/shm/spec08_logs; mkdir -p "$LOG" "$OUTROOT"
MAX=${MAX:-10}
DRY_RUN=${DRY_RUN:-0}
GPUS=(0 1 2 3 4 5 6 7)                      # box is idle (verified); no other track running

# pgrep -fc prints the count AND exits non-zero when it is 0, so `|| echo 0` would emit "0\n0".
active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }

# Round-robin, NOT least-memory: on an idle box every GPU reports 0 MiB, so a least-memory
# pick returns GPU0 for every job in the burst (jax allocates lazily under
# XLA_PYTHON_CLIENT_PREALLOCATE=false, so 45s of stagger is not enough to break the tie).
pick_gpu() { echo "${GPUS[$(( $1 % ${#GPUS[@]} ))]}"; }

# tag|config|seed
JOBS=(
  "cd_fp_s0|cube_double_state_measure_firstpassage_noortho_d50|0"
  "cd_fp_s1|cube_double_state_measure_firstpassage_noortho_d50|1"
  "sc_fp_s0|scene_state_measure_firstpassage_noortho_d50|0"
  "sc_fp_s1|scene_state_measure_firstpassage_noortho_d50|1"
  "cd_fp1c_s0|cube_double_state_measure_fp_noortho_1cube_d50|0"
  "cd_fp1c_s1|cube_double_state_measure_fp_noortho_1cube_d50|1"
  "sc_fpc_s0|scene_state_measure_fp_noortho_cubeonly_d50|0"
  "sc_fpc_s1|scene_state_measure_fp_noortho_cubeonly_d50|1"
  "cd_anch_s0|cube_double_state_gciql_flow_bilinear_d50|0"
  "cd_anch_s1|cube_double_state_gciql_flow_bilinear_d50|1"
)

n=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed <<<"$j"
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[e7] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 60
  done
  g=$(pick_gpu "$n")
  n=$((n+1))
  echo "[e7] $(date +%H:%M:%S) launch #$n/${#JOBS[@]} $tag cfg=$cfg seed=$seed on GPU$g"
  if [ "$DRY_RUN" = "1" ]; then
    echo "      DRY: $PY run_gciql.py --config-name $cfg seed=$seed output_root=$OUTROOT exp_name=$tag"
    continue
  fi
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      output_root="$OUTROOT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45     # jax co-init stagger (a simultaneous first-eval segfaulted the wave-2 campaign)
done
wait
echo "[e7] $(date +%H:%M:%S) ALL PHASE-1 JOBS EXITED"
