#!/usr/bin/env bash
# Ledger §9.16c wave A -- hindsight scale-out + the arm-2 scene replicate.
#
# §9.16b arm 3 (measure_testpoint_dist=hindsight) was the only arm to beat its control on both
# regression envs: cube-double 0.500/0.544 (control 0.368/0.348, phase 1 0.549) and scene
# 0.916/0.904 (control 0.332/0.352, phase 1 0.484), with the 0->100k decay eliminated. Before
# that claim enters the paper it needs a third env, a control env, and a seed-2 replicate:
#
#   p2h_p3_s{0,1}  puzzle-3x3   -- the hardest case: §9.16 records it FLAT at the degenerate
#                                  0.200 under canonical phase 2, grounding rate 0.20%.
#                                  Tests whether hindsight ADDS structure or only protects it.
#   p2h_cs_s{0,1}  cube-single  -- the control direction: does hindsight COST anything on the
#                                  one env where batch test-points already work (0.976/0.984)?
#                                  ZX form, warm-started from the same FP-ZX parents as visz_*.
#   p2h_cd_s2 / p2h_sc_s2       -- seed-2 replicates of the two cells that won.
#   p2e_sc_s1                   -- relaunch: the original stopped after params_400000 (last
#                                  eval 450k). The arm is NULL either way, this only closes the
#                                  record. The partial run is MOVED ASIDE, never deleted.
#
# Pre-registered reads live in the config headers and ledger §9.16b/§9.16c.
#
# Run in tmux:
#   tmux new -d -s hsd128 'scripts/rescuing_fb/launch_hs_scaleout.sh 2>&1 | tee -a /dev/shm/spec08_logs/hs_dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_hs_scaleout.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec08}
LOG=/dev/shm/spec08_logs/hs_scaleout; mkdir -p "$LOG" "$OUTROOT"
DRY_RUN=${DRY_RUN:-0}
MAX=${MAX:-12}
GPUS=(0 1 2 3 4 5 6 7)
EPOCH=500000
FP=$OUTROOT/factored-fb
SP7=/home/adityamohan/exp_spec07/factored-fb

active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }

# --- one-time: move the truncated arm-2 scene seed aside so the tag is free ---
PARTIAL=$FP/GCIQL-Scene-Flow-Measure-BallVisit-EpsDM-d50/p2e_sc_s1
if [ -d "$PARTIAL" ] && [ ! -f "$PARTIAL/params_500000.pkl" ]; then
  echo "[hs] arm-2 scene seed 1 is truncated (no params_500000) -- moving aside, not deleting"
  if [ "$DRY_RUN" = "1" ]; then
    echo "      DRY: mv $PARTIAL ${PARTIAL}_partial400k"
  else
    mv "$PARTIAL" "${PARTIAL}_partial400k" || { echo "[hs] FATAL could not move partial"; exit 1; }
  fi
fi

# tag|config|seed|phase-1 warm-start run dir
JOBS=(
  "p2h_p3_s0|puzzle_3x3_state_measure_ballvisit_hs_d50|0|$FP/GCIQL-Puzzle3x3-Flow-Measure-FirstPassage-NoOrtho-d50/p1_p3_s0"
  "p2h_p3_s1|puzzle_3x3_state_measure_ballvisit_hs_d50|1|$FP/GCIQL-Puzzle3x3-Flow-Measure-FirstPassage-NoOrtho-d50/p1_p3_s1"

  "p2h_cs_s0|cube_single_state_measure_ballvisit_hs_zx_d50|0|$SP7/GCIQL-State-Flow-Measure-FP-ZX-d50/zx_s0"
  "p2h_cs_s1|cube_single_state_measure_ballvisit_hs_zx_d50|1|$SP7/GCIQL-State-Flow-Measure-FP-ZX-d50/zx_s1"

  "p2h_cd_s2|cube_double_state_measure_ballvisit_hs_d50|2|$FP/GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d50/p1_cd_s2"
  "p2h_sc_s2|scene_state_measure_ballvisit_hs_d50|2|$FP/GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d50/p1_sc_s2"

  "p2e_sc_s1|scene_state_measure_ballvisit_epsdm_d50|1|$FP/GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d50/sc_fp_s1"
)

echo "[hs] $(date +%H:%M:%S) ${#JOBS[@]} cells (hindsight scale-out + arm-2 replicate), MAX=$MAX"
n=0; launched=0; skipped=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed src <<<"$j"
  if compgen -G "$FP/*/$tag" > /dev/null; then
    echo "[hs] skip $tag - output dir already exists"; skipped=$((skipped+1)); continue
  fi
  if [ ! -f "$src/params_$EPOCH.pkl" ]; then
    echo "[hs] FATAL $tag - phase-1 ckpt missing: $src/params_$EPOCH.pkl"; exit 1
  fi
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[hs] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 60
  done
  g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1)); launched=$((launched+1))
  echo "[hs] $(date +%H:%M:%S) launch $tag cfg=$cfg seed=$seed on GPU$g  warm<-$src"
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
echo "[hs] $(date +%H:%M:%S) launched=$launched skipped=$skipped"
