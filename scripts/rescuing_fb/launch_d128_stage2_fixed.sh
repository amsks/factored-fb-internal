#!/usr/bin/env bash
# Ledger §9.16c wave B -- stage 2 of the d128 rank campaign, all four cells, with a WORKING
# splice pre-check. Supersedes launch_d128_stage2_scene.sh (scene-only stopgap).
#
# WHY THIS EXISTS: launch_d128_2stage.sh decides pre-check pass/fail by grepping for "step 300"
# or done/finished/total_time in the last 5 log lines. run_gciql emits a tqdm bar and then
# echoes the resolved command line, so that marker is NEVER in the tail and the check can only
# ever fail. Observed 2026-08-02: all four stage-2 cells completed their 300-step pre-check off
# a complete splice and all four were reported FAIL and skipped -- the cube-double pair would
# have been silently dropped from the campaign.
#
# The pre-check itself is worth keeping: a d128 warm-start splice had never been exercised, and
# a shape mismatch there is a silent 10-hour waste. It just has to gate on the substance:
#   - `[warm_start] spliced ...; fresh-init []`  -- every module tree came from the parent. A
#     NON-EMPTY fresh-init list is the silent d128 shape-mismatch this check exists to catch.
#   - `300/300`  -- the steps actually ran.
#   - no Traceback/Exception.
# Verified against all four logs: seeds sc_s{0,1} and cd_s{0,1} spliced all five trees with
# fresh-init [] and ran clean. Passing logs are CACHED, so this never re-burns GPU minutes.
#
# Idempotent: cells whose output dir already exists are skipped, so the scene pair launched at
# 21:15 is left alone and only the cube-double pair starts.
#
# Run in tmux:
#   tmux new -d -s d128cd 'scripts/rescuing_fb/launch_d128_stage2_fixed.sh 2>&1 | tee -a /dev/shm/spec08_logs/d128fix_dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_d128_stage2_fixed.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec08}
LOG=/dev/shm/spec08_logs/d128; mkdir -p "$LOG" "$OUTROOT"
DRY_RUN=${DRY_RUN:-0}
SKIP_PRECHECK=${SKIP_PRECHECK:-0}
MAX=${MAX:-12}
EPOCH=500000
FP=$OUTROOT/factored-fb
GPUS=(4 5)   # 0-1 idle but historically the packed pair; 2-3 carry the scene stage-2 cells

active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }

# tag|config|seed|stage-1 parent run dir
S2=(
  "p2_cd_d128_s0|cube_double_state_measure_ballvisit_d128|0|$FP/GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d128/p1_cd_d128_s0"
  "p2_cd_d128_s1|cube_double_state_measure_ballvisit_d128|1|$FP/GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d128/p1_cd_d128_s1"
  "p2_sc_d128_s0|scene_state_measure_ballvisit_d128|0|$FP/GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d128/p1_sc_d128_s0"
  "p2_sc_d128_s1|scene_state_measure_ballvisit_d128|1|$FP/GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d128/p1_sc_d128_s1"
)

precheck_passed() { # logfile -> 0 pass / 1 fail
  local f=$1
  [ -s "$f" ] || return 1
  grep -qa '\[warm_start\] spliced' "$f" || return 1
  grep -qa 'fresh-init \[\]' "$f" || return 1
  grep -qa '300/300' "$f" || return 1
  grep -qaiE 'Traceback|Exception|error:' "$f" && return 1
  return 0
}

precheck() { # tag cfg seed src gpu -> 0 ok / 1 fail
  local tag=$1 cfg=$2 seed=$3 src=$4 gpu=$5
  local plog="$LOG/precheck_$tag.log"
  [ "$SKIP_PRECHECK" = "1" ] && return 0
  [ "$DRY_RUN" = "1" ] && { echo "      DRY: precheck $tag"; return 0; }
  if precheck_passed "$plog"; then
    echo "[d128fix] precheck CACHED OK $tag ($(basename "$plog"))"; return 0
  fi
  echo "[d128fix] $(date +%H:%M:%S) splice pre-check $tag (300 steps, d128 warm-start) on GPU$gpu"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpu \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      train_steps=300 log_interval=100 eval_interval=1000000 save_interval=1000000 \
      "+agent_overrides.warm_start_ckpt=$src" "+agent_overrides.warm_start_epoch=$EPOCH" \
      output_root=/dev/shm/d128_precheck exp_name="pre_$tag" wandb_mode=disabled \
      > "$plog" 2>&1
  if precheck_passed "$plog"; then
    echo "[d128fix] precheck OK $tag"; return 0
  fi
  echo "[d128fix] precheck FAIL $tag -- NOT launching the 500k run. Tail:"; tail -20 "$plog"; return 1
}

echo "[d128fix] $(date +%H:%M:%S) stage 2 -- ${#S2[@]} candidate cells, GPUs ${GPUS[*]}, MAX=$MAX"
n=0; launched=0; skipped=0
for j in "${S2[@]}"; do
  IFS='|' read -r tag cfg seed src <<<"$j"
  if compgen -G "$FP/*/$tag" > /dev/null; then
    echo "[d128fix] skip $tag - output dir already exists"; skipped=$((skipped+1)); continue
  fi
  if [ ! -f "$src/params_$EPOCH.pkl" ]; then
    echo "[d128fix] FATAL $tag - stage-1 parent missing: $src/params_$EPOCH.pkl"; exit 1
  fi
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[d128fix] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 60
  done
  g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1))
  if ! precheck "$tag" "$cfg" "$seed" "$src" "$g"; then
    echo "[d128fix] SKIPPING $tag after failed pre-check"; continue
  fi
  echo "[d128fix] $(date +%H:%M:%S) launch $tag cfg=$cfg seed=$seed on GPU$g  warm<-$src"
  if [ "$DRY_RUN" = "1" ]; then echo "      DRY: --config-name $cfg seed=$seed"; continue; fi
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      "+agent_overrides.warm_start_ckpt=$src" "+agent_overrides.warm_start_epoch=$EPOCH" \
      output_root="$OUTROOT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  launched=$((launched+1))
  sleep 45
done
wait
echo "[d128fix] $(date +%H:%M:%S) launched=$launched skipped=$skipped"
