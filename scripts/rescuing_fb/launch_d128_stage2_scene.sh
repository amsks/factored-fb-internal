#!/usr/bin/env bash
# Ledger §9.16c wave B -- scene-only stage 2 of the d128 rank campaign.
#
# WHY A SEPARATE SCRIPT: launch_d128_2stage.sh walks its stage-2 list serially and the first
# entry (p2_cd_d128_s0) polls for the cube-double stage-1 checkpoint. The scene stage-1 parents
# finished 2026-08-01 23:06 and 16:48, so p2_sc_d128_s{0,1} sat blocked behind that wait loop
# for ~19h with six GPUs idle. That script is currently RUNNING in tmux `d128rank`; bash reads a
# script lazily by byte offset, so editing it in place risks corrupting the live interpreter.
# This file duplicates only the scene stage-2 path and pins the two idle GPUs.
#
# Collision safety: launch dirs are created immediately, so when `d128rank` eventually reaches
# its own p2_sc_d128_* entries its `compgen -G "$FP/*/$tag"` guard will skip them.
#
# Run in tmux:
#   tmux new -d -s d128sc 'scripts/rescuing_fb/launch_d128_stage2_scene.sh 2>&1 | tee -a /dev/shm/spec08_logs/d128sc_dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_d128_stage2_scene.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec08}
LOG=/dev/shm/spec08_logs/d128; mkdir -p "$LOG" "$OUTROOT"
DRY_RUN=${DRY_RUN:-0}
SKIP_PRECHECK=${SKIP_PRECHECK:-0}
EPOCH=500000
FP=$OUTROOT/factored-fb
GPUS=(2 3)   # 0 and 1 are carrying the four live jobs

# tag|config|seed|stage-1 parent run dir
S2=(
  "p2_sc_d128_s0|scene_state_measure_ballvisit_d128|0|$FP/GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d128/p1_sc_d128_s0"
  "p2_sc_d128_s1|scene_state_measure_ballvisit_d128|1|$FP/GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d128/p1_sc_d128_s1"
)

# Did a pre-check log record a clean 300-step run off a complete d128 splice?
#
# The parent script's heuristic (grep "step 300", or done/finished/total_time in the last 5
# lines) is a FALSE NEGATIVE on this path: run_gciql emits a tqdm bar and then echoes the
# resolved command line, so no such marker is ever in the tail. Both scene seeds completed
# 300/300 with a full splice and were still reported FAIL. Gate on the substance instead:
#   - the splice happened and covered every module tree (`fresh-init []` -- a non-empty list
#     means a tree was silently randomly re-initialized, the exact d128 shape-mismatch failure
#     this pre-check exists to catch)
#   - the 300 steps actually completed
#   - nothing threw
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
  # Reuse an earlier passing pre-check rather than burning another ~10 GPU-minutes on it.
  if precheck_passed "$plog"; then
    echo "[d128sc] precheck CACHED OK $tag ($(basename "$plog"))"; return 0
  fi
  echo "[d128sc] $(date +%H:%M:%S) splice pre-check $tag (300 steps, d128 warm-start) on GPU$gpu"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpu \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      train_steps=300 log_interval=100 eval_interval=1000000 save_interval=1000000 \
      "+agent_overrides.warm_start_ckpt=$src" "+agent_overrides.warm_start_epoch=$EPOCH" \
      output_root=/dev/shm/d128_precheck exp_name="pre_$tag" wandb_mode=disabled \
      > "$plog" 2>&1
  if precheck_passed "$plog"; then
    echo "[d128sc] precheck OK $tag"; return 0
  fi
  echo "[d128sc] precheck FAIL $tag -- NOT launching the 500k run. Tail:"; tail -20 "$plog"; return 1
}

echo "[d128sc] $(date +%H:%M:%S) scene stage 2 -- ${#S2[@]} cells on GPUs ${GPUS[*]}"
n=0; launched=0
for j in "${S2[@]}"; do
  IFS='|' read -r tag cfg seed src <<<"$j"
  if compgen -G "$FP/*/$tag" > /dev/null; then
    echo "[d128sc] skip $tag - output dir already exists"; continue
  fi
  if [ ! -f "$src/params_$EPOCH.pkl" ]; then
    echo "[d128sc] FATAL $tag - stage-1 parent missing: $src/params_$EPOCH.pkl"; exit 1
  fi
  g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1))
  if ! precheck "$tag" "$cfg" "$seed" "$src" "$g"; then
    echo "[d128sc] SKIPPING $tag after failed pre-check"; continue
  fi
  echo "[d128sc] $(date +%H:%M:%S) launch $tag cfg=$cfg seed=$seed on GPU$g  warm<-$src"
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
echo "[d128sc] $(date +%H:%M:%S) launched=$launched"
