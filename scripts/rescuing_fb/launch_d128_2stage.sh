#!/usr/bin/env bash
# Ledger §9.16c wave B -- the d128 RANK campaign on cube-double + scene, both stages.
#
# WHY: §9.16b arm 2 (density-matched eps) came back NULL -- cd 0.340/0.380 vs control
# 0.368/0.348, q_abs 33.8 vs 34.6/35.3, neg-frac flat. The pre-registered NULL clause
# falsifies the sparsity story and names rank as the next suspect: a rank-50 bilinear critic
# may be unable to carry a multi-block conjunction plus the shared recurrence baseline, and
# the least-squares way out is to spend the rank on the baseline -- which IS the dwell
# constant the phase-2 fixed point collapses to (§9.16 Finding 2).
#
# TWO STAGES, because no d128 JAX phase-1 checkpoint exists and a d128 phase 2 cannot
# warm-start from a d50 param tree:
#   stage 1  p1_{cd,sc}_d128_s{0,1}   cold, 500k   (the *_firstpassage_noortho_d128 configs)
#   stage 2  p2_{cd,sc}_d128_s{0,1}   warm-started from the matching stage-1 run, 500k
#
# Stage 2 is chained: this script polls for each stage-1 params_500000.pkl, then runs a
# 300-step SPLICE PRE-CHECK against the real d128 checkpoint before committing the 500k run.
# That path (warm-start splice at latent_dim=128) has never been exercised; the pre-check
# costs ~2 minutes and turns a silent 10-hour waste into an immediate abort. Set
# SKIP_PRECHECK=1 to bypass it.
#
# Run in tmux:
#   tmux new -d -s d128 'scripts/rescuing_fb/launch_d128_2stage.sh 2>&1 | tee -a /dev/shm/spec08_logs/d128_dispatch.log'
# Dry run:     DRY_RUN=1 scripts/rescuing_fb/launch_d128_2stage.sh
# Stage 2 only (stage-1 ckpts already present):  STAGE=2 scripts/rescuing_fb/launch_d128_2stage.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec08}
LOG=/dev/shm/spec08_logs/d128; mkdir -p "$LOG" "$OUTROOT"
DRY_RUN=${DRY_RUN:-0}
SKIP_PRECHECK=${SKIP_PRECHECK:-0}
STAGE=${STAGE:-0}          # 0 = both, 1 = stage 1 only, 2 = stage 2 only
MAX=${MAX:-12}
GPUS=(0 1 2 3 4 5 6 7)
EPOCH=500000
FP=$OUTROOT/factored-fb

active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }

# tag|config|seed  (stage 1, cold)
S1=(
  "p1_cd_d128_s0|cube_double_state_measure_firstpassage_noortho_d128|0"
  "p1_cd_d128_s1|cube_double_state_measure_firstpassage_noortho_d128|1"
  "p1_sc_d128_s0|scene_state_measure_firstpassage_noortho_d128|0"
  "p1_sc_d128_s1|scene_state_measure_firstpassage_noortho_d128|1"
)
# tag|config|seed|stage-1 parent run dir
S2=(
  "p2_cd_d128_s0|cube_double_state_measure_ballvisit_d128|0|$FP/GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d128/p1_cd_d128_s0"
  "p2_cd_d128_s1|cube_double_state_measure_ballvisit_d128|1|$FP/GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d128/p1_cd_d128_s1"
  "p2_sc_d128_s0|scene_state_measure_ballvisit_d128|0|$FP/GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d128/p1_sc_d128_s0"
  "p2_sc_d128_s1|scene_state_measure_ballvisit_d128|1|$FP/GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d128/p1_sc_d128_s1"
)

n=0
launch() { # tag cfg seed [warm_src]
  local tag=$1 cfg=$2 seed=$3 src=${4:-}
  if compgen -G "$FP/*/$tag" > /dev/null; then
    echo "[d128] skip $tag - output dir already exists"; return 0
  fi
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[d128] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 60
  done
  local g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1))
  local WARMARGS=()
  [ -n "$src" ] && WARMARGS=("+agent_overrides.warm_start_ckpt=$src" "+agent_overrides.warm_start_epoch=$EPOCH")
  echo "[d128] $(date +%H:%M:%S) launch $tag cfg=$cfg seed=$seed on GPU$g ${src:+warm<-$src}"
  if [ "$DRY_RUN" = "1" ]; then echo "      DRY: --config-name $cfg seed=$seed"; return 0; fi
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      "${WARMARGS[@]}" output_root="$OUTROOT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
}

precheck() { # tag cfg seed src -> 0 ok / 1 fail
  local tag=$1 cfg=$2 seed=$3 src=$4
  [ "$SKIP_PRECHECK" = "1" ] && return 0
  [ "$DRY_RUN" = "1" ] && { echo "      DRY: precheck $tag"; return 0; }
  echo "[d128] $(date +%H:%M:%S) splice pre-check $tag (300 steps, d128 warm-start)"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=${GPUS[0]} \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      train_steps=300 log_interval=100 eval_interval=1000000 save_interval=1000000 \
      "+agent_overrides.warm_start_ckpt=$src" "+agent_overrides.warm_start_epoch=$EPOCH" \
      output_root=/dev/shm/d128_precheck exp_name="pre_$tag" wandb_mode=disabled \
      > "$LOG/precheck_$tag.log" 2>&1
  if grep -q "step 300" "$LOG/precheck_$tag.log" 2>/dev/null || tail -5 "$LOG/precheck_$tag.log" | grep -qi "done\|finished\|total_time"; then
    echo "[d128] precheck OK $tag"; return 0
  fi
  echo "[d128] precheck FAIL $tag -- NOT launching the 500k run. Tail:"; tail -20 "$LOG/precheck_$tag.log"; return 1
}

if [ "$STAGE" != "2" ]; then
  echo "[d128] $(date +%H:%M:%S) STAGE 1 -- ${#S1[@]} cold d128 phase-1 cells"
  for j in "${S1[@]}"; do IFS='|' read -r tag cfg seed <<<"$j"; launch "$tag" "$cfg" "$seed"; done
fi

if [ "$STAGE" = "1" ]; then wait; echo "[d128] stage 1 dispatched, stopping (STAGE=1)"; exit 0; fi

echo "[d128] $(date +%H:%M:%S) STAGE 2 -- waiting on stage-1 checkpoints"
for j in "${S2[@]}"; do
  IFS='|' read -r tag cfg seed src <<<"$j"
  if compgen -G "$FP/*/$tag" > /dev/null; then echo "[d128] skip $tag - exists"; continue; fi
  waited=0
  while [ ! -f "$src/params_$EPOCH.pkl" ]; do
    if [ "$DRY_RUN" = "1" ]; then echo "      DRY: would wait for $src/params_$EPOCH.pkl"; break; fi
    if [ $((waited % 1800)) -eq 0 ]; then
      echo "[d128] $(date +%H:%M:%S) waiting for $(basename "$src") (${waited}s elapsed)"
    fi
    sleep 300; waited=$((waited+300))
    if [ $waited -gt 172800 ]; then echo "[d128] FATAL $tag - stage-1 parent never finished (48h)"; exit 1; fi
  done
  if precheck "$tag" "$cfg" "$seed" "$src"; then
    launch "$tag" "$cfg" "$seed" "$src"
  else
    echo "[d128] SKIPPING $tag after failed pre-check"
  fi
done
wait
echo "[d128] $(date +%H:%M:%S) both stages complete"
