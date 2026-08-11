#!/usr/bin/env bash
# Density + DDPG+BC actor on cube-double and scene, soft and hard (4 cells x 3 seeds = 12).
#
# WHY: with the flow actor dropped, the density claim currently rests on ONE manipulation
# environment (cube-single, .879 vs published best 68). Spec 22 showed the cube-single
# headline is not flow-carried -- density loses .05 under the actor swap while stock GCIQL
# loses .31 -- but cube-double and scene were never run with a stock actor. These cells are
# what let the OGBench comparison be made without the flow actor in it at all.
#
# alpha=1.0 is OGBench's own GCIQL value for both envs (hyperparameters.sh:617/:708).
# Cooperative concurrency: active() counts EVERY main.py job, so this dispatcher idles
# rather than oversubscribing alongside other waves. Do NOT kill it to change MAX -- the
# signal propagates to its children.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_ddpgbcx}
LOG=/dev/shm/ddpgbcx_logs; mkdir -p "$LOG" "$OUTROOT"
DRY_RUN=${DRY_RUN:-0}; MAX=${MAX:-8}; GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); FP=$OUTROOT/factored-fb
active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }
JOBS=()
for cell in "cd_soft|cube_double_state_measure_density_d50_ddpgbc" \
            "sc_soft|scene_state_measure_density_d50_ddpgbc" \
            "cd_hard|cube_double_state_measure_density_hard_d50_ddpgbc" \
            "sc_hard|scene_state_measure_density_hard_d50_ddpgbc"; do
  IFS='|' read -r nm cfg <<<"$cell"
  [ -f "configs/gciql/$cfg.yaml" ] || { echo "[dbx] FATAL missing configs/gciql/$cfg.yaml"; exit 1; }
  for s in 0 1 2; do JOBS+=("dbx_${nm}_s${s}|$cfg|$s"); done
done
echo "[dbx] $(date +%H:%M:%S) ${#JOBS[@]} cells, MAX=$MAX (shared)"
n=0; launched=0; skipped=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed <<<"$j"
  if compgen -G "$FP/*/$tag" > /dev/null; then echo "[dbx] skip $tag (exists)"; skipped=$((skipped+1)); continue; fi
  # DRY_RUN must short-circuit BEFORE the concurrency wait: otherwise a dry run on a full
  # box blocks forever instead of printing the plan (cost me a 10-minute timeout once).
  if [ "$DRY_RUN" = "1" ]; then
    g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1)); launched=$((launched+1))
    echo "      DRY: $tag cfg=$cfg seed=$seed -> GPU$g"; continue
  fi
  while [ "$(active)" -ge "$MAX" ]; do echo "[dbx] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting for $tag"; sleep 120; done
  g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1)); launched=$((launched+1))
  echo "[dbx] $(date +%H:%M:%S) launch $tag cfg=$cfg seed=$seed on GPU$g"
  XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" output_root="$OUTROOT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[dbx] $(date +%H:%M:%S) launched=$launched skipped=$skipped"
