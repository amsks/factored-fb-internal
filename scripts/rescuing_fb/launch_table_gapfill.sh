#!/usr/bin/env bash
# Gap-fill for the env x actor x method table: the HARD cells that do not exist yet.
#   cube-single hard+ddpgbc  (3 seeds)  -- completes the cube-single row
#   antmaze     hard+flow    (3 seeds)  -- completes the antmaze row
#   antmaze     hard+ddpgbc  (3 seeds)  --   "
# Everything else the table needs is either collected or queued by launch_density_ddpgbc_xenv.sh.
#
# NOT covered here (torch stack, separate decision): vanilla FB + ddpgbc on cube-double,
# scene and antmaze, and FB zero-shot on every env -- no torch FB checkpoints are on disk.
#
# Cooperative: active() counts EVERY main.py job. DRY_RUN short-circuits before the wait.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_gapfill}; LOG=/dev/shm/gapfill_logs
mkdir -p "$LOG" "$OUTROOT"; DRY_RUN=${DRY_RUN:-0}; MAX=${MAX:-8}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); FP=$OUTROOT/factored-fb
active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }
JOBS=()
for cell in "cs_hard_ddpg|cube_single_state_measure_density_hard_d50_ddpgbc" \
            "am_hard_flow|antmaze_medium_state_measure_density_hard_d50" \
            "am_hard_ddpg|antmaze_medium_state_measure_density_hard_d50_ddpgbc"; do
  IFS='|' read -r nm cfg <<<"$cell"
  [ -f "configs/gciql/$cfg.yaml" ] || { echo "[gap] FATAL missing $cfg"; exit 1; }
  for s in 0 1 2; do JOBS+=("gap_${nm}_s${s}|$cfg|$s"); done
done
echo "[gap] $(date +%H:%M:%S) ${#JOBS[@]} cells, MAX=$MAX (shared)"
n=0; launched=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed <<<"$j"
  compgen -G "$FP/*/$tag" >/dev/null && { echo "[gap] skip $tag (exists)"; continue; }
  if [ "$DRY_RUN" = "1" ]; then echo "      DRY: $tag cfg=$cfg seed=$seed"; continue; fi
  while [ "$(active)" -ge "$MAX" ]; do echo "[gap] $(date +%H:%M:%S) active=$(active) waiting for $tag"; sleep 180; done
  g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1)); launched=$((launched+1))
  echo "[gap] $(date +%H:%M:%S) launch $tag on GPU$g"
  XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" output_root="$OUTROOT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait; echo "[gap] $(date +%H:%M:%S) launched=$launched"
