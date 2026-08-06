#!/usr/bin/env bash
# Spec 19 section 1 P1.1 (PAPER/Multistep-FB/specs/19-ors-program.md,
# contract 19a-behavior-density-impl.md) -- behavior-policy density training.
#
# What this produces: the frozen (F_D, B_D) towers of the occupancy model
# d^{pi_D}(.|s,a) -- the dataset behavior policy's discounted future
# object-configuration distribution. That is the upstream artifact the ORS shaping
# reward r_W(s,a,g) = -W2^2(delta_g, d^{pi_D}(.|s,a)) is read out of (spec 19b).
#
# Two deltas from the spec-18 density arm, both load-bearing:
#   * NO goal conditioning -- the F tower sees [s, a]. The behavior policy ignores
#     goals; conditioning on g would only add estimation burden.
#   * SARSA bootstrap -- the bootstrap action is the DATASET's next action, not a
#     sample from the learned actor (which is goal-conditioned BC and would
#     therefore estimate a different policy's occupancy).
# The actor in this mode is pure BC flow and is NOT evaluated; the run's product is
# the density towers. Do not read its evaluation/overall_success as a method result.
#
# ORDERING (spec 19 section 1): this is P1.1. The ORS extractor wave (P1.3) is HARD-GATED
# behind P1.2 -- the calibration probe must reach Spearman >= 0.65 per env against the
# dataset's own empirical discounted future distance. Launching extractors on an
# uncalibrated shaping signal is explicitly refused by the spec.
#
# Seeds: 0 first per spec 19 section 1 (3 seeds on cube-single only after the P1.3 gate).
#
# Run in tmux:
#   tmux new -d -s s19p1 'scripts/rescuing_fb/launch_spec19_p1.sh 2>&1 | tee -a /dev/shm/spec19_logs/dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_spec19_p1.sh
# Envs:     ENVS="cs cd" (default; add sc for scene)
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec19}
LOG=/dev/shm/spec19_logs; mkdir -p "$LOG" "$OUTROOT"
DRY_RUN=${DRY_RUN:-0}
MAX=${MAX:-8}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
SEEDS=${SEEDS:-0}
ENVS=${ENVS:-"cs cd"}
FP=$OUTROOT/factored-fb

active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }

cfg_for() {
  case "$1" in
    cs) echo cube_single_state_measure_density_behavior_d50 ;;
    cd) echo cube_double_state_measure_density_behavior_d50 ;;
    sc) echo scene_state_measure_density_behavior_d50 ;;
    *)  echo "" ;;
  esac
}

JOBS=()
for e in $ENVS; do
  c=$(cfg_for "$e")
  if [ -z "$c" ]; then echo "[s19] FATAL unknown env '$e'"; exit 1; fi
  if [ ! -f "configs/gciql/$c.yaml" ]; then echo "[s19] FATAL missing config configs/gciql/$c.yaml"; exit 1; fi
  for s in $SEEDS; do JOBS+=("d19_${e}_bdens_s${s}|$c|$s"); done
done

echo "[s19] $(date +%H:%M:%S) ${#JOBS[@]} cells (P1.1 behavior density), envs='$ENVS' seeds='$SEEDS', MAX=$MAX"
n=0; launched=0; done_already=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed <<<"$j"
  if compgen -G "$FP/*/$tag" > /dev/null; then
    echo "[s19] skip $tag - output dir already exists"; done_already=$((done_already+1)); continue
  fi
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[s19] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 60
  done
  g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1)); launched=$((launched+1))
  echo "[s19] $(date +%H:%M:%S) launch $tag cfg=$cfg seed=$seed on GPU$g"
  if [ "$DRY_RUN" = "1" ]; then
    echo "      DRY: --config-name $cfg seed=$seed output_root=$OUTROOT exp_name=$tag"; continue
  fi
  # XLA CUDA-graph capture disabled: d17_cs_part_s1 died twice with
  # CUDA_ERROR_STREAM_CAPTURE_INVALIDATED from a device_get at logging time, on two
  # different GPUs. Costs a little speed; a crash mid-wave costs much more.
  XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      output_root="$OUTROOT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[s19] $(date +%H:%M:%S) launched=$launched already_present=$done_already"
