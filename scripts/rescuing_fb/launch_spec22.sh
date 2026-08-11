#!/usr/bin/env bash
# Spec 22 section 7 (PAPER/Multistep-FB/specs/22-antmaze-actor-ablation.md) -- density on
# locomotion + the actor ablation. 11 jobs.
#
# TWO loaded questions, both pre-registered in spec 22 section 1 BEFORE any number landed:
#  1. Does the normalized successor density transfer to antmaze-medium, a domain with NO
#     dwell asymmetry (the agent IS the object -- there is no freely-moving/held/slipping
#     interaction variable for the measure to resolve)? density ~= stock GCIQL is the
#     EXPECTED NULL there, not a failure.
#  2. Actor attribution, both directions. On antmaze the FLOW actor is a known killer
#     independent of the critic (stock ddpgbc .837 vs flow .383 at bc0.3), so a
#     density+flow failure there must NOT be read as a measure failure. And on cube the
#     flow actor BEATS ddpgbc (.870 vs .563), so our headline density .929 may itself be
#     actor-carried -- the cube ddpgbc/awr mirror cells test exactly that.
#
# Reused controls -- do NOT relaunch, quote from the CSVs (spec 22 section 7):
#   antmaze gciql+ddpgbc .837/3s (summary_agents.csv:64) | +flow .383/3s bc0.3 (:67)
#           | +awr_flow .617/3s (:63)
#   cube    gciql+flow .870 (:14) | +ddpgbc bilinear .563 (:9) | density+flow .929/3s
#           (bridge_grafts.csv:97)
#
# Gates: G1 (locomotion viability) density+ddpgbc mean >= 0.70 over 3 seeds, bracketed by
# stock ddpgbc .837 and L2 measure-only+ddpgbc .784. Remaining gates in spec 22 section 7.
#
# Run in tmux:
#   tmux new -d -s s22am 'scripts/rescuing_fb/launch_spec22.sh 2>&1 | tee -a /dev/shm/spec22_logs/dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_spec22.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec22}
LOG=/dev/shm/spec22_logs; mkdir -p "$LOG" "$OUTROOT"
DRY_RUN=${DRY_RUN:-0}
# Other waves hold GPUs; MAX gates concurrency and the loop queues the rest.
MAX=${MAX:-5}
GPUS=(${GPUS:-0 4 5 6 7})
FP=$OUTROOT/factored-fb

active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }

# tag|config|seed
JOBS=(
  # antmaze-medium: the locomotion question. flow + ddpgbc get 3 seeds (discriminating),
  # awr_flow 1 (screen).
  "a22_am_dflow_s0|antmaze_medium_state_measure_density_d50|0"
  "a22_am_dflow_s1|antmaze_medium_state_measure_density_d50|1"
  "a22_am_dflow_s2|antmaze_medium_state_measure_density_d50|2"
  "a22_am_dddpg_s0|antmaze_medium_state_measure_density_d50_ddpgbc|0"
  "a22_am_dddpg_s1|antmaze_medium_state_measure_density_d50_ddpgbc|1"
  "a22_am_dddpg_s2|antmaze_medium_state_measure_density_d50_ddpgbc|2"
  "a22_am_dawr_s0|antmaze_medium_state_measure_density_d50_awrflow|0"

  # cube-single mirror: is our .929 headline actor-carried?
  "a22_cs_dddpg_s0|cube_single_state_measure_density_d50_ddpgbc|0"
  "a22_cs_dddpg_s1|cube_single_state_measure_density_d50_ddpgbc|1"
  "a22_cs_dddpg_s2|cube_single_state_measure_density_d50_ddpgbc|2"
  "a22_cs_dawr_s0|cube_single_state_measure_density_d50_awrflow|0"
)

for j in "${JOBS[@]}"; do
  IFS='|' read -r _t cfg _s <<<"$j"
  [ -f "configs/gciql/$cfg.yaml" ] || { echo "[s22] FATAL missing config configs/gciql/$cfg.yaml"; exit 1; }
done

echo "[s22] $(date +%H:%M:%S) ${#JOBS[@]} cells, MAX=$MAX, GPUS=${GPUS[*]}"
n=0; launched=0; done_already=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed <<<"$j"
  if compgen -G "$FP/*/$tag" > /dev/null; then
    echo "[s22] skip $tag - output dir already exists"; done_already=$((done_already+1)); continue
  fi
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[s22] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 120
  done
  g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1)); launched=$((launched+1))
  echo "[s22] $(date +%H:%M:%S) launch $tag cfg=$cfg seed=$seed on GPU$g"
  if [ "$DRY_RUN" = "1" ]; then
    echo "      DRY: --config-name $cfg seed=$seed output_root=$OUTROOT exp_name=$tag"; continue
  fi
  # XLA CUDA-graph capture disabled (the d17_cs_part_s1 stream-capture crashes).
  XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      output_root="$OUTROOT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[s22] $(date +%H:%M:%S) launched=$launched already_present=$done_already"
