#!/usr/bin/env bash
# Spec 25 stage 1 (PAPER/Multistep-FB/specs/25-z-coupling-port.md section 3): cube-single
# screens for the three index-coupling ports spec 24's z-indexed density measure lacked --
# C1 structural logit tilt, C2 ortho-free reward-decodability aux, C3 index-consistent actor
# supervision (both knobs) -- plus the C1+C2+C3 stack. 1 seed each, 500k, round-1 doses fixed
# by the spec (beta=1.0, rdecode_coef=1.0, enc_tau=0.0005, bc_hindsight on), no sweeps.
#
# Screen gate S1: density_w_ztv bounded away from 0 at 500k AND E1 >= 0.5 (g-anchor ~.93).
# Pre-registered NULL: all four flat on density_w_ztv -> stop the z program, write the
# reward-embedding actor spec instead (section 3).
#
# Modeled on launch_hard_xenv.sh. Concurrency: active() counts EVERY main.py job on the box
# (not just this wave's), so this dispatcher cooperates with whatever else is running rather
# than oversubscribing it. Do NOT kill this dispatcher to change MAX or anything else: it is
# the pipeline parent of its children and SIGTERM/SIGINT propagates to them -- that has
# killed live cells here before. DRY_RUN is checked BEFORE the wait-for-a-free-slot loop, not
# after, so `DRY_RUN=1` on a full box still prints and exits promptly instead of blocking.
#
# The four configs' agent_overrides use knob names pinned verbatim in spec 25 section 0
# (measure_density_z_tilt, measure_density_z_rdecode_coef, measure_density_z_bc_hindsight,
# measure_density_z_enc_tau) -- these do not exist in tools/wandb_mode_shim/gciql_flow.py yet
# (spec 25's implementation is being written concurrently). Until that lands, real (non-dry)
# launches will fail fast in the agent's own config validation; this script only authors and
# dry-run-validates the wave, it does not gate on the implementation landing.
#
# Run in tmux:
#   tmux new -d -s spec25s1 'scripts/rescuing_fb/launch_spec25_stage1.sh 2>&1 | tee -a /dev/shm/spec25_stage1_logs/dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_spec25_stage1.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec25_stage1}
LOG=/dev/shm/spec25_stage1_logs; mkdir -p "$LOG" "$OUTROOT"
DRY_RUN=${DRY_RUN:-0}
MAX=${MAX:-8}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
FP=$OUTROOT/factored-fb

active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }

JOBS=(
  "z25_cs_tilt|cube_single_state_measure_density_z_tilt_d50|0"
  "z25_cs_actor|cube_single_state_measure_density_z_actor_d50|0"
  "z25_cs_dec|cube_single_state_measure_density_z_dec_d50|0"
  "z25_cs_full|cube_single_state_measure_density_z_full_d50|0"
)

for j in "${JOBS[@]}"; do
  IFS='|' read -r _t cfg _s <<<"$j"
  [ -f "configs/gciql/$cfg.yaml" ] || { echo "[z25s1] FATAL missing config configs/gciql/$cfg.yaml"; exit 1; }
done

echo "[z25s1] $(date +%H:%M:%S) ${#JOBS[@]} cells, MAX=$MAX (shared with any other wave)"
n=0; launched=0; done_already=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed <<<"$j"
  if compgen -G "$FP/*/$tag" > /dev/null; then
    echo "[z25s1] skip $tag - output dir already exists"; done_already=$((done_already+1)); continue
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "[z25s1] DRY: --config-name $cfg seed=$seed output_root=$OUTROOT exp_name=$tag"; continue
  fi
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[z25s1] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 120
  done
  g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1)); launched=$((launched+1))
  echo "[z25s1] $(date +%H:%M:%S) launch $tag cfg=$cfg seed=$seed on GPU$g"
  XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      output_root="$OUTROOT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[z25s1] $(date +%H:%M:%S) launched=$launched already_present=$done_already"
