#!/usr/bin/env bash
# Spec 28 P1 (PAPER/Multistep-FB/specs/28-z-interface-canonicalization.md section P1):
# the conditioning x actor-reward 2x2, cube-single 500k seed 0 screen of the two NEW
# cells. Cells (i) and (iv) already exist (cs .93 / .111), so only these two run.
#
#   ii   z(g) conditioning + kg (exact, query-set normalized)   THE DECISIVE CELL
#   iii  g   conditioning + B̃^T z(g)/sqrt(d)                    readout validity alone
#
# ---------------------------------------------------------------------------------
# GATE -- DO NOT RUN THIS SCRIPT YET.
# P1 is gated on the P0a action-ranking probe. If P0a returns "conditioning indicted"
# (q_g action-value span ~ 0 under z-conditioning), spec 28 pre-registers that cell (ii)
# is SKIPPED, not run -- the compute would buy a foregone conclusion. Only dispatch once
# P0a has read out and the gate is recorded. Until then: `DRY_RUN=1` only.
# ---------------------------------------------------------------------------------
#
# Single-variable deltas: both configs differ from their anchors by exactly one knob
# (measure_density_z_actor_reward=kernel on the spec-24 z lattice; and
# measure_density_actor_reward=latent on the spec-18 g lattice). measure_density_z_rand_frac
# stays 0.5 -- spec 28's tie-break cell ii' (rand_frac=0) is a SEPARATE, later cell and is
# deliberately not in this wave.
#
# Modeled on launch_spec25_stage1.sh. Concurrency: active() counts EVERY main.py job on
# the box (not just this wave's), so this dispatcher cooperates with whatever else is
# running rather than oversubscribing it. Do NOT kill this dispatcher to change MAX or
# anything else: it is the pipeline parent of its children and SIGTERM/SIGINT propagates
# to them -- that has killed live cells here before. DRY_RUN is checked BEFORE the
# wait-for-a-free-slot loop, not after, so `DRY_RUN=1` on a full box still prints and
# exits promptly instead of blocking.
#
# Run in tmux (ONLY after the P0a gate resolves):
#   tmux new -d -s spec28p1 'scripts/rescuing_fb/launch_spec28_p1.sh 2>&1 | tee -a /dev/shm/spec28_p1_logs/dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_spec28_p1.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec28_p1}
LOG=/dev/shm/spec28_p1_logs; mkdir -p "$LOG" "$OUTROOT"
DRY_RUN=${DRY_RUN:-0}
MAX=${MAX:-8}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
FP=$OUTROOT/factored-fb
export GCIQL_WANDB_ENTITY=${GCIQL_WANDB_ENTITY:-amsks}

active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }

JOBS=(
  "s28_cs_ii_kernelrew|cube_single_state_measure_density_z_kernelrew_d50|0"
  "s28_cs_iii_latentrew|cube_single_state_measure_density_latentrew_d50|0"
)

for j in "${JOBS[@]}"; do
  IFS='|' read -r _t cfg _s <<<"$j"
  [ -f "configs/gciql/$cfg.yaml" ] || { echo "[s28p1] FATAL missing config configs/gciql/$cfg.yaml"; exit 1; }
done

echo "[s28p1] $(date +%H:%M:%S) ${#JOBS[@]} cells, MAX=$MAX (shared with any other wave), entity=$GCIQL_WANDB_ENTITY"
n=0; launched=0; done_already=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed <<<"$j"
  if compgen -G "$FP/*/$tag" > /dev/null; then
    echo "[s28p1] skip $tag - output dir already exists"; done_already=$((done_already+1)); continue
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "[s28p1] DRY: --config-name $cfg seed=$seed output_root=$OUTROOT exp_name=$tag"; continue
  fi
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[s28p1] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 120
  done
  g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1)); launched=$((launched+1))
  echo "[s28p1] $(date +%H:%M:%S) launch $tag cfg=$cfg seed=$seed on GPU$g"
  XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      output_root="$OUTROOT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[s28p1] $(date +%H:%M:%S) launched=$launched already_present=$done_already"
