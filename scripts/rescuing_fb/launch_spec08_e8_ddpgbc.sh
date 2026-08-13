#!/usr/bin/env bash
# Spec 08 E8 -- ACTOR-ISOLATION arm on cube-single: does the reach-first objective survive
# the stock GCIQL gaussian DDPG+BC actor, or does it need the flow-BC actor?
#
# Everything in the note shares ONE actor class, so its comparisons isolate the value
# objective. The cost is that it cannot say whether the reaching value needs that actor.
# This arm swaps only actor_kind, on both sides of the comparison:
#
#   ours   cube_single_state_measure_firstpassage_noortho_d50_ddpgbc   (phase 1 + ddpgbc)
#   anchor cube_single_state_gciql_flow_bilinear_d50_ddpgbc            (GC-IQL + ddpgbc)
#
# Flow-actor reference on the same env/budget: phase 1 0.967+-0.012, anchor 0.928+-0.014.
#
# STAGE 1 (this script): alpha sweep at 1 seed per arm. The ddpgbc BC weight is `alpha`
# (NOT bc_coeff, which is the flow branch's knob and is inert here). OGBench's default is
# 0.3 and its README warns DDPG+BC is more alpha-sensitive than AWR, so a single-alpha run
# could floor for tuning reasons alone -- exactly the ambiguity that makes our puzzle-3x3
# anchor uncitable. 3 alphas x 2 arms = 6 jobs.
# STAGE 2 (manual, after reading stage 1): top up the winning alpha to 3 seeds per arm.
#
# Run in tmux:
#   tmux new -d -s s08e8d 'scripts/rescuing_fb/launch_spec08_e8_ddpgbc.sh 2>&1 | tee -a /dev/shm/spec08_logs/e8d_dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_spec08_e8_ddpgbc.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec08}
LOG=/dev/shm/spec08_logs/e8d; mkdir -p "$LOG" "$OUTROOT"
DRY_RUN=${DRY_RUN:-0}
MAX=${MAX:-14}
GPUS=(0 1 2 3 4 5 6 7)
SEEDS=${SEEDS:-0}
ALPHAS=${ALPHAS:-"0.3 1.0 3.0"}

active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }
pick_gpu() { echo "${GPUS[$(( $1 % ${#GPUS[@]} ))]}"; }

# label|config
ARMS=(
  "m|cube_single_state_measure_firstpassage_noortho_d50_ddpgbc"
  "a|cube_single_state_gciql_flow_bilinear_d50_ddpgbc"
)

n=0
for arm in "${ARMS[@]}"; do
  IFS='|' read -r lab cfg <<<"$arm"
  for alpha in $ALPHAS; do
    for seed in $SEEDS; do
      tag="db_${lab}_a${alpha/./}_s${seed}"
      if compgen -G "$OUTROOT/factored-fb/*/$tag" > /dev/null; then
        echo "[e8d] skip $tag - output dir exists"; continue
      fi
      while [ "$(active)" -ge "$MAX" ]; do
        echo "[e8d] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 60
      done
      g=$(pick_gpu "$n"); n=$((n+1))
      echo "[e8d] $(date +%H:%M:%S) launch $tag cfg=$cfg alpha=$alpha seed=$seed on GPU$g"
      if [ "$DRY_RUN" = "1" ]; then
        echo "      DRY: --config-name $cfg seed=$seed +agent_overrides.alpha=$alpha exp_name=$tag"
        continue
      fi
      XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
        "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
          "+agent_overrides.alpha=$alpha" \
          train_steps=500000 eval_interval=50000 save_interval=100000 \
          output_root="$OUTROOT" exp_name="$tag" \
          > "$LOG/$tag.log" 2>&1 &
      sleep 45
    done
  done
done
wait
echo "[e8d] $(date +%H:%M:%S) ALL DDPGBC JOBS EXITED"
