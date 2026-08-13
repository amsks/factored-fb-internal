#!/bin/bash
# Spec 10 campaign v2 (2026-07-27): two trained arms against the SMB residuals.
#   Arm A  measure_eps_multiscale='0.2,0.4,0.8'  -> B2 boxes + T3 R-resolution
#   Arm B  actor_z_greedy_frac=0.3               -> T6 margin + T5 avoidance depth
# Warm-started from the BallVisit-ZX phase-2 ckpts (500k), 300k continuation steps.
# 4 jobs, fixed GPUs 0/3/4/7 (1/5/6 = other session, never touch). Mirrors
# launch_spec08_e1.sh conventions (same config, same wandb shim behavior).
set -u
cd /home/adityamohan/git/Factored-FB
PY=$(pwd)/.venv/bin/python
OUTROOT=/home/adityamohan/exp_spec10
LOG=/dev/shm/spec10_logs; mkdir -p "$LOG" "$OUTROOT"
CFG=cube_single_state_measure_ballvisit_zx_d50
VISZ=/home/adityamohan/exp_spec08/factored-fb/GCIQL-State-Flow-Measure-BallVisit-ZX-d50
WS="+agent_overrides.warm_start_epoch=500000 train_steps=300000"

# tag|seed|gpu|extra
JOBS=(
  "msA_s0|0|0|+agent_overrides.warm_start_ckpt=$VISZ/visz_s0 $WS +agent_overrides.measure_eps_multiscale='0.2,0.4,0.8' run_group=GCIQL-State-Flow-Measure-BallVisit-ZX-MultiEps-d50"
  "msA_s1|1|3|+agent_overrides.warm_start_ckpt=$VISZ/visz_s1 $WS +agent_overrides.measure_eps_multiscale='0.2,0.4,0.8' run_group=GCIQL-State-Flow-Measure-BallVisit-ZX-MultiEps-d50"
  "azB_s0|0|4|+agent_overrides.warm_start_ckpt=$VISZ/visz_s0 $WS +agent_overrides.actor_z_greedy_frac=0.3 run_group=GCIQL-State-Flow-Measure-BallVisit-ZX-ActorZ-d50"
  "azB_s1|1|7|+agent_overrides.warm_start_ckpt=$VISZ/visz_s1 $WS +agent_overrides.actor_z_greedy_frac=0.3 run_group=GCIQL-State-Flow-Measure-BallVisit-ZX-ActorZ-d50"
)

for j in "${JOBS[@]}"; do
  IFS='|' read -r tag seed g extra <<<"$j"
  echo "[v2] $(date +%H:%M:%S) launch $tag on GPU$g"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$CFG" seed="$seed" \
      output_root="$OUTROOT" exp_name="$tag" $extra \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[v2] all campaign-v2 jobs finished"
