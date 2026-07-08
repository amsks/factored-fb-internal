#!/usr/bin/env bash
# Bridge mechanism follow-up to the E2 psi cross-eval finding (2026-07-07): the measure
# objective's psi geometry CONFLICTS with scalar TD's, visible only under stopgrad-phi
# (cos .893->.815, aucE .921->.810 at lambda=1); shared legs absorb it up to lambda=1.
# Two discriminators, 5 cells x 3 seeds = 15 jax runs @500k:
#   dose-response : MeasureMix sgphi lambda {3,10,30} (trace the degradation curve) +
#                   shared lambda 10 (where does absorption stop?)  [cube]
#   occupancy     : measure_only + DDPGBC actor on antmaze-medium (domain-appropriate
#                   actor; cube MeasureOnly=9.9 anchor, antmaze IQL-ddpgbc=0.88 anchor)
# Guardrail: critic/q_mean in [-100,10] (q-band; grad_norm is dose-loaded in sgphi arms).
# Packed round-robin ~2/GPU with XLA_PYTHON_CLIENT_PREALLOCATE=false. ALWAYS in tmux:
#   tmux new -s mdose 'DRY_RUN=0 scripts/launch_measure_dose_antmaze.sh 2>&1 | tee /dev/shm/mdose_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(pwd)/.venv/bin/python}
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
LOGDIR=/dev/shm/mdose_logs
mkdir -p "$LOGDIR"

configs=(
  cube_single_state_gciql_flow_bilinear_d50_measuremix_l3_sgphi
  cube_single_state_gciql_flow_bilinear_d50_measuremix_l10_sgphi
  cube_single_state_gciql_flow_bilinear_d50_measuremix_l30_sgphi
  cube_single_state_gciql_flow_bilinear_d50_measuremix_l10
  antmaze_medium_gciql_bilinear_d50_measureonly_ddpgbc
  cube_single_state_gciql_flow_bilinear_d50_C5_zmix
  cube_single_state_gciql_flow_bilinear_d50_C5_zmix_actorz
)

JOBS=()
for cfg in "${configs[@]}"; do
  for s in 0 1 2; do
    JOBS+=("${cfg}__s${s}|||--config-name $cfg seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=${cfg}_s${s}")
  done
done

GPUS=(0 1 2 3 4 5 6 7)
echo "measure dose+antmaze: ${#JOBS[@]} runs @${STEPS}"
if [ "$DRY_RUN" = "1" ]; then
  i=0; for j in "${JOBS[@]}"; do echo "GPU${GPUS[$((i % ${#GPUS[@]}))]}  ${j%%|||*}"; i=$((i+1)); done
  echo "(dry run; set DRY_RUN=0 to launch)"; exit 0
fi

i=0
for j in "${JOBS[@]}"; do
  tag=${j%%|||*}; args=${j#*|||}
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  echo "[launch] GPU$g  $tag"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    $PY run_gciql.py $args >"$LOGDIR/${tag}.log" 2>&1 &
  i=$((i+1))
  sleep 8
done
echo "Launched ${#JOBS[@]} jax runs -> $LOGDIR ; waiting..."
wait
echo "ALL DONE"
