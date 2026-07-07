#!/usr/bin/env bash
# HIQL-bilinear mechanism cells (spec 2026-07-07): E-B rep50 / E-C schaul / E-D highbil
# (native + flow-both), 3 seeds @500k = 12 jax runs. ALWAYS in tmux:
#   tmux new -s hiqlmech 'DRY_RUN=0 scripts/launch_hiql_mech.sh 2>&1 | tee /dev/shm/hiqlmech_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(pwd)/.venv/bin/python}
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
SEEDS=(${SEEDS:-0 1 2})
LOGDIR=/dev/shm/hiqlmech_logs
mkdir -p "$LOGDIR"

configs=(
  cube_single_state_hiql_native_bilinear_d50_rep50_sg10
  cube_single_state_hiql_native_bilinear_d50_schaul_sg10
  cube_single_state_hiql_native_highbil_d50_sg10
  cube_single_state_hiql_flow_both_highbil_d50_sg10
)

JOBS=()
for cfg in "${configs[@]}"; do
  for s in "${SEEDS[@]}"; do
    JOBS+=("${cfg}__s${s}|||--config-name $cfg seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=${cfg}_s${s}")
  done
done

GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
echo "HIQL mechanism cells: ${#JOBS[@]} runs @${STEPS}"
if [ "$DRY_RUN" = "1" ]; then
  i=0; for j in "${JOBS[@]}"; do echo "GPU${GPUS[$((i % ${#GPUS[@]}))]}  ${j%%|||*}"; i=$((i+1)); done
  exit 0
fi

i=0
for j in "${JOBS[@]}"; do
  tag=${j%%|||*}; args=${j#*|||}
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  echo "[launch] GPU$g  $tag"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    $PY run_gciql.py $args >"$LOGDIR/${tag}.log" 2>&1 &
  i=$((i+1))
  sleep 5
done
echo "Launched ${#JOBS[@]} jax runs -> $LOGDIR ; waiting..."
wait
echo "ALL DONE"
