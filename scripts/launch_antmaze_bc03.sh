#!/usr/bin/env bash
# Antmaze flow-cell reruns (td_jepa HP audit 2026-07-07): GCIQL flow cells at the
# domain-correct bc_coeff=0.3 (bilinear + MLP control) + first-ever antmaze runs of
# GCIVL-flow-bilinear and HIQL-v3-flow (AWR-flow agents, tuned alphas = defaults).
# 4 cells x 3 seeds @500k. ALWAYS in tmux:
#   tmux new -s ambc03 'DRY_RUN=0 scripts/launch_antmaze_bc03.sh 2>&1 | tee /dev/shm/ambc03_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(pwd)/.venv/bin/python}
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
LOGDIR=/dev/shm/ambc03_logs
mkdir -p "$LOGDIR"

configs=(
  antmaze_medium_gciql_flow_bilinear_d50_bc03
  antmaze_medium_gciql_flow_mlp_bc03
  antmaze_medium_gcivl_flow_bilinear_d50
  antmaze_medium_hiql_flow_both_highbil_d50
)

JOBS=()
for cfg in "${configs[@]}"; do
  for s in 0 1 2; do
    JOBS+=("${cfg}__s${s}|||--config-name $cfg seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=${cfg}_s${s}")
  done
done

GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
echo "Antmaze bc03/coverage cells: ${#JOBS[@]} runs @${STEPS}"
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
