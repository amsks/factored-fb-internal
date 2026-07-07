#!/usr/bin/env bash
# v4 both-bilinear HIQL cells (follow-up to hiql-bilinear mechanism spec):
# Schaul bilinear main V (TD+low) + stop-grad Schaul bilinear high head,
# goal_rep trained by the low actor. Native + flow-both, 3 seeds @500k.
# ALWAYS in tmux:
#   tmux new -s bothbil 'DRY_RUN=0 scripts/launch_hiql_bothbil.sh 2>&1 | tee /dev/shm/bothbil_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(pwd)/.venv/bin/python}
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
LOGDIR=/dev/shm/bothbil_logs
mkdir -p "$LOGDIR"

configs=(
  cube_single_state_hiql_native_bothbil_d50_sg10
  cube_single_state_hiql_flow_both_bothbil_d50_sg10
)

JOBS=()
for cfg in "${configs[@]}"; do
  for s in 0 1 2; do
    JOBS+=("${cfg}__s${s}|||--config-name $cfg seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=${cfg}_s${s}")
  done
done

GPUS=(${GPUS:-0 1 2 3 4 5})
echo "HIQL v4 both-bilinear: ${#JOBS[@]} runs @${STEPS}"
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
