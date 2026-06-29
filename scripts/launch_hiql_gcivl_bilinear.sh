#!/usr/bin/env bash
# HIQL + GCIVL bilinear-d50 @500k x 3 seeds (6 jax runs), cube-single state.
# Runs alongside the FB knob ladder. XLA preallocate=false so they don't grab a whole GPU.
# ALWAYS in tmux:  tmux new -s hiqlbil 'DRY_RUN=0 scripts/launch_hiql_gcivl_bilinear.sh 2>&1 | tee /dev/shm/hiqlbil_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(pwd)/.venv/bin/python}
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
LOGDIR=/dev/shm/hiql_gcivl_bil_logs
mkdir -p "$LOGDIR"

configs=( cube_single_state_hiql_bilinear_d50 cube_single_state_gcivl_bilinear_d50 )

JOBS=()
for cfg in "${configs[@]}"; do
  for s in 0 1 2; do
    JOBS+=("${cfg}__s${s}|||--config-name $cfg seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=${cfg}_s${s}")
  done
done

# pack onto GPUs 2..7 (one each) to leave 0,1 lighter for the FB torch ladder
GPUS=(2 3 4 5 6 7)
echo "HIQL/GCIVL bilinear: ${#JOBS[@]} runs @${STEPS}"
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
