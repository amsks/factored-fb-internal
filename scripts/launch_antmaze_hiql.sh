#!/usr/bin/env bash
# OGBench-SOTA grid 1b: antmaze-medium HIQL, MLP vs bilinear-d50, 8 seeds @1M.
# HPs = OGBench tuned antmaze-medium-navigate HIQL (= agent defaults incl.
# subgoal_steps=25; hyperparameters.sh:116 only sets the default alphas).
# Protocol per 2026-07-07 spec: 8 seeds @1M, eval every 100k (avg last 3 offline).
# Packed 4/GPU on the free GPUs with XLA_PYTHON_CLIENT_PREALLOCATE=false. ALWAYS in tmux:
#   tmux new -s amhiql 'DRY_RUN=0 scripts/launch_antmaze_hiql.sh 2>&1 | tee /dev/shm/amhiql_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(pwd)/.venv/bin/python}
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-1000000}
LOGDIR=/dev/shm/amhiql_logs
mkdir -p "$LOGDIR"

configs=(
  antmaze_medium_hiql_mlp
  antmaze_medium_hiql_bilinear_d50
)

JOBS=()
for cfg in "${configs[@]}"; do
  for s in 0 1 2 3 4 5 6 7; do
    JOBS+=("${cfg}__s${s}|||--config-name $cfg seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=${cfg}_s${s}")
  done
done

GPUS=(${GPUS:-3 5 6 7})
echo "AntMaze-medium HIQL: ${#JOBS[@]} runs @${STEPS}"
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
