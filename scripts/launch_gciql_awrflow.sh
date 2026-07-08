#!/usr/bin/env bash
# {AWR-flow x GCIQL} readout-matrix cells: cube + antmaze, bilinear-d50, 3 seeds @500k.
# Defer-launches: waits until the box drains (run_gciql procs <= MAX_PROCS) so it
# doesn't pile onto bothbil/ambc03/mdose. ALWAYS in tmux:
#   tmux new -s awrflow 'DRY_RUN=0 scripts/launch_gciql_awrflow.sh 2>&1 | tee /dev/shm/awrflow_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(pwd)/.venv/bin/python}
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
MAX_PROCS=${MAX_PROCS:-18}
LOGDIR=/dev/shm/awrflow_logs
mkdir -p "$LOGDIR"

configs=(
  cube_single_state_gciql_awrflow_bilinear_d50
  antmaze_medium_gciql_awrflow_bilinear_d50
)

JOBS=()
for cfg in "${configs[@]}"; do
  for s in 0 1 2; do
    JOBS+=("${cfg}__s${s}|||--config-name $cfg seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=${cfg}_s${s}")
  done
done

GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
echo "GCIQL awr_flow cells: ${#JOBS[@]} runs @${STEPS} (defer gate: procs <= $MAX_PROCS)"
if [ "$DRY_RUN" = "1" ]; then
  i=0; for j in "${JOBS[@]}"; do echo "GPU${GPUS[$((i % ${#GPUS[@]}))]}  ${j%%|||*}"; i=$((i+1)); done
  exit 0
fi

while [ "$(pgrep -fc run_gciql.py)" -gt "$MAX_PROCS" ]; do
  echo "[defer] $(date +%H:%M) box busy ($(pgrep -fc run_gciql.py) procs > $MAX_PROCS), waiting..."
  sleep 300
done

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
