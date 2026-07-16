#!/usr/bin/env bash
# Expressivity ablation of the bilinear bridge cell (2026-07-13).
#
#   stock bilinear (0.904):  Q(s,a,g) = phi(s,a) . psi(g)   / sqrt(d)
#   this cell:               Q(s,a,g) = phi(s,a) . psi(s,g) / sqrt(d)
#
# Same head, same rank d=50, same flow actor, same IQL objective; psi now sees the state,
# which destroys the low-rank GOAL bottleneck. Asks whether the bilinear win is the
# bottleneck or merely the inner-product head.
#
# 3 seeds x 500k. Anchors: bilinear-d50 = 0.904, MLP GCValue = 0.717.
#
# ALWAYS in tmux ([[always-run-in-tmux]]):
#   tmux new -d -s bilsg 'DRY_RUN=0 scripts/launch_bilinear_sg.sh 2>&1 | tee /dev/shm/bilsg_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(pwd)/.venv/bin/python}   # run_gciql.py is a hydra launcher: it runs HERE and
                                    # subprocesses ogbench main.py into the jax venv itself.
                                    # Do NOT point this at .venv-jax -- it has no hydra.
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
LOGDIR=/dev/shm/bilsg_logs
mkdir -p "$LOGDIR"

cfg=cube_single_state_gciql_flow_bilinear_sg_d50

JOBS=()
for s in 0 1 2; do
  JOBS+=("${cfg}__s${s}|||--config-name $cfg seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=${cfg}_s${s}")
done

GPUS=(${GPUS:-5 6 7})
echo "bilinear_sg: ${#JOBS[@]} runs @${STEPS} over ${#GPUS[@]} GPUs"
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
