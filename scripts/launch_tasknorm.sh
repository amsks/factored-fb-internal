#!/usr/bin/env bash
# Task-weighted successor moments (spec docs/superpowers/specs/2026-07-10-task-weighted-moments-design.md).
#
#   N1  C_tasknorm   measure residual tested against psi(value_goals) (the task norm nu)
#                    -> vs C-fbz-projz-ortho1000 = 0.231 +/- 0.030 (same cell, test states = rho)
#   N2  C_taskproj   + rank-1: project each residual onto its own z_i = psi(g_i)
#                    -> fixed point is GCIQL's backup with a learned kernel reward
#
# ALWAYS in tmux:
#   tmux new -d -s tasknorm 'DRY_RUN=0 scripts/launch_tasknorm.sh 2>&1 | tee /dev/shm/tasknorm_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(pwd)/.venv/bin/python}   # run_gciql.py is a hydra launcher: it runs HERE and
                                    # subprocesses ogbench main.py into .venv-jax itself.
                                    # Do NOT point this at .venv-jax -- that venv has no hydra.
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
LOGDIR=/dev/shm/tasknorm_logs
mkdir -p "$LOGDIR"

configs=(
  cube_single_state_gciql_flow_bilinear_d50_C_tasknorm
  cube_single_state_gciql_flow_bilinear_d50_C_taskproj
)

JOBS=()
for cfg in "${configs[@]}"; do
  for s in 0 1 2; do
    JOBS+=("${cfg}__s${s}|||--config-name $cfg seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=${cfg}_s${s}")
  done
done

GPUS=(${GPUS:-0 1 2 3 4 5})
echo "task-norm moments: ${#JOBS[@]} runs @${STEPS} over ${#GPUS[@]} GPUs"
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
