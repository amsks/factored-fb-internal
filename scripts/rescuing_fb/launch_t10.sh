#!/usr/bin/env bash
# T10 centered (advantage) successor measure launch
# (PAPER/rescuing-fb/04-centered-measure-spec.md §5), 2026-07-16.
#
# 3 seeds x 2 arms, 500k, eval every 50k, ALL packed on GPU 7 (GPUs 0/1/2/3/4/5/6
# belong to other tracks -- never touch). Stagger starts by >=90 s: co-initializing
# jax processes OOM even with XLA_PYTHON_CLIENT_PREALLOCATE=false.
#   T10-a aux : GCIQL-State-Flow-CenteredMeasure-Aux-d50   (lam_DeltaM=1)
#   T10-b only: GCIQL-State-Flow-CenteredMeasure-Only-d50  (SARSA base leg ->
#               needs OGBENCH_NEXT_ACTIONS=1, set per-run below)
#
# ALWAYS in tmux ([[always-run-in-tmux]]):
#   tmux new -d -s t10 'DRY_RUN=0 scripts/rescuing_fb/launch_t10.sh 2>&1 | tee /dev/shm/t10_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$(pwd)/.venv/bin/python}   # run_gciql.py is a hydra launcher: it runs HERE and
                                    # subprocesses ogbench main.py into the jax venv itself.
                                    # Do NOT point this at .venv-jax -- it has no hydra.
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
OUTROOT=${OUTROOT:-/home/adityamohan/exp_t10}
GPU=${GPU:-7}
STAGGER=${STAGGER:-100}
SEEDS=(${SEEDS:-0 1 2})
LOGDIR=/dev/shm/t10_logs
mkdir -p "$LOGDIR" "$OUTROOT"

# tag ||| extra-env ||| hydra args
JOBS=()
for s in "${SEEDS[@]}"; do
  JOBS+=("t10a_aux_s${s}||| |||--config-name cube_single_state_t10_centered_aux_d50 seed=$s train_steps=$STEPS output_root=$OUTROOT exp_name=t10a_aux_s${s}")
done
for s in "${SEEDS[@]}"; do
  JOBS+=("t10b_only_s${s}|||OGBENCH_NEXT_ACTIONS=1|||--config-name cube_single_state_t10_centered_only_d50 seed=$s train_steps=$STEPS output_root=$OUTROOT exp_name=t10b_only_s${s}")
done

echo "t10: ${#JOBS[@]} runs @${STEPS} packed on GPU $GPU -> $OUTROOT (stagger ${STAGGER}s)"
if [ "$DRY_RUN" = "1" ]; then
  for j in "${JOBS[@]}"; do echo "GPU$GPU  ${j%%|||*}"; done
  exit 0
fi

first=1
for j in "${JOBS[@]}"; do
  tag=${j%%|||*}; rest=${j#*|||}; extra_env=${rest%%|||*}; args=${rest#*|||}
  if [ "$first" = "0" ]; then
    echo "[stagger] sleeping ${STAGGER}s before $tag (jax co-init OOMs)"
    sleep "$STAGGER"
  fi
  first=0
  echo "[launch] GPU$GPU  $tag  (env: ${extra_env:-none})"
  env $extra_env XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU \
    "$PY" run_gciql.py $args >"$LOGDIR/${tag}.log" 2>&1 &
  echo "[mem] GPU$GPU after $tag:"; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i "$GPU" || true
done
echo "Launched ${#JOBS[@]} jax runs -> $LOGDIR ; waiting..."
wait
echo "ALL DONE"
