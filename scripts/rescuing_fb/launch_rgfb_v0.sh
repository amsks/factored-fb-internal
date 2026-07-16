#!/usr/bin/env bash
# RG-FB v0 launch (PAPER/rescuing-fb/02-rgfb-spec.md §2.5), 2026-07-16.
#
# 3 seeds x 500k, eval every 50k, one seed per GPU (GPUs 2/3/4 -- GPUs 1/5/6
# belong to the other session's track, never touch). Persistent output root.
#
# ALWAYS in tmux ([[always-run-in-tmux]]):
#   tmux new -d -s rgfb 'DRY_RUN=0 scripts/rescuing_fb/launch_rgfb_v0.sh 2>&1 | tee /dev/shm/rgfb_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$(pwd)/.venv/bin/python}   # run_gciql.py is a hydra launcher: it runs HERE and
                                    # subprocesses ogbench main.py into the jax venv itself.
                                    # Do NOT point this at .venv-jax -- it has no hydra.
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
OUTROOT=${OUTROOT:-/home/adityamohan/exp_rgfb}
LOGDIR=/dev/shm/rgfb_logs
mkdir -p "$LOGDIR" "$OUTROOT"

cfg=cube_single_state_rgfb_d50

JOBS=()
for s in 0 1 2; do
  JOBS+=("rgfb_v0_s${s}|||--config-name $cfg seed=$s train_steps=$STEPS output_root=$OUTROOT exp_name=rgfb_v0_s${s}")
done

GPUS=(${GPUS:-2 3 4})
echo "rgfb_v0: ${#JOBS[@]} runs @${STEPS} over GPUs ${GPUS[*]} -> $OUTROOT (group RGFB-Cube-Single-State-d50-v0)"
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
