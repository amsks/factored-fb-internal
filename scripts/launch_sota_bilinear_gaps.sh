#!/usr/bin/env bash
# SOTA-Bilinear paper: fill two gaps, 3 seeds @500k each (15 jax runs).
#   (1) scene GCIQL bilinear-d50 + DDPG+BC actor  -> the missing scene-SOTA contender
#   (2) HIQL debug at the CORRECT subgoal_steps=10 (all prior HIQL runs used the
#       OGBench default 25, which confounds the bilinear-vs-MLP comparison):
#         native AWR : MLP vs bilinear            (canonical HIQL; should reproduce ~15)
#         flow actor : MLP vs bilinear            (secondary; staged sg10 configs)
# Packed ~2/GPU with XLA_PYTHON_CLIENT_PREALLOCATE=false. ALWAYS in tmux:
#   tmux new -s sotagap 'DRY_RUN=0 scripts/launch_sota_bilinear_gaps.sh 2>&1 | tee /dev/shm/sotagap_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(pwd)/.venv/bin/python}
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
LOGDIR=/dev/shm/sotagap_logs
mkdir -p "$LOGDIR"

configs=(
  scene_state_gciql_bilinear_d50_ddpgbc
  cube_single_state_hiql_native_mlp_sg10
  cube_single_state_hiql_native_bilinear_d50_sg10
  cube_single_state_hiql_flow_both_mlp_sg10
  cube_single_state_hiql_flow_both_bilinear_d50_sg10
)

JOBS=()
for cfg in "${configs[@]}"; do
  for s in 0 1 2; do
    JOBS+=("${cfg}__s${s}|||--config-name $cfg seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=${cfg}_s${s}")
  done
done

GPUS=(0 1 2 3 4 5 6 7)
echo "SOTA-bilinear gaps: ${#JOBS[@]} runs @${STEPS}"
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
