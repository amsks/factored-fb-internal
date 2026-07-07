#!/usr/bin/env bash
# E2: scale-controlled IQL + lambda*measure interpolation (zeroshot spec §E2; audit-20260706 §1).
# The FB successor-measure GRAFTED onto the working ~0.90 IQL bilinear-d50 flow cell with
# measure_scale_norm ON (dimensionless gradient) so it no longer crashes via <F,z> inflation.
# 5 cells x 3 seeds = 15 jax runs @500k. Reads the answer to "does the measure gradient actively
# destroy task-relevant value structure, coexist, or help?" via a lambda dose-response + a
# stopgrad-phi (quarantined) arm.
#   I-l001/l01/l1   : shared-legs, measure_coef 0.01/0.1/1.0
#   Q-l01/l1-sgphi  : quarantined (measure gradient -> psi only), coef 0.1/1.0
# Guardrail: flag any run whose critic/q_mean escapes [-100,10] or grad_norm>500 (confound back).
# Packed round-robin ~2/GPU with XLA_PYTHON_CLIENT_PREALLOCATE=false. ALWAYS in tmux:
#   tmux new -s e2measure 'DRY_RUN=0 scripts/launch_e2_measure_mix.sh 2>&1 | tee /dev/shm/e2measure_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(pwd)/.venv/bin/python}
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
LOGDIR=/dev/shm/e2measure_logs
mkdir -p "$LOGDIR"

configs=(
  cube_single_state_gciql_flow_bilinear_d50_measuremix_l001
  cube_single_state_gciql_flow_bilinear_d50_measuremix_l01
  cube_single_state_gciql_flow_bilinear_d50_measuremix_l1
  cube_single_state_gciql_flow_bilinear_d50_measuremix_l01_sgphi
  cube_single_state_gciql_flow_bilinear_d50_measuremix_l1_sgphi
)

JOBS=()
for cfg in "${configs[@]}"; do
  for s in 0 1 2; do
    JOBS+=("${cfg}__s${s}|||--config-name $cfg seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=${cfg}_s${s}")
  done
done

GPUS=(0 1 2 3 4 5 6 7)
echo "E2 measure-mix: ${#JOBS[@]} runs @${STEPS}"
if [ "$DRY_RUN" = "1" ]; then
  i=0; for j in "${JOBS[@]}"; do echo "GPU${GPUS[$((i % ${#GPUS[@]}))]}  ${j%%|||*}"; i=$((i+1)); done
  echo "(dry run; set DRY_RUN=0 to launch)"; exit 0
fi

i=0
for j in "${JOBS[@]}"; do
  tag=${j%%|||*}; args=${j#*|||}
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  echo "[launch] GPU$g  $tag"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    $PY run_gciql.py $args >"$LOGDIR/${tag}.log" 2>&1 &
  i=$((i+1))
  sleep 8
done
echo "Launched ${#JOBS[@]} jax runs -> $LOGDIR ; waiting..."
wait
echo "ALL DONE"
