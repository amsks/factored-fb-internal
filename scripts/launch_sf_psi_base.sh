#!/usr/bin/env bash
# 2026-07-09 audit follow-up (PAPER/fb-gciql-bridge/sf-design-space.md).
#
# Two questions, 4 cells x 3 seeds @500k:
#
#   (1b) readout-vs-TD crux -- normal scalar IQL TD, crippled goal leg:
#        bilinear_d50_frozen_psi   psi = frozen random MLP, d=50
#        bilinear_identity_psi     psi(g) = g, d=28
#        bilinear_d28              rank control for the identity arm (learned psi, d=28)
#        vs the untouched control GCIQL-State-Flow-Bilinear-d50 = 0.90
#
#   (E3) corrected SF, re-run: both prior E3 runs regressed the reward on psi(s') while
#        GCDataset ties reward+mask to the CURRENT index (fixed in fb83d68).
#        sf_feature_td_rewreg_shift        agnostic phi
#        sf_feature_td_rewreg_shift_cond   phi(s,a,z)  (audit point 3: goal-dependent mask)
#
# ALWAYS in tmux ([[always-run-in-tmux]]):
#   tmux new -d -s psibase 'DRY_RUN=0 scripts/launch_sf_psi_base.sh 2>&1 | tee /dev/shm/psibase_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(pwd)/.venv/bin/python}   # run_gciql.py is a hydra launcher: it runs HERE and subprocesses
                                    # ogbench main.py into the jax venv itself. Do NOT point this at
                                    # .venv-jax -- that venv has no hydra.
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
LOGDIR=/dev/shm/psibase_logs
mkdir -p "$LOGDIR"

configs=(
  cube_single_state_gciql_flow_bilinear_d50_frozen_psi
  cube_single_state_gciql_flow_bilinear_identity_psi
  cube_single_state_gciql_flow_bilinear_d28
  cube_single_state_gciql_flow_sf_feature_td_rewreg_shift
  cube_single_state_gciql_flow_sf_feature_td_rewreg_shift_cond
)

JOBS=()
for cfg in "${configs[@]}"; do
  for s in 0 1 2; do
    JOBS+=("${cfg}__s${s}|||--config-name $cfg seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=${cfg}_s${s}")
  done
done

GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
echo "psi_base + corrected-E3: ${#JOBS[@]} runs @${STEPS} over ${#GPUS[@]} GPUs"
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
