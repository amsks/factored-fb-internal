#!/usr/bin/env bash
# Joint-psi SF cells (spec docs/superpowers/specs/2026-07-10-joint-sf-ortho-design.md).
#
#   A  sf_joint_ortho1k          b=psi(s'), no mask   -> visitation density; matched to
#                                GCIQL-State-Flow-Bilinear-d50-C-FBzProjZ-O1000 = 0.231 +/- 0.030
#                                in every respect EXCEPT the estimator (per-sample vs BxB).
#   C  sf_joint_ortho1k_absorb   b=psi(s), goal mask  -> Q = gamma^T (first-hitting)
#
# ALWAYS in tmux:
#   tmux new -d -s sfjoint 'DRY_RUN=0 scripts/launch_sf_joint.sh 2>&1 | tee /dev/shm/sfjoint_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(pwd)/.venv/bin/python}   # run_gciql.py is a hydra launcher: it runs HERE and
                                    # subprocesses ogbench main.py into .venv-jax itself.
                                    # Do NOT point this at .venv-jax -- that venv has no hydra.
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
LOGDIR=/dev/shm/sfjoint_logs
mkdir -p "$LOGDIR"

configs=(
  cube_single_state_gciql_flow_sf_joint_ortho1k
  cube_single_state_gciql_flow_sf_joint_ortho1k_absorb
)

JOBS=()
for cfg in "${configs[@]}"; do
  for s in 0 1 2; do
    JOBS+=("${cfg}__s${s}|||--config-name $cfg seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=${cfg}_s${s}")
  done
done

GPUS=(${GPUS:-0 1 2 3 4 5})
echo "joint-psi SF: ${#JOBS[@]} runs @${STEPS} over ${#GPUS[@]} GPUs"
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
