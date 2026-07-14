#!/usr/bin/env bash
# Annealed SM finetuning (spec docs/superpowers/specs/2026-07-10-annealed-sm-finetune-design.md).
#
#   Anneal-Zero    control: no measure loss. First eval MUST read ~0.90 or the restore is broken.
#   Anneal-SgPhi   lambda 0 -> 10  over 300k, hold 100k. Crosses the cold cliff (l3 = 0.176).
#   Anneal-Shared  lambda 0 -> 100 over 300k, hold 100k. Cold cliff above 10, never located.
#
# Step 0 writes the finetune-ready checkpoints (anchor params, fresh opt_state, step=0). Without
# them the LR schedule cannot restore at all, and lambda would start fully ramped.
#
# ALWAYS in tmux:
#   tmux new -d -s anneal 'DRY_RUN=0 scripts/launch_anneal.sh 2>&1 | tee /dev/shm/anneal_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
PY=${PY:-$REPO/.venv/bin/python}    # run_gciql.py is a hydra launcher: it runs HERE and
                                    # subprocesses ogbench main.py into .venv-jax itself.
                                    # Do NOT point this at .venv-jax -- that venv has no hydra.
JAXPY=${JAXPY:-/dev/shm/.venv-jax/bin/python}
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-400000}
CKPT_ROOT=${CKPT_ROOT:-/dev/shm/anneal_ckpt}
ANCHOR=~/bridge_ckpt_backup/gciql_outputs/factored-fb/GCIQL-State-Flow-Bilinear-d50
LOGDIR=/dev/shm/anneal_logs
mkdir -p "$LOGDIR"

# ---- step 0: finetune-ready checkpoints (idempotent) -------------------------------------------
prepare() {
  for s in 000 001 002; do
    if [ -f "$CKPT_ROOT/sd${s}/params_0.pkl" ]; then
      echo "[prep] sd${s} already present"
      continue
    fi
    SRC=$(ls -d ${ANCHOR}/sd${s}_* 2>/dev/null | head -1)
    [ -z "$SRC" ] && { echo "[prep] MISSING anchor for sd${s}"; exit 1; }
    echo "[prep] sd${s} <- $SRC"
    "$JAXPY" "$REPO/scripts/prepare_anneal_ckpt.py" \
        --src "$SRC" --epoch 1000000 --out "$CKPT_ROOT/sd${s}" --lr-warmup 10000 --seed "$((10#$s))"
  done
}

configs=(
  cube_single_state_gciql_flow_anneal_zero
  cube_single_state_gciql_flow_anneal_sgphi
  cube_single_state_gciql_flow_anneal_shared
)

JOBS=()
for cfg in "${configs[@]}"; do
  for s in 0 1 2; do
    sd=$(printf "sd%03d" "$s")
    JOBS+=("${cfg}__s${s}|||--config-name $cfg seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=${cfg}_s${s} restore_path=$CKPT_ROOT/$sd restore_epoch=0")
  done
done

GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
echo "annealed SM finetune: ${#JOBS[@]} runs @${STEPS} over ${#GPUS[@]} GPUs"
if [ "$DRY_RUN" = "1" ]; then
  echo "(dry run: would prepare checkpoints under $CKPT_ROOT, then:)"
  i=0; for j in "${JOBS[@]}"; do echo "GPU${GPUS[$((i % ${#GPUS[@]}))]}  ${j%%|||*}"; i=$((i+1)); done
  exit 0
fi

prepare

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
