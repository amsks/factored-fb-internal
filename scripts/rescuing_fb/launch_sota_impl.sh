#!/usr/bin/env bash
# ogbench-sota implementations launcher (PAPER/ogbench-sota/implementations.md).
# 2026-07-20. DRY_RUN=1 by default (prints the plan, launches nothing).
#
# ARMS (select with ARMS="...", default "all"):
#   latefusion  item1 : GCIQL-State-Flow-LateFusion-d50            (3 seeds, 500k) [+ rerank]
#   pmatched    item2 : GCIQL-State-Flow-MLP-PMatched              (3 seeds, 500k) [+ rerank]
#   phinormcap  item4b: bilinear-d50 phi_norm_cap, DDPGBC + Flow   (3 seeds each, 500k)
#   bounded     item5 : GCIQL-State-Flow-BilinearSGBounded-d50     (3 seeds, 500k)
#                       ** CALIBRATE psi_bound_scale first (see the config) **
#   scenesplit  item7 : GCIQL-Scene-{Flow,DDPGBC}-BilinearSplit    (3 seeds each, 500k)
#                       ** SET value_split_dims from the scene goal layout first **
#
# CPU probes (run directly, not training campaigns):
#   item3  scripts/rescuing_fb/t5c_srank.py          (srank/PR trajectories on checkpoints)
#   item4a scripts/rescuing_fb/t3_offpeak_multienv.py (T3_PEAK_GEOMETRY=1, the ||phi||-Q rays)
#   item6  scripts/rescuing_fb/t9_advantage_rank.py  (true-advantage epsilon-rank, CPU-heavy)
#   rerank scripts/atlas/rerank_eval.py              (frozen-critic N-sweep for items 1/2)
#
# Env invariants: tmux; wandb entity amsks; XLA_PYTHON_CLIENT_PREALLOCATE=false; jax co-init
# OOMs -> stagger. GPUs 1/5/6 = other track (never touch); verify nvidia-smi + git log first.
#   tmux new -d -s sota 'DRY_RUN=0 ARMS="latefusion pmatched" scripts/rescuing_fb/launch_sota_impl.sh 2>&1 | tee /dev/shm/sota_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$(pwd)/.venv/bin/python}   # hydra launcher (repo .venv, NOT .venv-jax)
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
OUTROOT=${OUTROOT:-/home/adityamohan/exp_sota_impl}
STAGGER=${STAGGER:-90}
SEEDS=(${SEEDS:-0 1 2})
ARMS=${ARMS:-all}
GPUS=(${GPUS:-0 2 3 4 7})
LOGDIR=/dev/shm/sota_impl_logs
mkdir -p "$LOGDIR" "$OUTROOT"
want() { [ "$ARMS" = "all" ] || [[ " $ARMS " == *" $1 "* ]]; }

declare -a JOBS=()
add() { for s in "${SEEDS[@]}"; do JOBS+=("$1_s${s}|||--config-name $2 seed=$s train_steps=$STEPS output_root=$OUTROOT exp_name=$1_s${s}"); done; }
want latefusion && add latefusion cube_single_state_latefusion_d50
want pmatched   && add pmatched   cube_single_state_mlp_pmatched
want phinormcap && { add phinormcap_ddpgbc cube_single_state_phinormcap_d50_ddpgbc; add phinormcap_flow cube_single_state_phinormcap_d50_flow; }
want bounded    && add bounded    cube_single_state_bilinear_sg_bounded_d50
want scenesplit && { add scenesplit_flow scene_state_bilinear_split_flow; add scenesplit_ddpgbc scene_state_bilinear_split_ddpgbc; }

echo "[sota] ARMS='$ARMS' -> ${#JOBS[@]} runs @${STEPS} over GPUs ${GPUS[*]} -> $OUTROOT"
if [ "${#JOBS[@]}" = "0" ]; then echo "[sota] no arms selected"; exit 1; fi
if [ "$DRY_RUN" = "1" ]; then
  i=0; for j in "${JOBS[@]}"; do echo "GPU${GPUS[$((i % ${#GPUS[@]}))]}  ${j%%|||*}"; i=$((i+1)); done
  echo "[sota] DRY_RUN=1 (set DRY_RUN=0 to launch). Nothing launched."; exit 0
fi
i=0; first=1
for j in "${JOBS[@]}"; do
  tag=${j%%|||*}; args=${j#*|||}; g=${GPUS[$((i % ${#GPUS[@]}))]}
  [ "$first" = "0" ] && { echo "[stagger] ${STAGGER}s before $tag"; sleep "$STAGGER"; }
  first=0
  echo "[launch] GPU$g  $tag"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py $args >"$LOGDIR/${tag}.log" 2>&1 &
  i=$((i+1))
done
echo "[sota] launched ${#JOBS[@]} runs -> $LOGDIR ; waiting..."; wait; echo "ALL DONE"
