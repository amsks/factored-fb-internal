#!/usr/bin/env bash
# Parallel DrQ-front-end pixel sweep for OGBench GCIQL/GCIVL on
# visual-cube-single-play-v0. Runs CONFIGS x SEEDS as 500k-step runs, one run
# per GPU, round-robin across GPUS; each GPU processes its assigned jobs
# sequentially. Workers are detached (nohup) so they survive the launching
# shell. Checkpoints -> /dev/shm (storage=shm), per-run logs -> logs/, wandb
# online by default. Run groups come from each config
# (factored-fb-{gciql,gcivl}-pixel-drq).
#
#   GPUS="3 5 6 7"  SEEDS="0 1 2 3 4"
#   CONFIGS="cube_single_visual_drq cube_single_visual_gcivl_drq"
#   WANDB_MODE=online|offline|disabled  STORAGE=shm|nvme
#
# Only launch on GPUs verified healthy (the eval-render 20s/it pathology is
# invisible at idle in nvidia-smi); GPUs 0,1,2,4 were degraded as of 2026-05-22.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

read -r -a GPUS <<< "${GPUS:-3 5 6 7}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2 3 4}"
read -r -a CONFIGS <<< "${CONFIGS:-cube_single_visual_drq cube_single_visual_gcivl_drq}"
WANDB_MODE="${WANDB_MODE:-online}"
STORAGE="${STORAGE:-shm}"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

# Build job list "config:seed" (configs outer, seeds inner).
JOBS=()
for cfg in "${CONFIGS[@]}"; do
    for s in "${SEEDS[@]}"; do JOBS+=("$cfg:$s"); done
done

manifest="logs/drq_sweep_${TS}.manifest"
ng=${#GPUS[@]}
{
    echo "[drq-sweep] $TS  gpus=${GPUS[*]}  jobs=${#JOBS[@]}  wandb=$WANDB_MODE  storage=$STORAGE"
} | tee "$manifest"

# Round-robin assign jobs to GPUs; spawn one detached worker per GPU.
for gi in "${!GPUS[@]}"; do
    g="${GPUS[$gi]}"
    gjobs=()
    for ji in "${!JOBS[@]}"; do
        (( ji % ng == gi )) && gjobs+=("${JOBS[$ji]}")
    done
    [ ${#gjobs[@]} -eq 0 ] && continue
    echo "[drq-sweep] GPU $g <- ${gjobs[*]}" | tee -a "$manifest"
    GPU="$g" GJOBS="${gjobs[*]}" TS="$TS" WANDB_MODE="$WANDB_MODE" STORAGE="$STORAGE" \
        nohup bash "$REPO/scripts/_drq_worker.sh" \
        > "logs/drq_sweep_worker_g${g}_${TS}.log" 2>&1 &
    echo "  worker pid=$! gpu=$g" | tee -a "$manifest"
    sleep 8  # stagger JIT/EGL startup to avoid a host-RAM OOM spike
done
echo "[drq-sweep] launched workers on ${#GPUS[@]} GPUs; manifest=$manifest" | tee -a "$manifest"
echo "[drq-sweep] tail a run: tail -f logs/gciql_pixel_drq_s0_${TS}.log"
