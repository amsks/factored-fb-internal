#!/usr/bin/env bash
# Per-GPU sequential worker for run_drq_pixel_sweep.sh. Not meant to be run
# directly: the launcher invokes it (detached, one per GPU) with these env vars:
#   GPU         physical GPU index to pin (CUDA_VISIBLE_DEVICES)
#   GJOBS       space-separated "config:seed" jobs for this GPU
#   TS          shared launch timestamp (for log names)
#   WANDB_MODE  online|offline|disabled
#   STORAGE     shm|nvme
# Runs each job to completion sequentially, pinned to $GPU, logging per run.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

for job in $GJOBS; do
    cfg="${job%:*}"
    seed="${job##*:}"
    case "$cfg" in *gcivl*) sh=gcivl ;; *) sh=gciql ;; esac
    log="logs/${sh}_pixel_drq_s${seed}_${TS}.log"
    echo "[worker g$GPU] $(date +%T) START $cfg seed=$seed -> $log"
    CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python run_gciql.py \
        --config-name "$cfg" \
        storage="$STORAGE" seed="$seed" wandb_mode="$WANDB_MODE" > "$log" 2>&1
    echo "[worker g$GPU] $(date +%T) DONE  $cfg seed=$seed rc=$?"
done
echo "[worker g$GPU] $(date +%T) ALL DONE"
