#!/usr/bin/env bash
# Full GCIQL state run on cube-single-play-v0 with OGBench's EXACT
# hyperparameters (1M steps, alpha=1.0 — config defaults, not overridden).
# Multi-seed; online wandb by default so it overlays with FB runs (GCIQL eval
# metrics are mirrored to FB's wandb key scheme by tools/wandb_mode_shim).
# Mirrors scripts/run_sweep.sh ergonomics.
#   SEEDS="0 1 2"  WANDB_MODE=online|offline|disabled  RUN_GROUP=...  STORAGE=shm|nvme
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SEEDS="${SEEDS:-0 1 2}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_GROUP="${RUN_GROUP:-factored-fb-gciql}"
STORAGE="${STORAGE:-shm}"

echo "[gciql-state] seeds=$SEEDS wandb=$WANDB_MODE group=$RUN_GROUP storage=$STORAGE"
for seed in $SEEDS; do
    echo "[gciql-state] $(date +%T)  launching seed=$seed (1M steps, cube-single-play-v0)"
    python run_gciql.py --config-name cube_single_state \
        storage="$STORAGE" seed="$seed" wandb_mode="$WANDB_MODE" run_group="$RUN_GROUP"
    echo "[gciql-state] $(date +%T)  done seed=$seed"
done
echo "[gciql-state] ALL DONE"
