#!/usr/bin/env bash
# Full GCIQL pixel run on visual-cube-single-play-v0 using OGBench's visual
# hyperparameters (configs/gciql/cube_single_visual.yaml: 500k steps,
# encoder=impala_small, p_aug=0.5, batch_size=256). Mirrors
# scripts/run_gciql_state.sh ergonomics. Online wandb by default so GCIQL-pixel
# eval metrics overlay with FB-pixel runs (mirrored to FB's wandb key scheme by
# tools/wandb_mode_shim).
#   SEEDS="0 1 2"  WANDB_MODE=online|offline|disabled  RUN_GROUP=...  STORAGE=shm|nvme
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SEEDS="${SEEDS:-0 1 2}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_GROUP="${RUN_GROUP:-factored-fb-gciql}"
STORAGE="${STORAGE:-shm}"

echo "[gciql-visual] seeds=$SEEDS wandb=$WANDB_MODE group=$RUN_GROUP storage=$STORAGE"
echo "[gciql-visual] env=visual-cube-single-play-v0 (500k steps, impala_small, p_aug=0.5)"
for seed in $SEEDS; do
    echo "[gciql-visual] $(date +%T)  launching seed=$seed"
    python run_gciql.py --config-name cube_single_visual \
        storage="$STORAGE" seed="$seed" wandb_mode="$WANDB_MODE" run_group="$RUN_GROUP"
    echo "[gciql-visual] $(date +%T)  done seed=$seed"
done
echo "[gciql-visual] ALL DONE"
