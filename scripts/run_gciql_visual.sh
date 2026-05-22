#!/usr/bin/env bash
# Full OGBench visual run on visual-cube-single-play-v0 using OGBench's visual
# hyperparameters (500k steps, encoder=impala_small, p_aug=0.5, batch_size=256).
# Agent/recipe is selected by CONFIG (a Hydra config under configs/gciql):
#   CONFIG=cube_single_visual        -> GCIQL (alpha=1.0,  group factored-fb-gciql-pixel)
#   CONFIG=cube_single_visual_gcivl  -> GCIVL (alpha=10.0, group factored-fb-gcivl-pixel)
# Online wandb by default so eval metrics overlay with FB-pixel runs (mirrored to
# FB's wandb key scheme by tools/wandb_mode_shim).
#   CONFIG=...  SEEDS="0 1 2"  WANDB_MODE=online|offline|disabled  RUN_GROUP=...  STORAGE=shm|nvme
# RUN_GROUP is optional: if unset, the selected config's own run_group default is used.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CONFIG="${CONFIG:-cube_single_visual}"
SEEDS="${SEEDS:-0 1 2}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_GROUP="${RUN_GROUP:-}"
STORAGE="${STORAGE:-shm}"

echo "[ogbench-visual] config=$CONFIG seeds=$SEEDS wandb=$WANDB_MODE group=${RUN_GROUP:-<config default>} storage=$STORAGE"
echo "[ogbench-visual] env=visual-cube-single-play-v0 (500k steps, impala_small, p_aug=0.5)"
for seed in $SEEDS; do
    echo "[ogbench-visual] $(date +%T)  launching seed=$seed"
    extra=()
    [ -n "$RUN_GROUP" ] && extra+=("run_group=$RUN_GROUP")
    python run_gciql.py --config-name "$CONFIG" \
        storage="$STORAGE" seed="$seed" wandb_mode="$WANDB_MODE" "${extra[@]}"
    echo "[ogbench-visual] $(date +%T)  done seed=$seed"
done
echo "[ogbench-visual] ALL DONE"
