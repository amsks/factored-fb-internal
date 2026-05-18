#!/usr/bin/env bash
# Launch GCIQL (vendored OGBench, JAX) via the thin Hydra launcher.
# Env-var parameterised, mirrors scripts/run_sweep.sh ergonomics.
#   VARIANT=state|visual  SEEDS="0 1 2"  WANDB_MODE=online|offline|disabled
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

VARIANT="${VARIANT:-state}"
SEEDS="${SEEDS:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
CONFIG="cube_single_${VARIANT}"

echo "[gciql] variant=$VARIANT seeds=$SEEDS wandb=$WANDB_MODE config=$CONFIG"
for seed in $SEEDS; do
    echo "[gciql] $(date +%T)  launching seed=$seed"
    python run_gciql.py --config-name "$CONFIG" \
        seed="$seed" wandb_mode="$WANDB_MODE"
    echo "[gciql] $(date +%T)  done seed=$seed"
done
