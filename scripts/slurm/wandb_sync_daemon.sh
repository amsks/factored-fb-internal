#!/usr/bin/env bash
# scripts/slurm/wandb_sync_daemon.sh — Continuously sync WandB offline runs to the cloud.
#
# Run on the LOGIN NODE inside tmux while cluster jobs are executing:
#   tmux new -s wandb-sync
#   bash scripts/slurm/wandb_sync_daemon.sh
#
# The daemon wakes every INTERVAL seconds, finds every offline-run-* directory
# under OUTPUTS_DIR and GCIQL_DIR, and calls `wandb sync` on each.
# On exit (Ctrl-C or tmux kill) it performs one final sync.
#
# Env overrides:
#   INTERVAL=300          sync period in seconds (default 300 = 5 min)
#   OUTPUTS_DIR=...       override FB outputs root
#   GCIQL_DIR=...         override GCIQL outputs root
#   WANDB_ENTITY=amsks    WandB entity (default amsks)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

OUTPUTS_DIR="${OUTPUTS_DIR:-/bigwork/nhwpmoha/Factored-FB/outputs}"
GCIQL_DIR="${GCIQL_DIR:-/bigwork/nhwpmoha/Factored-FB/gciql_outputs}"
INTERVAL="${INTERVAL:-300}"
WANDB_ENTITY="${WANDB_ENTITY:-amsks}"

source "$REPO/.venv/bin/activate"

sync_all() {
    local count=0
    while IFS= read -r -d '' run_dir; do
        echo "[sync] $(date +%T)  syncing: $run_dir"
        WANDB_ENTITY="$WANDB_ENTITY" wandb sync "$run_dir" 2>&1 | tail -1 || true
        (( count++ )) || true
    done < <(find "$OUTPUTS_DIR" "$GCIQL_DIR" -type d -name "offline-run-*" -print0 2>/dev/null)
    echo "[sync] $(date +%T)  synced $count run(s)"
}

cleanup() {
    echo ""
    echo "[sync] EXIT — running final sync..."
    sync_all
    echo "[sync] done."
}
trap cleanup EXIT

echo "[sync] daemon started  outputs=${OUTPUTS_DIR}  gciql=${GCIQL_DIR}  interval=${INTERVAL}s"
echo "[sync] entity=${WANDB_ENTITY}  project=factored-fb"

while true; do
    sync_all
    sleep "$INTERVAL"
done
