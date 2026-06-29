#!/usr/bin/env bash
# Periodic safety sync of the RUNNING bridge-knobs (FB) + HIQL/GCIVL-bilinear (jax)
# campaigns off volatile /dev/shm onto persistent disk. Checkpoints land at 500k
# (save_interval=500000); this loop mirrors them as soon as they appear, plus logs.
# Run in tmux:  tmux new -s cksync 'scripts/sync_running_campaigns.sh'
set -uo pipefail
BACKUP_DIR=${BACKUP_DIR:-$HOME/bridge_ckpt_backup}
INTERVAL=${INTERVAL:-600}      # seconds between passes
mkdir -p "$BACKUP_DIR"

CKPT_INC=( --include='*/' \
  --include='final.pt' --include='step_*.pt' --include='train_state.json' --include='train.log' \
  --include='params_*.pkl' --include='eval.csv' --include='train.csv' --include='flags.json' \
  --include='.hydra/***' \
  --exclude='*' )

sync_tree(){ local src=$1; [ -d "$src" ] || return 0
  rsync -aR "${CKPT_INC[@]}" "/dev/shm/./${src#/dev/shm/}" "$BACKUP_DIR/" 2>/dev/null; }

pass(){
  sync_tree /dev/shm/bridge_knobs_out
  sync_tree /dev/shm/gciql_outputs/factored-fb/HIQL-State-Bilinear-d50
  sync_tree /dev/shm/gciql_outputs/factored-fb/GCIVL-State-Bilinear-d50
  # logs (cheap, full copy)
  rsync -a /dev/shm/bridge_knobs_logs "$BACKUP_DIR/" 2>/dev/null
  rsync -a /dev/shm/hiql_gcivl_bil_logs "$BACKUP_DIR/" 2>/dev/null
}

echo "[cksync] -> $BACKUP_DIR every ${INTERVAL}s"
while true; do
  ts=$(date '+%H:%M:%S' 2>/dev/null || echo '?')
  pass
  nck=$(find "$BACKUP_DIR/bridge_knobs_out" -name '*.pt' 2>/dev/null | wc -l)
  npk=$(find "$BACKUP_DIR" -path '*State-Bilinear*' -name 'params_*.pkl' 2>/dev/null | wc -l)
  echo "[cksync $ts] FB .pt=$nck  jax params=$npk  size=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1)"
  sleep "$INTERVAL"
done
