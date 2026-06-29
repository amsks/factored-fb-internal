#!/usr/bin/env bash
# Back up every checkpoint the 10-seed x 1M bridge campaign needs to RESUME, off
# the volatile /dev/shm onto persistent disk. A reboot wipes /dev/shm; without
# this, the resume seeds (0-2 of each ablation) can no longer be continued and
# would have to be re-run from scratch.
#
# Usage:   BACKUP_DIR=/mnt/scratch1/bridge_ckpt_backup scripts/backup_bridge_ckpts.sh
# Default: $HOME/bridge_ckpt_backup
#
# Restore later with:  rsync -a "$BACKUP_DIR"/dev/shm/  /dev/shm/
set -uo pipefail

BACKUP_DIR=${BACKUP_DIR:-$HOME/bridge_ckpt_backup}
GOUT=/dev/shm/gciql_outputs/factored-fb
FOUT=/dev/shm/factored-fb_outputs
FB_RUNS=/dev/shm/factored-fb/runs
mkdir -p "$BACKUP_DIR"
echo "Backing up to: $BACKUP_DIR"

# rsync filter: only the files needed to resume / re-extract, not the 100s of MB
# of intermediate state. (final ckpts + the small metadata/csv per run.)
COMMON_INCLUDES=(
  --include='*/'
  --include='params_500000.pkl' --include='params_1000000.pkl'
  --include='final.pt' --include='step_*.pt'
  --include='eval.csv' --include='train.csv' --include='flags.json'
  --include='.hydra/***'
  --exclude='*'
)

backup_tree() {  # $1 = source dir under /dev/shm
  local src=$1
  [ -d "$src" ] || { echo "  (skip, missing: $src)"; return; }
  echo "  $src"
  rsync -aR "${COMMON_INCLUDES[@]}" "/dev/shm/./${src#/dev/shm/}" "$BACKUP_DIR/"
}

echo "== GCIQL/JAX checkpoints =="
backup_tree "$GOUT"
echo "== FB/torch checkpoints (factored-fb_outputs) =="
backup_tree "$FOUT"
echo "== FB plain-FB anchor (factored-fb/runs) =="
backup_tree "$FB_RUNS"

echo
echo "Done. Size:"
du -sh "$BACKUP_DIR" 2>/dev/null
echo "Counts:"
echo "  GCIQL params:  $(find "$BACKUP_DIR" -name 'params_*.pkl' 2>/dev/null | wc -l)"
echo "  FB .pt:        $(find "$BACKUP_DIR" -name '*.pt' 2>/dev/null | wc -l)"
