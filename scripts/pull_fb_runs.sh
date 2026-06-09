#!/usr/bin/env bash
# scripts/pull_fb_runs.sh — run this ON YOUR LOCAL MACHINE to rsync FB/PSM state
# runs from the AWS GPU box into results/factored-fb-<label>/. Pulls run-output
# dirs (.hydra config, checkpoints/*.pt, train.log, local wandb) for four groups
# and renames two of them locally (v1/v2 -> human names).
#
# Usage:
#   bash scripts/pull_fb_runs.sh [DEST_ROOT]   # DEST_ROOT defaults to ./results
#
# Prereqs: an SSH host alias `aws-gpu` in ~/.ssh/config (Host aws-gpu ...).
set -euo pipefail

REMOTE="${REMOTE:-aws-gpu}"                       # ssh host alias for the AWS box
REMOTE_RUNS="${REMOTE_RUNS:-/dev/shm/factored-fb/runs}"   # source base on remote
DEST_ROOT="${1:-./results}"                       # local destination root

# -a archive  -v verbose  -z compress  -P = --partial --progress (resumable)
RSYNC_OPTS=(-avzP --partial)
# Skip the big checkpoint .pt files (keeps configs + wandb + logs only):
# RSYNC_OPTS+=(--exclude='checkpoints/')

# remote group name -> local label. v1=goal_cond, v2=fixed_b/USFA (confirmed via
# the feat/goal-conditioned-fb V1/V2 description in CLAUDE.md).
declare -A GROUPS=(
  [psm_flowbc_state]="psm_flowbc_state"
  [fb_state_L256]="fb_state_L256"
  [fb_state_v1]="goal_conditioned_fb"
  [fb_state_v2]="usfa"
)

for src in "${!GROUPS[@]}"; do
  label="${GROUPS[$src]}"
  dst="$DEST_ROOT/factored-fb-${label}"
  echo "==================================================================="
  echo "  $src  ->  $dst"
  echo "==================================================================="
  mkdir -p "$dst"
  # Remote shell expands the glob; all 10 seed dirs land under $dst/.
  rsync "${RSYNC_OPTS[@]}" \
    "$REMOTE:$REMOTE_RUNS/*__${src}__s*" \
    "$dst/"
done

echo
echo "Done. Pulled into:"
for src in "${!GROUPS[@]}"; do
  echo "  $DEST_ROOT/factored-fb-${GROUPS[$src]}/   <- $src"
done
