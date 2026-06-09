#!/usr/bin/env bash
# pull_fb_runs.sh — run this ON YOUR LOCAL MACHINE to rsync FB/PSM state runs
# from the AWS GPU box. Pulls run-output dirs (.hydra config, checkpoints/*.pt,
# train.log, local wandb) for four groups and renames two of them locally.
#
# Usage:
#   bash pull_fb_runs.sh [DEST_ROOT]      # DEST_ROOT defaults to ./results
#
# Lands runs under ./results/ using the same factored-fb-<method> folder
# convention as the rest of that tree (e.g. factored-fb-crl-flowbc).
#
# Prereqs: an SSH host alias `aws-gpu` in ~/.ssh/config (Host aws-gpu ...).
set -euo pipefail

REMOTE="aws-gpu"                              # <-- your ssh host alias
REMOTE_RUNS="/dev/shm/factored-fb/runs"       # source base on the AWS box
DEST_ROOT="${1:-./results}"                   # local destination root

# -a archive  -v verbose  -z compress  -P = --partial --progress (resumable)
RSYNC_OPTS=(-avzP --partial)
# Skip the big checkpoint .pt files (keeps configs + wandb + logs only):
# RSYNC_OPTS+=(--exclude='checkpoints/')

MAX_RETRIES=8        # per-group; transient SSH drops (exit 255) auto-resume via --partial
RETRY_SLEEP=15       # seconds between attempts

# remote group name  ->  local folder name (factored-fb-<method> convention,
# matching the rest of ./results/). Two parallel indexed arrays so this works
# on bash 3.2 (macOS default), which lacks associative arrays (declare -A).
# NOTE: v1->goal-conditioned-fb, v2->usfa is an ASSUMPTION from the order you
# gave ("goal conditioned FB and USFA"). If it's reversed, swap the two values.
SRC_GROUPS=(psm_flowbc_state fb_state_L256                fb_state_v1                       fb_state_v2)
DST_GROUPS=(factored-fb-psm-flowbc factored-fb-fb-L256    factored-fb-goal-conditioned-fb   factored-fb-usfa)

for i in "${!SRC_GROUPS[@]}"; do
  src="${SRC_GROUPS[$i]}"
  dst="${DST_GROUPS[$i]}"
  echo "==================================================================="
  echo "  $src  ->  $DEST_ROOT/$dst"
  echo "==================================================================="
  mkdir -p "$DEST_ROOT/$dst"
  # Remote shell expands the glob; all 10 seed dirs land under $DEST_ROOT/$dst/.
  # Retry per group so a transient SSH drop (exit 255) resumes instead of aborting
  # the whole run; --partial means already-transferred bytes are not re-fetched.
  attempt=1
  until rsync "${RSYNC_OPTS[@]}" \
          "$REMOTE:$REMOTE_RUNS/*__${src}__s*" \
          "$DEST_ROOT/$dst/"; do
    rc=$?
    if [ "$attempt" -ge "$MAX_RETRIES" ]; then
      echo "  !! $src: rsync still failing (rc=$rc) after $attempt attempts; moving on." >&2
      break
    fi
    echo "  .. $src: rsync exited $rc (attempt $attempt/$MAX_RETRIES); resuming in ${RETRY_SLEEP}s..." >&2
    attempt=$((attempt + 1))
    sleep "$RETRY_SLEEP"
  done
done

echo
echo "Done. Layout:"
echo "  $DEST_ROOT/factored-fb-psm-flowbc/         <- psm_flowbc_state"
echo "  $DEST_ROOT/factored-fb-fb-L256/            <- fb_state_L256"
echo "  $DEST_ROOT/factored-fb-goal-conditioned-fb/<- fb_state_v1"
echo "  $DEST_ROOT/factored-fb-usfa/               <- fb_state_v2"
echo
echo "Each seed dir keeps its source name, e.g. 20260602_203933__fb_state_v1__s0/"
