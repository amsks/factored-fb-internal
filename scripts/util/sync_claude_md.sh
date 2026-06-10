#!/usr/bin/env bash
# sync_claude_md.sh — push the local-only (gitignored) CLAUDE.md to the AWS box.
# CLAUDE.md is not committed, so it has to travel out-of-band. Run this after
# editing CLAUDE.md so the remote agent sessions see the same orientation doc.
#
# Usage: bash scripts/util/sync_claude_md.sh
set -euo pipefail

REMOTE="aws-gpu"                                   # ssh host alias
REMOTE_REPO="/home/adityamohan/git/Factored-FB"    # repo root on the AWS box
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rsync -avz "$HERE/CLAUDE.md" "$REMOTE:$REMOTE_REPO/CLAUDE.md"
echo "Synced CLAUDE.md -> $REMOTE:$REMOTE_REPO/CLAUDE.md"
