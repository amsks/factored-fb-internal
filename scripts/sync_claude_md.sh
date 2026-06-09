#!/usr/bin/env bash
# scripts/sync_claude_md.sh — push the local-only CLAUDE.md to the AWS GPU box.
# CLAUDE.md is gitignored (never committed), so it must be synced out-of-band
# after editing. Run from your local machine (or wherever you edit CLAUDE.md).
#
# Usage:
#   bash scripts/sync_claude_md.sh
#
# Prereqs: ssh host alias `aws-gpu`; override REMOTE/REPO_PATH via env if needed.
set -euo pipefail

REMOTE="${REMOTE:-aws-gpu}"                  # ssh host alias for the AWS box
REPO_PATH="${REPO_PATH:-git/Factored-FB}"    # repo path on the remote (relative to $HOME)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rsync -avz "$HERE/CLAUDE.md" "$REMOTE:$REPO_PATH/CLAUDE.md"
echo "Synced CLAUDE.md -> $REMOTE:$REPO_PATH/CLAUDE.md"
