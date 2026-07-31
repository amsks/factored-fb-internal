#!/bin/bash
# Persistent tmux workspace for Factored-FB on TACC Vista.
#
#   bash scripts/tmux_dev.sh            # create (or attach to) session "fb"
#   bash scripts/tmux_dev.sh myname     # same, under a different session name
#   bash scripts/tmux_dev.sh fb kill    # tear the session down
#
# Why: tmux survives dropped SSH connections, so a long-running `claude` (or a
# tail on a training log) keeps going when your laptop sleeps. Reconnect with
# the SAME login node — tmux sockets live in /tmp, which is node-local:
#
#   ssh amsks@login2.vista.tacc.utexas.edu   # <- must match $(hostname)
#
# Windows created (numbered from 1 — see base-index in ~/.tmux.conf):
#   1 claude  — claude running in the repo (the point of the whole thing)
#   2 shell   — plain shell, venv activated
#   3 jobs    — squeue watch, refreshed every 30s
set -euo pipefail

SESSION="${1:-fb}"
ACTION="${2:-attach}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$SCRATCH/.venv-jax-fb}"

if [[ "$ACTION" == "kill" ]]; then
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "[tmux] killed '$SESSION'" \
        || echo "[tmux] no session '$SESSION'"
    exit 0
fi

# Already there? Just go. (Nested tmux would be a mess, so switch instead.)
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[tmux] attaching to existing '$SESSION' on $(hostname -s)"
    if [[ -n "${TMUX:-}" ]]; then
        exec tmux switch-client -t "$SESSION"
    else
        exec tmux attach-session -t "$SESSION"
    fi
fi

echo "[tmux] creating '$SESSION' on $(hostname -s)"
echo "[tmux] repo = $REPO"
echo "[tmux] venv = $VENV"

# Sourced by every window so each pane lands in the repo with the venv live.
ACTIVATE="cd '$REPO'"
[[ -f "$VENV/bin/activate" ]] && ACTIVATE="$ACTIVATE && source '$VENV/bin/activate'"

tmux new-session  -d -s "$SESSION" -n claude -c "$REPO"
tmux send-keys    -t "$SESSION:claude" "$ACTIVATE && claude" C-m

tmux new-window   -t "$SESSION" -n shell -c "$REPO"
tmux send-keys    -t "$SESSION:shell" "$ACTIVATE" C-m

tmux new-window   -t "$SESSION" -n jobs -c "$REPO"
tmux send-keys    -t "$SESSION:jobs" \
    "watch -n 30 'squeue -u \$USER -o \"%.10i %.12P %.20j %.2t %.10M %.6D %R\"'" C-m

tmux select-window -t "$SESSION:claude"

# Mouse, scrollback and the "#S @#h" status bar all come from ~/.tmux.conf now;
# setting them here too would just fight it.

if [[ -n "${TMUX:-}" ]]; then
    exec tmux switch-client -t "$SESSION"
else
    exec tmux attach-session -t "$SESSION"
fi
