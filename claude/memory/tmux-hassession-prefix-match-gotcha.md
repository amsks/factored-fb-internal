---
name: tmux-hassession-prefix-match-gotcha
description: "tmux has-session -t NAME does PREFIX matching: a chained watcher named NAME2 waiting on 'has-session -t NAME' matches ITSELF and spins forever. Always use -t =NAME (exact match) in deferred-launch wait loops."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1578c60e-8303-4d93-bbf9-fa70237631fc
---

`tmux has-session -t sfrewreg` returned true forever because it PREFIX-matched the
watcher's own session `sfrewreg2` after `sfrewreg` died — the chained v2 launch never
fired and 3 GPUs sat idle (hit 2026-07-09, E3 rewreg campaign).

**Why:** tmux target resolution tries exact match, then unique prefix match. Any watcher
session whose name extends the watched session's name matches itself.

**How to apply:** in every deferred-launch wait loop (`while tmux has-session -t X; do
sleep 60; done`), write `-t =X` (leading `=` forces exact match) — or name the watcher
something that is not an extension of the watched name. Check this whenever chaining
tmux campaigns ([[always-run-in-tmux]], [[tmux-sessions-registry]]).
