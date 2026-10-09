---
name: always-run-in-tmux
description: "All training/long-running jobs must launch in a detached tmux session, never bare nohup/background"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 6787d218-60cd-4a6d-ad25-71a042116338
---

Always launch training runs / long jobs in a **detached tmux session**, not a bare
`nohup ... &` or a Claude background Bash task.

**Why:** runs must survive the chat dying and stay trackable across future Claude
sessions. A `nohup &` started from a chat tool call is invisible to the next
session and hard to attach to; a named tmux session can be listed (`tmux ls`),
attached (`tmux attach -t <name>`), and its purpose recovered.

**How to apply:**
- Launch: `tmux new-session -d -s <name> "<cmd> 2>&1 | tee /dev/shm/<name>_launch.log"`.
- Name the session after the run group (e.g. `fb_state_encmatch`).
- After launching, record it in [[tmux-sessions-registry]] (session name, purpose,
  seeds/GPUs, date) so future sessions know what's running.
- Inspect with `tmux capture-pane -pt <name>` or tail the tee'd log; don't rely on
  the chat's background-task notifications for run state.
