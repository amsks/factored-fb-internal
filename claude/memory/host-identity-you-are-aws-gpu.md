---
name: host-identity-you-are-aws-gpu
description: Sessions run ON the aws-gpu box itself — do NOT ssh aws-gpu; verify host before reaching for remote-pull mental model
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 73bd5d8a-de5e-4c2d-a10c-a539bd53878c
---

This Claude Code session almost always runs **ON the aws-gpu GPU box itself**, NOT on the laptop. Identity check: hostname `ip-172-31-*.ec2.internal`, NVIDIA L4 GPUs present (`nvidia-smi -L`), `/dev/shm/factored-fb` exists, and `~/.ssh/config` is empty. When those hold, you ARE aws-gpu.

**Why:** CLAUDE.md and `scripts/pull_fb_runs.sh` / `scripts/sync_claude_md.sh` are all written from the *laptop's* POV — they frame `aws-gpu` as a remote rsync/ssh target ("run this ON YOUR LOCAL MACHINE", "ssh host alias aws-gpu", "pull from the AWS box"). With no positive host signal at session start, that framing makes you try `ssh aws-gpu`, which fails ("Could not resolve hostname aws-gpu") because a host has no ssh alias to itself.

**How to apply:** To check run status / tmux / GPUs, act LOCALLY (tmux ls, nvidia-smi, read /dev/shm/factored-fb_outputs, query wandb amsks/factored-fb directly). Never `ssh aws-gpu` from here. The `aws-gpu` alias and the pull/sync scripts only make sense from the laptop checkout. If hostname check ever shows you're NOT on an L4 EC2 box (no GPUs, ssh config has an aws-gpu alias), then you're on the laptop and remote ops apply. Related: [[wandb-entity-amsks]], [[tmux-sessions-registry]].
