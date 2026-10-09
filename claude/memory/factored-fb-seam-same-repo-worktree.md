---
name: factored-fb-seam-same-repo-worktree
description: "~/git/Factored-FB and ~/git/Factored-FB-seam are two git WORKTREES of one repo on different branches — the branch decides what exists, not the directory"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0c552c33-cc1f-4376-8470-a0e87d93b4eb
  modified: 2026-08-31T17:43:01.677Z
---

`~/git/Factored-FB` (branch `gciql-fb-bridge`) and `~/git/Factored-FB-seam` (branch
`density-fb`) are **two worktrees of the same repository**, not two repos. `git worktree
list` from either shows both, plus `/dev/shm/fb-analysis` (detached).

So "is file X in my repo vs their repo" is the wrong question — **the branch decides**.
Concretely: `campaign/wandb_agg_daemon.py` and the whole `tests/impls/` tree exist on
`density-fb` (and `feat/jaxify`, `origin/tl-*`) but **do not exist on `gciql-fb-bridge`
at all**. Verify with `git cat-file -e <branch>:<path>` before claiming a file's home.

**Why:** on 2026-08-31 the other track's session asserted "wandb_agg_daemon doesn't exist
in Factored-FB-seam at all, it belongs in ~/git/Factored-FB" — exactly inverted. The file
was sitting in its own working tree; the destination branch was `density-fb`, i.e. its
own checkout. Both sessions reason in terms of "my repo / your repo" and that framing is
what produces the error.

**How to apply:** when routing a file between the two tracks, resolve the target by
branch (`git cat-file -e`, `git log --oneline <branch> -- <path>`), never by directory
name. Never write into the other worktree's tree while its track has runs going — pull
content out with `git show <branch>:<path>` instead of reading/writing its working
directory, and let the branch's owner land the commit. See [[exorl-validation-campaign-20260828]].
