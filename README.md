# factored-fb-internal: files outside git

This branch holds everything from the Factored-FB working folders on the training box that
was never committed: run logs, analysis outputs, paper drafts, notes, Claude memory and
settings. It was collected on 2026-10-09 when the box's folders were consolidated. All other
branches of this repo are the full git history of LUH-AI/Factored-FB (local branches, plus
`luh-origin/*` = the LUH-AI remote's branches as last fetched on the box).

Layout (each folder mirrors the paths inside the original checkout):

- `Factored-FB/`: main checkout (branch gciql-fb-bridge): analysis/, logs/, results/,
  PAPER/ drafts, docs/superpowers, .superpowers, .ref-papers, CLAUDE.md.
- `Factored-FB-seam/`: the ICLR worktree (branch density-fb): analysis/, logs/, PAPER/,
  website/, the cube video scripts, scripts/launch_sq_ablation.sh.
- `backup-factored-fb/`: the May 2026 snapshot folder (logs, docs, uncommitted-tracked.patch).
- `baselines/`: files that existed only locally: td_jepa's run_one.py, logs, uv.lock and
  uncommitted diff (upstream URL and commit in ORIGIN); the official RLDP code copy; the
  LaTeX of the earlier manipulation paper (its figure/real_folder is on HF).
- `claude/`: Claude Code memory for this project, the project's .claude/settings.local.json,
  and the global CLAUDE.md.

Not here, because they are binary or large: checkpoints, .npz/.pkl caches, wandb runs,
session transcripts and the large figure folder. They are on Hugging Face (private).
