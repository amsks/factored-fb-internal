---
name: density-fb-clean-branch-20260930
description: 2026-09-30 clean ICLR code branch density-fb-clean (worktree ../Factored-FB-clean); verified bit-exact vs snapshot; on HF iclr2027 under code/density-fb-clean; NOT pushed to GitHub (no creds)
metadata:
  node_type: memory
  type: project
  originSessionId: 668ff0f9-5340-40f5-9ef3-68f3ec7c8787
  modified: 2026-09-30T15:45:02.356Z
---

Branch `density-fb-clean` (worktree `/home/adityamohan/git/Factored-FB-clean`), built from
`backup/density-fb-snapshot-20260928`, holds only the code the ICLR paper uses: critics
density_fb (+ density_fb_z, density_fb_state), fb (fb_tdj/fb_ref), hilp (hilp_tdjc), rldp;
actors flowbc, ddpgbc; envs OGBench/ExoRL/MetaWorld/ManiSkill; figure scripts; tabular
analysis/ (now tracked). 304 files, -206k lines.

Verification: on 14 paper configs (built from real run config.json), trained params,
goal-conditioned actions and inferred-z actions are bit-exact vs the original code
(only removed logging metrics missing). `scripts/score_tables.py --exp <hf exp>` reproduces
all 16 OGBench+ExoRL cells. `scripts/launch_ogbench.sh` / `launch_exorl.sh` compose configs
identical to 26 reported runs. 444 tests pass.

**Why:** user asked (2026-09-30) for a referenceable clean branch; chose the name themselves
("density-fb-clean, or iclr-clean").
**How to apply:** use this branch for any paper-reproduction question; the old campaign
launchers (launch_tdjtab, launch_exorl_sqt...) are gone there. Uploaded to HF
amsks/factored-fb-checkpoints@iclr2027 as `code/density-fb-clean/` + thin
`code/density-fb-clean.bundle`. GitHub push still OWED: box has no GitHub credentials.
MetaWorld/ManiSkill table cells still have no provenance on this box.
