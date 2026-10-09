---
name: hx-cd-hard-collected-20260816
description: "2026-08-16 hx_cd_hard wave collected — hard cube-double flow 3-seed .680±.070; tex cell had unsourced .667±.072, fixed; s0_re restore-continuation was contaminating the CSV group (merged-wave hazard), per-run exclude_runs mechanism added to the compiler"
metadata: 
  node_type: memory
  type: project
  originSessionId: 973481bf-5a6b-4ead-8080-c40f063e1e9d
  modified: 2026-08-16T21:04:06.959Z
---

Measure track. The `hx_cd_hard_{s0,s1,s2}` wave (group
`GCIQL-CubeDouble-Flow-Measure-DensityHard-d50`, ~/exp_hardx) completed 500k:
**.692 / .700 / .648 → .680 ± .070 (95% t-CI)**. Soft cd stays the winner
(.736 ± .139, bold unchanged); hard ≈ soft within CIs on cube-double, so the
cs/scene "hard helps" story does NOT strengthen here — whoever owns spec 21's
dirac-mix gate ("hard win is 1-seed") should re-read it against these seeds.

**Fixes applied (working tree, 2026-08-16, uncommitted):**
- `density-fb.tex` tab:main hard-cd flow cell: was **.667 ± .072 with no
  recorded provenance** (caption listed per-seed values for every hard cell
  except this one) → now .680 ± .070 + seeds added to the caption.
- `hx_cd_hard_s0_re` is a RESTORE-CONTINUATION of s0 (`restore_epoch=100000`,
  `restore_path=…/hx_cd_hard_s0`, ran to 600k, config delta only the two
  post-spec24 default-off z knobs). The CSV compiler had merged it as a 4th
  seed (.768 → group mean .702) — the exact merged-wave hazard from
  [[spec22-antmaze-null-20260813]]. Added **per-run exclusion** to the
  pipeline: `exclude_runs=(names,)` + `exclude_runs_reason` in a
  `results_map.py` entry; `compile_results_csvs.py` drops them and writes a
  `DROPPED-RUNS` line to compile_audit.txt. Verified: row back to 3 seeds.

⚠ Pattern to reuse: any `*_re` / restarted run sharing a wandb group will
contaminate its group's CSV stats — check `flags.json` `restore_path` before
trusting a per-seed column, and exclude via `exclude_runs`.

45 UNMAPPED groups remain in compile_audit.txt (standing spec 27 Q5 debt).
