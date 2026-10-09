---
name: paper-gap-audit-20260728
description: "2026-07-28 audit of two-phase-successor-measure-note.tex vs all recorded evidence — 6 gaps CLOSED in the tex, remaining list of unrun/unwritten items; CSVs + DECISIONS.md stale since 07-22"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a9f88cb-6be5-4d02-a66c-f8dcdd47169f
  modified: 2026-07-28T15:45:15.551Z
---

Two-agent audit of `PAPER/rescuing-fb/two-phase-successor-measure-note.tex`
against every recorded result (specs 07–11, `analysis/rescuing-fb/*.json`, run-dir
`eval.csv`, `/dev/shm/spec10_logs`). The note's gaps were **missing controls and
stale tables**, not missing prose — it has no TODOs and a real limitations section.

## CLOSED in the tex this session (all verified against source before writing)

1. **§`sec:exp-replication`** (new) — torch/FB-house replication + the hitdirac
   reward-support control. See [[torchfb-measure-port-20260727]].
2. **§`sec:exp-calib`** (new) — spec-10 L1 pointwise calibration, 45% negative
   mass. See [[spec10-L1-calibration-fails-20260728]].
3. **§`sec:exp-p2`** — qualified the "~2% calibration error" as an *aggregate
   scale* check, blind to allocation and sign.
4. **`tab:suite`** — was STALE. Replaced with verified 500k finals:
   cube-double anchor 0.265 / p1 0.549 / p2 0.358; scene 0.281 / 0.484 / 0.342;
   puzzle-3x3 0.117 / 0.200 / 0.198; cube-single 0.928 / 0.967 / 0.955.
   Added the **phase-2 regression** reading (−0.19 cube-double, −0.14 scene) and
   a **puzzle-3x3 degeneracy** footnote (every seed = task1 1.0, tasks 2–5 0.0;
   anchor task1 0.74 — the "2×" is one task's margin, all methods fail).
5. **E5 argmax evidence** folded into §`sec:exp-calib`: Pr[argmax_a Q = argmax_a M]
   = 0.295/0.299, ICC(R|g) = 0.242/0.292. ⚠ The note's line ~128 claims this
   diagnostic is "not yet run" — it WAS run (spec 08 E5); that line is stale.
6. **Ridge/dense split** — note called it "the pending forensic"; E3 REFUTED it
   (s0 0.896→0.472/0.512/0.648, s1 0.128→0.432/0.464/0.432). Replaced with the
   table + "leading explanation now eliminated".

Builds clean under tectonic, no undefined refs.

## STILL OPEN — evidence exists but unwritten

- **E8 DDPGBC actor-isolation** (α-sweep, 1 seed): p1+DDPGBC 0.884/0.876/0.784 at
  α=0.3/1.0/3.0 vs anchor+DDPGBC 0.604/0.864/0.868 — **advantage reverses at
  α=3.0**. This is the arm that licenses or kills the standing caveat "all
  comparisons share one actor class". Stage 2 (3 seeds) never ran.
- **E2 frozen/frozen** 0.972 vs frozen/updated 0.940 — the "≤" half of the
  pre-registered read FAILS (mild actor drift in phase 2).
- **Sparsity controls**: cube-double conjunctive 0.549 vs 1-cube control 0.150;
  scene 5-way 0.484 vs cube-only 0.228 ⇒ off-diagonal density does NOT predict
  phase-1 success.
- **T3 (spec 11) failed to separate** phase-2 diag (0.440/0.512) from anchor
  (0.352/0.608/0.432) — spec 11 said that triggers suite redesign; not invoked.
- **B1 rank degradation**: azB Spearman 0.147/−0.004, 2B-trained 0.347/−0.197.

## NEVER RUN

E4 kernel normalization; E6 region/linear eval on zf0/zfan; spec-07 item 5
(position-only-B, d-sweep); spec-10 L3/L7/L8/L9/L10; 5 of 9 E8 envs
(cube-triple/quadruple, puzzle-4x4/4x5/4x6 — configs exist, never launched);
zero-shot harness off cube-single (blocked on `env_geometry.py` registry).

## HOUSEKEEPING DEBT

`results/summary_agents.csv` + `results/bridge_grafts.csv` last written
**2026-07-22**; every spec-07 phase-2 / spec-08 / spec-10 / spec-11 / torch
campaign is missing despite groups being registered in `scripts/results_map.py`.
`DECISIONS.md` last entry 07-27. Run the [[results-csv-pipeline]] refresh.
