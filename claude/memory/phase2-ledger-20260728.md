---
name: phase2-ledger-20260728
description: "PAPER/rescuing-fb/PHASE2-LEDGER.md is the single consolidated reference for all phase-2 work — every variant, both eval axes, all refuted pre-registrations, open threads"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a9f88cb-6be5-4d02-a66c-f8dcdd47169f
  modified: 2026-07-28T20:47:24.672Z
---

**Start any phase-2 brainstorming from
`PAPER/rescuing-fb/PHASE2-LEDGER.md`** (written 2026-07-28, 219 lines). It
consolidates what was previously scattered across specs 07–11, `DECISIONS.md`,
`two-phase-successor-measure-note.tex`, `analysis/rescuing-fb/**`, run-dir
`eval.csv`, and `/dev/shm` logs. Every number in it was re-read from source for
that document — several published tables were stale.

Sections: (1) what phase 2 is · (2) 9 variants × control quality + the 4-env
table · (3) measure quality from `smb_matrix_v1.csv` + the L1 test · (4) why the
two axes disagree · (5) **7 experiments that contradicted their own
pre-registration** · (6) where the record is stale · (7) in-flight non-negativity
campaign · (8) open threads.

The load-bearing facts it establishes:
- **Phase 2 costs success off cube-single** (cube-double −0.19, scene −0.14) and
  fails its calibration gate ⇒ currently it neither helps control nor buys a
  measure. See [[spec10-L1-calibration-fails-20260728]].
- **`visz` (separated phase 2) is the best-calibrated variant, `vis` (diagonal)
  the worst** (B1 Spearman +0.33/+0.37 vs −0.57). The non-negativity campaign
  runs on the torch stack, which uses `sample_mixed_z(train_goal_ratio=0.5)` so
  is *closer to* separated — but equivalence to `visz` is UNVERIFIED and worth
  pinning before generalizing.
- **Ordering, not sign, is the likely real defect** — Spearman was only
  +0.09…+0.35 independent of negativity. L7 (trained region columns) and L8
  (additive B(X)) are the designed fixes and are both unrun.

**Volatility handled:** `/dev/shm/spec10_logs/` gets wiped; all 52 logs/scripts
are now backed up to `~/exp_spec10/torchfb/_logs_backup/` (12M). Re-run that copy
after any new campaign.

**How to apply:** update this ledger after every phase-2 campaign instead of
letting results scatter again; it is the thing that made the stale `tab:suite`
and the unrecorded DDPGBC arm visible. See [[paper-gap-audit-20260728]].
