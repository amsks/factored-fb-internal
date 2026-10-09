# FB-Manipulation ICLR port — design

**Date:** 2026-07-29
**Source:** `PAPER/rescuing-fb/two-phase-successor-measure-note.tex`
**Target:** `PAPER/FB-Manipulation-ICLR/`

## Scope

Write **only** Preliminaries, Method (theory), Experiments, Limitations. No
abstract, no introduction, no related work. Sections numbered from 2 so the
eventual introduction is §1 and nothing renumbers.

## Decisions taken in brainstorming

1. **Framing: method paper.** Density-ratio TD is structurally unanchored;
   replacing the point query with a finite region makes every entry a bounded
   squared residual. The theory→prediction→confirmation arc (§3.4 predicts a
   missing kernel normalization, §4.4 measures it as exactly N̄=0.0137) is the
   centerpiece.
2. **Privileged kernel owned in Preliminaries** as a stated assumption of the
   setting, not introduced quietly in Method.
3. **Theory rests on the singular immediate term alone.** `M = Q·R` is demoted to
   a remark — E5 refuted it pointwise (ICC 0.242/0.292 vs a 0.9 bar), and the
   method needs no factorization to be justified.
4. **Main-text experiments:** divergence mechanism + phase-1 2×2; L1 calibration;
   L2 novel-reward transfer; reward-z campaign; multi-environment negative.
5. **Multi-environment regression in the main text**, not buried in Limitations.
6. **Old goal-decode reward suite → appendix.** Two disjoint reward suites in nine
   pages is avoidable confusion.

## Template

ICLR 2027 style is announced but `iclr2027.zip` 404s as of 2026-07-29. Using the
official `iclr2026` style retargeted (year strings only). Swap when published.
Page limit 9, strictly enforced; these four sections target ~6–6.5.

## Structure

- §2 Preliminaries — successor measure, FB factorization, zero-shot protocol,
  **the abstraction assumption**, diagonal slice vs separated form.
- §3 Method — 3.1 singular immediate term · 3.2 the substitution · 3.3 two
  backups · 3.4 what the region form owes (the prediction).
- §4 Experiments — six claim-shaped subsections, each closing with a `takeaway`
  tcolorbox stating the finding *and* its scope limit.
- §5 Limitations.

## Correctness policy

Every number re-read from source (`results/spec10_torchfb/L1fix_*.json`,
`L2_*.json`, `~/exp_spec10/torchfb/_logs_backup/*.log`), with the 2026-07-28
five-agent audit corrections applied:

- L1 uses the **corrected** harness numbers, never the voided 66–215× table.
- `cube-single` phase 2 is **0.940**, not 0.955 (cross-attributed third seed).
- The 0.295 argmax figure carries its chance-level caveat (16 uniform actions).
- No claim that phase 2 "fails its calibration gate" or "does not compose" —
  both are contradicted by the corrected record.

## Audit findings applied to the source note

Twelve issues found; items 1–5 are claims the note makes *against* its own
corrected results (in §caveats and §exp-suite), items 6–7 are "not yet run"
claims for experiments that ran, item 8 is the 0.955/0.940 cross-attribution,
items 10–12 are consistency and portability. The ICLR port fixes all of them by
construction; the source note still needs 1–8 repaired separately.
