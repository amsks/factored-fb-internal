# Multi-environment results log — scene, puzzle-3x3, cube-double

**Date:** 2026-07-30. **Scope:** the four OGBench state-based manipulation
environments for which a tuned vanilla-FB number exists in the literature
(`cube-single`, `cube-double`, `scene`, `puzzle-3x3`) — the spec-08 E8 set.

**Provenance:** every number in §1–§2 was re-read today from
`~/exp_spec08/factored-fb/<group>/<run>/eval.csv`, final row (`step=500000`),
not copied from prior prose. Where this log and
`PAPER/rescuing-fb/two-phase-successor-measure-note.tex` Table `tab:suite`
disagree, this file is the later read; as of this writing they agree.

---

## 1. Headline: native goal-conditioned evaluation, 500k steps

Overall success = mean over the five native GC tasks, in $[0,1]$.
The recipe is held fixed at the cube-single phase-1 winner; the **only**
per-environment quantity is the goal region (`measure_hit_spec`), which mirrors
each environment's own success criterion.

| env | anchor (GC-IQL bilinear d50) | phase 1 (reaching) | phase 2 (visitation) | Δ (p2 − p1) |
|---|---|---|---|---|
| `cube-single` | 0.928 | **0.967** | 0.955 | −0.01 |
| `cube-double` | 0.265 | **0.549** | 0.358 | **−0.19** |
| `scene`       | 0.281 | **0.484** | 0.342 | **−0.14** |
| `puzzle-3x3`  | 0.117 | 0.200 †  | 0.198 † | flat † |

† degenerate cell — see §3.

Seeds: 3 for phase 1 and the anchor, 2 for phase 2 (3 on `cube-single`).
All cells complete at 500k with 11 evaluation points each; no truncated runs.

### Per-seed values (verified 2026-07-30)

| env | cell | run dir | per-seed | mean |
|---|---|---|---|---|
| `scene` | anchor | `GCIQL-Scene-Flow-Bilinear-d50/an_sc_s{0,1,2}` | 0.260 / 0.268 / 0.316 | 0.281 |
| `scene` | phase 1 | `GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d50/{sc_fp_s0,sc_fp_s1,p1_sc_s2}` | 0.468 / 0.480 / 0.504 | 0.484 |
| `scene` | phase 2 | `GCIQL-Scene-Flow-Measure-BallVisit-FrozenPsi-d50/p2_sc_s{0,1}` | 0.332 / 0.352 | 0.342 |
| `scene` | **cube-only control** | `GCIQL-Scene-Flow-Measure-FP-NoOrtho-CubeOnly-d50/sc_fpc_s{0,1}` | 0.224 / 0.232 | 0.228 |
| `puzzle-3x3` | anchor | `GCIQL-Puzzle3x3-Flow-Bilinear-d50/an_p3_s{0,1,2}` | 0.056 / 0.148 / 0.148 | 0.117 |
| `puzzle-3x3` | phase 1 | `GCIQL-Puzzle3x3-Flow-Measure-FirstPassage-NoOrtho-d50/p1_p3_s{0,1,2}` | 0.200 / 0.200 / 0.200 | 0.200 |
| `puzzle-3x3` | phase 2 | `GCIQL-Puzzle3x3-Flow-Measure-BallVisit-FrozenPsi-d50/p2_p3_s{1,2}` | 0.200 / 0.196 | 0.198 |
| `cube-double` | 1-cube control | `GCIQL-CubeDouble-Flow-Measure-FP-NoOrtho-1Cube-d50` | — | 0.150 |

---

## 2. What these environments establish

### 2.1 Phase 1 exceeds its matched anchor on all four

+0.04 on `cube-single` (anchor already near ceiling), **2.1×** on `cube-double`,
**1.7×** on `scene`, nominally 1.7× on `puzzle-3x3` (does not count — §3).

The anchor shares the dataset, the actor class (flow-BC + one-step
distillation), the critic rank ($d{=}50$ bilinear) and the training budget.
**The reaching objective — not the representation, not the extraction — is the
only thing that differs.** The lead caveat of the work therefore weakens from
"one environment" to "four environments, one benchmark family, state
observations". It does not vanish.

### 2.2 Phase 2 does not hold the phase-1 level once off cube-single

`cube-double` 0.549 → 0.358 (−0.19) and `scene` 0.484 → 0.342 (−0.14), while
`cube-single` is flat (0.967 → 0.955) and `puzzle-3x3` is flat only because it
is degenerate.

Phase 2 still clears the anchor everywhere, so the curriculum is not broken —
but **the earlier reading that the visitation phase is free was drawn when
`cube-single` was the only completed column and does not survive the other
three.** Combined with the calibration section (where the phase-2 object is
signed, with 24–29% of its mass negative, and $\arg\max_a Q \neq \arg\max_a M$
on ~7 of 10 state–goal pairs), the honest summary is:

> **Phase 2 currently costs success and does not yet buy a usable measure.**
> Its justification rests on semantics that are not yet demonstrated.

### 2.3 The sparsity hypothesis is refuted

Pre-registered prediction: a denser off-diagonal (more hindsight positives per
row) should train better. The conjunctive-vs-single-object controls say
otherwise:

| comparison | conjunctive region | single-object control | ratio |
|---|---|---|---|
| `cube-double` (both cubes vs 1 cube) | 0.549 | 0.150 | **3.7×** |
| `scene` (full conjunction vs cube only) | 0.484 | 0.228 | **2.1×** |

The `cube-double` conjunctive region is ~30× **sparser** and wins by 3.7×.
**Sparsity does not predict success.** Whatever phase 1 is exploiting, it is not
positive-density in the hindsight matrix.

Measured off-diagonal positives per row at batch 1024:
`cube-single` 13.9 · `cube-double` 0.46 · `scene` 2.85 · `puzzle-3x3` 2.04.

### 2.4 Absolute numbers fall away from cube-single regardless of method

Phase 1 goes 0.97 → 0.55 → 0.48 → 0.20 as the goal region becomes a conjunction
over more objects, and the anchor falls with it. Whether that is the conjunction
itself, the resulting hindsight sparsity, or simply that these environments are
harder, **this table cannot separate** — the explanations are confounded across
the four environments. Pulling them apart needs a region-width ablation *within*
a single environment.

---

## 3. `puzzle-3x3` is a degenerate cell — do not read it as a win

Every phase-1 seed scores **exactly** 1.0 on task 1 and **exactly** 0.0 on tasks
2–5 (this is why all three seeds read 0.200 identically). The anchor does the
same with task 1 at 0.74 (per-seed 0.28 / 0.74 / 0.74). Phase 2 seed 2 reads
task 1 = 0.98, not exactly 1.0.

So the ~2× ratio is **one task's margin, not a spread over five**, and no method
here solves any of the other four tasks. The row should be reported as "all
methods fail" with a tie-break.

**Unresolved, and flagged as un-citable:** stock GC-IQL is published at
0.95 ± 0.01 on `puzzle-3x3` while our bilinear-$d50$ anchor reaches 0.117 — an
eightfold gap that no other row shows. Likeliest cause is per-environment
tuning: our anchor carries `bc_coeff=3.0`, selected on `cube-single`, to every
environment unchanged. Until that is separated from an architectural effect,
**treat the `puzzle-3x3` anchor as un-tuned, not as a measurement of what GC-IQL
can do there.**

---

## 4. Relation to the published FB numbers — protocol mismatch

Vanilla FB is reported at **0.496 / 0.026 / 0.128 / 0.048** on these four
environments (TD-JEPA paper), far below every column in §1. **These are not
comparable.**

| | our §1 table | published FB |
|---|---|---|
| task specification | native GC — goal handed to the policy | reward labels; $z$ inferred from $10^4$ relabelled samples |
| episodes/task | 50–100 | 10 |
| steps / seeds | 500k / 2–3 | 1M / 8 |

Native GC is the **easier** setting. Likewise the OGBench reference column
(GC-IQL 0.68/0.40/0.51/0.95, HIQL 0.15/0.06/0.38/0.12) is 1M steps, 8 seeds,
stock implementations — a different experiment. **The only column matched to
ours in budget, actor class and critic rank is our own anchor**, and that is
what the phase columns should be read against.

⇒ **No head-to-head FB claim is currently licensed on `cube-double`, `scene`, or
`puzzle-3x3`.** Protocol-matched zero-shot numbers exist only for
`cube-single` — which §5 may now change.

---

## 5. NEW (2026-07-30, unevaluated): torch scene + cube-double checkpoints

Four runs in the **FB codebase** (torch port, `agents/fb/measure/`) finished
today. These use FB's own **zero-shot relabel protocol**, i.e. exactly the
protocol §4 says we lack off `cube-single`.

| run | path | last ckpt |
|---|---|---|
| phase 1, scene | `~/exp_spec10/torchfb/p1_scn_s42` | 500k (06:44) |
| phase 2, scene | `~/exp_spec10/torchfb/p2_scn_s42` | 500k (09:56) |
| phase 1, cube-double | `~/exp_spec10/torchfb/p1_cdbl_s42` | 500k (06:25) |
| phase 2, cube-double | `~/exp_spec10/torchfb/p2_cdbl_s42` | 500k (09:12) |

All four have `final.pt` plus `step_{100..500}k.pt`. **Seed 42 only.**

Phase-2 scene config (from `.hydra/overrides.yaml`), carrying the adopted
rzK50 reward-z:

```
agent=fb_measure_flowbc  domain=scene  num_train_steps=500000
measure_mode=visitation  freeze_b=true  nhat_head=true
reward_z_ratio=0.5  reward_z_family=kernel
hit_spec='19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6'
warm_start_ckpt=.../p1_scn_s42/checkpoints/final.pt  seed=42
```

> ⚠ **No evaluations have been run on these.** `results/spec10_torchfb/`
> contains no `scn` / `cdbl` JSON. The checkpoints that would extend the
> protocol-matched FB comparison off `cube-single` are sitting unscored.

---

## 6. Box and campaign state (2026-07-30)

- **GPUs idle** — no compute processes.
- **No live tmux** for `mcons` (valid-measure) or `rewardz`. Both campaigns have
  landed: `results/spec10_torchfb/` holds `negmass_*`, `L1fix_*`, `L2_*` JSONs
  for arms mcA–mcE plus the d128 arms. Verdict owed as ledger §9.15b.
- **Never launched:** `cube-triple`, `cube-quadruple`, `puzzle-4x4`,
  `puzzle-4x5`, `puzzle-4x6` — 15 configs exist in `configs/gciql/`
  (bilinear / firstpassage_noortho / ballvisit per env), zero runs. This is 5 of
  the 9 envs in the full E8 suite.
- **Blocked:** the JAX-side zero-shot reward-family harness off `cube-single`
  (needs `env_geometry.py`). The §5 torch runs are a route around this.

---

## 7. Open actions, in priority order

1. **Evaluate the §5 checkpoints.** They are the only path to a licensed
   head-to-head FB comparison on a second and third environment, and they cost
   eval time only — the training is already paid for.
2. Add a second seed to the §5 runs before quoting anything from them.
3. Region-width ablation *within* one environment, to decouple conjunction from
   sparsity (§2.4).
4. Re-run the `puzzle-3x3` anchor with tuned `bc_coeff` before the 8× gap
   (§3) goes into any draft.
5. Decide whether `puzzle-3x3` stays in the headline table at all, given §3.

---

## Sources

- Run dirs: `~/exp_spec08/factored-fb/` (JAX, native GC),
  `~/exp_spec10/torchfb/` (torch, zero-shot protocol)
- Full experimental record: `PAPER/rescuing-fb/two-phase-successor-measure-note.tex`
  (`sec:exp-suite`, Table `tab:suite`)
- Variant ledger, both eval axes: `PAPER/rescuing-fb/PHASE2-LEDGER.md`
- Method derivation: `PAPER/rescuing-fb/fb-to-2phase.tex`
