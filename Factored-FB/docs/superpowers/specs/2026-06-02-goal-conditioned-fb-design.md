# Spec — Goal-conditioned FB (V1 explicit-goal, V2 fixed-B/USFA) on state cube-single

**Date:** 2026-06-02 · **Branch base:** `main` · **Env:** state `cube-single-play-v0`, FB+FlowBC, PyTorch `.venv`, GPU.

## Question
GCIQL (≈0.68 success) beats FB (≈0.50) on cube-single. The headline difference is that **GCIQL is
explicitly goal-conditioned** — goal observation `g` into the value and actor, nonlinear `s×g`
interaction — whereas FB routes the goal through a learned low-dim vector `z=B(g)` and a **bilinear**
value `V=Fᵀz`. This spec gives FB the same privilege, two ways, and measures whether it closes the gap.

## Prior findings this builds on (established — do NOT re-derive)
From `2026-05-25-fb-Bg-cause-isolation.md` and `2026-05-25-privileged-z-findings.md`:
- FB fails at **transport/placement near the goal**, not reach/grasp.
- **Two ceilings**, both measured on the frozen state run (s3, ortho=1000):
  1. **`z=B(g)` is the bottleneck.** `R²(B→d)=0.20` (geometry largely absent from learned `B`;
     pixel-B=0.76). Near-goal Spearman(V,−d): `z=B(g)`=0.038 (chance) vs best-free `z*`=0.201.
  2. **The bilinear `V=Fᵀz` is also lossy.** `z*`≈0.28 ≪ F's featurizer geometry (`left_enc→d`
     R²=0.53).
- Near-goal data is abundant-but-proportionally-rare (~3.5% within 5cm); **no absolute data lack.**
- Coverage-balanced ρ reweight did **not** move `z=B(g)` or success → the binding constraint is
  `B`'s goal-encoding capacity and/or the forward path (rank/architecture), **not** the ρ measure.

**Mapping to this spec:** V2 removes Ceiling 1 (perfect goal geometry in `B` by construction);
V1 removes Ceiling 2 (lets the goal interact nonlinearly, breaking the bilinear bottleneck).
Either, both, or neither closing the FB↔GCIQL gap is a clean, publishable result.

## Baseline to beat
State FB+FlowBC, the paper's cube run: success ≈0.50 (GCIQL 0.68). Reference checkpoints/numbers
already exist for `L_dim=50`.

---

## Variants

All variants sit on the **FB+FlowBC base stack** (`agent=fb_flowbc`, `bc_coeff=3.0`, `ortho_coef=1000`,
`lr_b=1e-4`) — the FB we have been running on cube. All changes are **flag-gated and default-off**, so
with defaults the training path is byte-identical to current FB.

### Baseline (reference)
FB+FlowBC, `L_dim=50`. Already run; used as the reference and for the legacy reward-inference eval.

### A — L_dim=256 ablation (capacity control)
FB+FlowBC, `L_dim=256`, nothing else changed. Pure capacity bump to the forward map's state feature
`h=left_encoder(s)` (does **not** touch `z_dim` or task expressiveness). This is the
`2026-05-25` Experiment C1 lever. Tells us whether any V1/V2 gain is goal-conditioning vs just
wider features. **No code change — config only.**

### V1 — explicit goal-conditioning (attacks the bilinear ceiling)
Condition the **forward map and the actor** on the raw goal observation `g`, in addition to keeping
`z=B(g)`:
- `F(s,a,z,g)` and `π(a|s,z,g)`. The bilinear `Fᵀz` term is preserved (FB loss unchanged); `g`
  enters as an extra MLP input so the networks can model nonlinear `s×g` interaction.
- **Self-consistency:** the policy family is now indexed by `(z,g)`, so its occupancy legitimately
  depends on `g`. The `g` fed to `F`/π must be the **same (permuted) goal that produced `z`** in
  `sample_mixed_z` — thread the permuted goal tensor alongside `z`, do not re-sample.
- Training goal source: identical to current FB relabeling — `g` = the (permuted) next-state used for
  `z=B(g)` (50% of batch); for the random-z half, `g` is undefined → feed the same goal as `z`'s
  preimage is impossible, so for random-z rows feed a **zero/placeholder goal with a mask flag**
  (decision point #2).
- Code: extend `ForwardMap` and the FlowBC actor (`NoiseConditionedActor`) input dims by `goal_dim`;
  gate on `goal_cond=true`.

### V2 — fixed B = raw cube xyz (USFA / successor features, attacks the B(g) ceiling)
Replace the learned backward map with a **fixed featurizer** `B(s) := φ(cube_xyz(s))`:
- `φ` extracts the cube xyz from the observation (`CUBE_OBS_SLICE = obs[19:22]`, the scaled
  block_0 xyz per `agents/tdmpc2/agent.py`) and returns it as the 3-dim `B`. **No learned params on
  the B side**; `z_dim` on the B/F output becomes 3 (decoupled from `L_dim`).
- Learn only `F` (and the FlowBC actor) against this fixed `B`. The orthonormality regularizer on `B`
  is **disabled** (nothing to orthonormalize). The FB successor loss is otherwise unchanged.
- **Scaling consistency (must verify):** training reads scaled cube xyz from the obs slice
  (`(pos−[0.425,0,0])·10`); the eval task goal `goal_xyzs` is in real metres → apply the **same
  transform** to the goal before `z=φ(goal)`. Verify the slice + transform against the data as the
  first implementation step.
- Optional later: richer `φ` (e.g. `[effector_xyz, cube_xyz]`, or random-Fourier features). Start
  with raw cube xyz since that is exactly what defines the goal.

---

## Eval protocol
Give every variant the **explicit task goal** (the privilege GCIQL gets at eval) rather than the
current reward-inference `z`, which confounds goal-conditioning with data coverage in the 10k subsample.

| Variant | eval `z` | goal into nets |
|---|---|---|
| Baseline / A | `z = B(g*)` (embed the explicit goal **observation**) — matched control | no |
| V1 | `z = B(g*)`, plus `g*` fed to `F`/π | **yes** |
| V2 | `z = φ(scaled goal_xyzs)` — no reward inference, no goal obs needed | (φ is B) |

- **Matched control:** baseline and V1 both use `z=B(g*)`; the *only* difference is whether `g*` is
  also fed to the nets → isolates explicit conditioning as the single variable.
- **Legacy number:** also report baseline FB with the **current reward-inference** `z` for continuity
  with prior FB results (secondary, not the head-to-head control).
- **`g*` sourcing (decision point #3):** V1/baseline need a full goal *observation* with the cube at
  target. Candidate sources, in order of preference: (a) OGBench GC env info
  (`env.unwrapped.cur_task_info`) if it exposes a goal observation; (b) a canonical near-goal data
  state (the argmax-reward state from the relabel subsample). V2 needs none of this. **Verifying (a)
  is the first eval-side implementation step.**
- Wire via the existing `agent.eval_context(env, domain, task)` hook in `evals/ogbench.py:79` (FB
  currently falls back to `_infer_z`); the goal-conditioned variants implement `eval_context`.

---

## Run matrix
State `cube-single-play-v0`, FB+FlowBC, **10 seeds @ 1M steps** (match the existing FB cube campaign),
same launcher `scripts/agents/fb/state.sh` (extended with variant env-vars).

| # | Variant | `L_dim` | New flags | Purpose |
|---|---|---|---|---|
| 0 | Baseline | 50 | — | reference (existing) |
| 1 | A: capacity | 256 | — | does wider `h` alone help (C1 lever) |
| 2 | V1: explicit goal | 50 | `goal_cond=true` | nonlinear goal interaction (Ceiling 2) |
| 3 | V2: fixed-B/USFA | 50 | `fixed_b=cube_xyz` | perfect goal geometry in B (Ceiling 1) |

GCIQL's existing 10-seed numbers are the upper reference.

> **Decision point #1 (L_dim for V1/V2):** spec assumes V1/V2 at the base `L_dim=50` (so V1-50 vs
> baseline-50 and V2-50 vs baseline-50 are same-capacity comparisons), with #1 the separate capacity
> control. If instead V1/V2 should run at `L_dim=256` (to compare against the 256 ablation), it is a
> one-line config change — confirm in review.

## Metrics
- **Primary:** eval success rate (mean ± std over 10 seeds), per task and aggregate.
- **Secondary:** legacy reward-inference success for baseline (continuity).
- **Potential (not now):** the existing probes — `R²(B→d)` (trivially 1.0 for V2 — skip),
  privileged-z near-goal Spearman, value-by-phase — run on V1 if it moves success, to attribute *why*.

## Decision points to confirm before implementation
1. **`L_dim` for V1/V2** — 50 (base, same-capacity) [spec default] vs 256 (match the ablation).
2. **V1 random-z rows** — goal placeholder+mask vs feed `g=B⁻¹`-preimage (impossible) vs only
   condition on `g` for the goal-z half. Spec proposes placeholder+mask.
3. **`g*` source for V1/baseline eval** — OGBench `cur_task_info` goal obs vs canonical near-goal
   data state.

## Isolation guarantee
`goal_cond` / `fixed_b` default off; absent flags → byte-identical to current FB+FlowBC. The
`L_dim=256` ablation is config-only. Eval changes live behind `eval_context` and the new variants;
baseline keeps its current eval path.

## Implementation surface (for the plan phase — not exhaustive)
- `agents/fb/model.py` (`ForwardMap` input dim; fixed-B path), `agents/fb/agent.py` (`sample_mixed_z`
  thread goal; disable ortho for fixed-B), `agents/fb/flow_bc/*` (actor goal input).
- `configs/agent/fb_flowbc.yaml` (+ `goal_cond`, `fixed_b`, `goal_dim`), `configs/domain/cube_single.yaml`.
- `evals/ogbench.py` (`eval_context` for the goal-conditioned variants; scaled-goal `z=φ(goal)` for V2).
- `scripts/agents/fb/state.sh` (variant env-vars: `GOAL_COND`, `FIXED_B`, `L_DIM`).
- Tests: default-off byte-identical; V1 `(z,g)` pairing; V2 `φ`+scaling and ortho-disabled.
