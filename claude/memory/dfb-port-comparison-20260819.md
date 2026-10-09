---
name: dfb-port-comparison-20260819
description: "2026-08-19 cross-branch Density-FB inventory — exactly TWO implementations, both jax; the feat/metaworld-runs port (impls/critics/density_fb.py, pinned at bridge 542c058, soft-only) is DIVERGENT with two live bugs (bootstrap-goal mismatch; non-sg'd Q-mean centering with exact gradient cancellation); no torch Density-FB exists anywhere"
metadata: 
  node_type: memory
  type: project
  originSessionId: 973481bf-5a6b-4ead-8080-c40f063e1e9d
  modified: 2026-08-19T05:30:06.287Z
---

Measure track. Full audit in the 2026-08-19 session; canonical =
`gciql-fb-bridge:tools/wandb_mode_shim/gciql_flow.py` (density_loss now ~2711
after the spec-30a commits 8b2399f/e8e5642 — line anchors in older specs are
shifted).

**Inventory:** grep of every branch found exactly two Density-FB
implementations, BOTH jax: the canonical bridge one, and
`origin/feat/metaworld-runs:impls/critics/density_fb.py` (253 lines,
OGBench-style impls/ registry for MetaWorld play, + yaml + test). The port is
self-pinned at bridge commit **542c058** (pre-spec-22): g-indexed SOFT variant
only — no hard mode, no zero-shot/de-biased readout, none of specs 22–30.
There is NO torch Density-FB on any branch (agents/fb/measure = the LSIF-style
torch measure port, different objective; hitdirac control uncommitted) ⇒ the
agents/dfb torch rewrite is new work, not a duplicate.

**⚠ UPDATE 2026-08-19 (later same day): BOTH BUGS FIXED-CORRECTLY on the
branch (commit d8dce30, validated end-to-end + 30/31 tests incl. the
greedy-gradient-nonzero property; the 1 failure = x64 test-ordering
pollution, passes in isolation).** Bug 1: `boot["next_action_vgoal"]` drawn
at cond["value_goal"], stage_loss hard-errors without it. Bug 2: q_fn
returns (uncentered q, centered normalizer leg), centering consumed only
inside sg — canonical composition restored. New default-off knobs: W1
sarsa bootstrap (whose next_actions clamp FIXES the bridge's traj-end
off-by-one), W2 actor temp. One non-default delta: `measure_density_hard:
true` is now that branch's yaml default (deliberate, but its comment cites
stale numbers — soft wins cs/cd on the bridge). Port is now SAFE for
MetaWorld numbers, gated by its own cube-anchor P5 gate. Zero-shot readout
still absent there. Original findings below kept for the record:

**Two substantive divergences (found first, since fixed — kept as the
bug-class record):**
1. **Bootstrap-goal mismatch:** port's `sample_bootstrap` draws the TD
   bootstrap action from π(·|s′, **actor_goals**) (flowbc.py:83 → cond["goal"])
   while the target-F leg is indexed by **value_goals** — under its shipped
   yaml (value_p .2/.5/.3 vs actor_p 0/1/0) the goals differ a.s., so the
   target evaluates F̄(s′,a⁺,g_value) at an action from a DIFFERENT policy.
   Canonical uses value_goals for both (density_loss 2777/2859). Fix: condition
   sample_bootstrap on value_goal.
2. **actor_q_center recomposition:** port subtracts a NON-stop-gradded batch
   mean inside q_fn (q1 = q1 − q1.mean()) before the actor's sg-normalizer;
   the differentiated −Q.mean() cancels the greedy gradient EXACTLY when all
   rows' min selects the same head, attenuates otherwise. Canonical centers
   only the sg'd loss normalizer (1864–1878 old anchors). Port's docstring
   flags the placement but not the cancellation.

**Torch rewrite DONE same day (uncommitted):** `agents/dfb/` (hit_spec.py /
model.py / agent.py + configs/agent/dfb.yaml + train.py branch BEFORE the FB
`shared` whitelist block), mirroring agents/fb idiom; g-flagship scope only
(hard+soft, flow + ddpgbc, de-biased zero-shot readout as `zero_shot_q`;
`infer_z` deliberately raises — no linear E[rB] inverse exists, spec 28).
17/17 tests in tests/test_dfb_agent.py incl. float64 numpy loss parity (hard
AND soft), tabular (I−γP)⁻¹ readout at 1e-11, stop-grad discipline, and the
hard−soft = (1−γ)(I−k_soft) identity. Known carried caveat: eval min-over-
heads sub-normalization kept as shipped convention, quantified in a test.
Jax knob names preserved so arm configs translate. z/dual_centered/TDInfoNCE
seams left, not ported.

**Why:** stale ports keyed to old commits accumulate wiring-level divergence
that textual kernel identity hides — both bugs are in the WIRING (goal
plumbing, gradient path), not the copied kernels.
**How to apply:** when auditing a port, diff the cond/goal plumbing and
gradient paths, not just the loss math; check the pin commit in the docstring
first. Related: [[spec30-review-response-20260816]].
