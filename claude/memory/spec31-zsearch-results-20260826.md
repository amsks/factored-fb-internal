---
name: spec31-zsearch-results-20260826
description: "2026-08-26 spec31 FULL battery (eval-only, ra2_cs_centdual 500k, 1 seed): soft-FB-style zero-order z-search CANNOT replace the encoder — in 7/8 families CEM (1280 cands, sphere) returns the SEED exactly (encoder is an isolated J-spike, landscape flat: dense max-excl-seed .1597 vs seed .2400 vs median .1548); the one family where search moved (negavoid) it exploited model error (J up, rollout .20→.10); utility cell LOST to the goal baseline (cov .55 vs .62) ⇒ encoder stays the interface, search = validator/diagnostic"
metadata: 
  node_type: memory
  type: project
  originSessionId: 74028a31-0c8e-407a-a040-e7761054a936
  modified: 2026-08-26T19:32:51.042Z
---

Measure track, spec 31 (PAPER/Multistep-FB/specs/31-z-search-interface.md +
31a impl plan). Idea源: Bagatella soft FB (arXiv 2602.06769, no public code)
— test-time zero-order search over z for general utilities. Ported eval-only
onto [[spec30-wave1-finals-20260819]]'s ra2_cs_centdual ckpt.

**Code (committed ed16fc8..b1575c5 on gciql-fb-bridge, 7 commits):**
`scripts/rescuing_fb/z_search.py` (numpy CEM core + lazy-jax weights factory,
8 tests in tests/test_z_search.py), `eval_spec31_zsearch.py` (6-stage
battery), g293 refactored onto the shared core (fam_defs module-level,
oracle J via make_weights_fn) — eval_spec30_g293.py is now COMMITTED.
Artifacts: results/spec31_zsearch/ (gate/families_summary/sweep/utility/
corr jsons). Final whole-branch review (fable): mergeable; deferred minors
listed in the review, none load-bearing.

**Findings (1 seed, 50 eps/cell):**
- Gate PASS: native .804 / frozen-pool z-api .828 vs ref .816.
- 7/8 families: z* == encoder seed EXACTLY (cos 1.000, J_seed==J*). The
  encoder direction is an isolated spike in a FLAT J landscape over the
  √50-sphere (random/CEM candidates: max ≈ median ≈ .155 vs seed .24 on
  dense_g1; CEM refit rounds never re-approach the seed). 50-d sphere
  sampling has no gradient signal toward the task cone.
- negavoid (only family where search moved, cos .008): J −.0367→−.0302 but
  rollout success .20→.10 — MODEL EXPLOITATION exactly where J varies.
- corr(J, return): headline spearman −.62/−.36 is a flat-cluster noise
  artifact (11 near-identical-J random z's + the seed); macro-scale J is
  RIGHT on tma10 (seed ret 4.3 vs ~0) but dense_g1's disc-return proxy
  barely discriminates (~10 for everything — dense reward floors high).
- Utility showcase (KL-matching to uniform-over-region): search LOST to the
  goal-seeking contrast arm (region-bin coverage .547 vs .625, KL 9.56 vs
  8.87) — the z-family likely contains no "spread over region" behavior.
- Affine cells: renorm-convention invariance held EXACTLY everywhere
  (theorem assert, 8/8); min-of-heads convention FLIPPED the argmax under
  shifts in 4/8 families ⇒ sub-normalization is a real hazard; renormalized
  readout should be the scoring convention for any search-style use.
- Sweep/init-ablation trivially stable (z* = seed, which doesn't depend on
  the integration pool or init states).

**Why:** soft FB's search license is max-ent TRAINING (smooth z→M, compact
ball); our non-soft ckpt lacks it, and the finding confirms the license is
load-bearing — porting the test-time method alone does not transfer.

**How to apply:** the centered dual encoder remains the reward→z interface
([[spec30-wave1-finals-20260819]] G29-3 still the critical path, incl. the
pool-provenance gate fix); the z-search machinery is kept as (a) an
encoder VALIDATOR (seed unbeaten among ~1.8k candidates is evidence the
encoder is near-optimal in-family), (b) the exact-affine re-ranking table,
(c) a diagnostic for future soft-trained arms — a soft/max-ent TRAINING arm
is the prerequisite before retrying search-as-interface. NO results_map
entry (eval-only, no new run group; ra2_* registration still owed by
spec30). Paper positioning: "same cure, different disease" framing DIED —
replace with "the smoothness license is what soft training buys".
