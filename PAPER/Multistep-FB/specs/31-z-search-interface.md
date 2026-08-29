# Spec 31 — Zero-order z-search as the task-inference interface

Status: WRITTEN 2026-08-26, design user-approved (eval-only; encoder-seeded
CEM; search-first G29-3; linear battery + one general-utility cell). House
rules as specs 27–30: TDD, eval-only (NO training changes, NO new ckpts),
thin adaptation of the committed g293 harness with its control-gate
discipline, tests `.venv/bin/python -m pytest`, evals CPU/GPU jax, runs in
tmux, results polled directly (g293 watchers are unreliable), wandb amsks,
no co-author lines.

## 0. Provenance and evidence base

Idea: port the test-time interface of Soft FB (Bagatella, Rupf, Martius,
Krause, arXiv:2602.06769 — no public code; all details from the paper) into
Density-FB. Soft FB optimizes general utilities J^π = f(M^π) that do not fit
the linear FB reward interface by ZERO-ORDER SEARCH over policy embeddings:
sample ~1024 z from the compact ball, estimate each candidate's occupancy
without rollouts via the implicit model
M̂_z(·) ∝ E_{s0∼μ0, a0∼π(s0,z)}[F(s0,a0,z)ᵀB(·)ρ(·)], score f(M̂_z) on
dataset samples, take the argmax (random shooting; CEM as the upgrade).
Their license for search being well-behaved is max-entropy TRAINING
(z↦M is C^∞; ‖z‖<1 compact with (1−‖z‖) as adaptive entropy strength).

Why this fits Density-FB better than it fits FB: our w(s,a,z) =
softmax(F·B/τ) is ALREADY a normalized categorical occupancy over pool
samples — f(M̂_z) is a literal function of the weight vector, for any f,
no hand-normalization. And the search never ENCODES a reward at all, only
EVALUATES candidates through the model's own measure — which sidesteps the
pool-composition sensitivity that hard-aborted G29-3 attempt 2 (the
encoder's ẑ depends on the encoding pool; native .804 vs harness-pool
.704, task 3 .06 vs .48).

Prior art in-repo: the `oracle` stage of
`scripts/rescuing_fb/eval_spec30_g293.py` (lines ~484–533) already
implements the random-shooting core (256 √d-sphere candidates, J(z) =
mean over 16 reset states of w(s0, π(s0,z), z)ᵀr over a 2048-pool
subsample) as a DIAGNOSTIC. It has never produced a number (both G29-3
attempts died before it ran). This spec promotes that machinery from
diagnostic to PRIMARY interface.

Honest caveat to carry into the writeup: our z-indexed arms were NOT
trained soft; the smoothness/compactness license is weaker. De facto
substitute: candidates live on the √d-sphere, which is the actor's
training distribution (`z_rand_frac=.5` sphere rows), so search never
leaves the conditioning distribution. This is tested (§3 triad +
corr(J, return)), not assumed.

## 1. The interface

New module `scripts/rescuing_fb/z_search.py`:

```
z_star, diag = z_search(f, agent, pool, init_obs, seed_z=None,
                        n_init=512, n_round=256, rounds=3,
                        elite_frac=0.1, rng=...)
```

- **f** is any utility on pool weights. Linear reward: `f(w) = w @ r_sub`.
  General utilities: any callable of w (no gradient needed — zero-order).
- **Objective**: `J(z) = mean_{s0 ∈ init_obs} f( w(s0, π(s0,z), z) )` with
  w = min-over-heads of the two softmax rows (the actor/eval convention,
  double_q min — spec 30 Item 4 language applies). The J evaluator is the
  g293 oracle's J REFACTORED OUT of `eval_spec30_g293.py` into this
  module and imported back — not rewritten (byte-identical numerics is
  the control gate).
- **Candidates, round 0**: `seed_z` (the centered dual encoder's ẑ,
  computed through the ckpt's OWN frozen `ZEvalPoolState` pool — never a
  harness-built pool, per the G29-3 attempt-2 diagnosis) + (n_init−1)
  iid √d-sphere samples.
- **CEM**: `rounds` refinement rounds; elites = top `elite_frac` by J;
  refit diagonal Gaussian on elites; resample `n_round`; re-project every
  sample to ‖z‖=√d (the standing norm invariant — CEM lives on the
  sphere, not in the ball).
- **diag** (always returned): J(ẑ), J(z*), cos(ẑ, z*), per-round
  J_max/J_median trajectory, full round-0 (z, J) table for the
  corr(J, return) cell. J(ẑ) vs J(z*) IS the encoder-vs-search
  comparison; no separate machinery.

## 2. What J integrates over

- **Init states**: env-reset states (n=16, fixed seeds), exactly as the
  g293 oracle — `reset()` only, no stepping, offline claim intact.
  Ablation cell: dataset episode-start states instead (expected no-op;
  one family suffices).
- **Integration pool**: 2048-point subsample of the dataset marginal
  (g293 convention). Pool-size sweep M∈{256, 1024, 8192} FOR THE SEARCH
  PATH: the search still integrates over a pool, so its pool-robustness
  needs its own evidence. Stated hypothesis: the pool enters only as an
  integration measure, not as an encoding basis, so z* sensitivity
  should collapse relative to the encoder's (quantify as cos(z*_M, z*_M′)
  and rollout deltas; compare against the encoder's ẑ(pool_A) vs
  ẑ(pool_B) numbers from the gate rerun).

## 3. The battery (search-first G29-3)

New harness `scripts/rescuing_fb/eval_spec31_zsearch.py`, adapted from
`eval_spec30_g293.py`. Gate structure kept, but the gate is ONLY the
native re-roll sanity (encoder path through the ckpt's own pool ≈ .8);
the harness-pool encoder cell moves from gate to reported diagnostic.

- **Ckpt**: the on-disk `ra2_cs_centdual` (spec 30 wave-1 winner, .816).
- **Families**: spec-30 P3 list unchanged — region/dense/hard, max/sum
  mixtures over two goals, target−λ·avoid, per-object conjunctions,
  negatives, SHIFTED copies (r+β). Two arms per family, rollout-validated:
  encoder ẑ (comparison) and search z* (primary). Spec-30 interpretation
  triad retained verbatim: closed-form fails + search succeeds ⇒
  task-encoder failure; both fail + reconstruction good ⇒ policy-family
  coverage failure; reconstruction fails ⇒ task-basis failure.
- **Affine invariance by construction**: with a fixed candidate set and
  simplex-normalized w, argmax_z(w@(αr+β)) = argmax_z(w@r) for α>0 (β
  adds the constant β·1). Precision: the shipped min-over-heads w is
  SUB-normalized (spec 30 Item 4), so β contributes β·‖w(z)‖₁, which
  varies weakly with z — invariance is EXACT for single-head or
  renormalized w, near-exact under the min convention. Harness: assert
  bit-identical z* on the renormalized readout; on the min readout,
  report whether any r+β cell flips the argmax (expected: none). The
  encoder's shift-instability (spec 30 §6.6.1c) vanishes by
  construction either way — this is the headline structural claim.
- **General-utility showcase (ONE cell)**: distribution matching on cube
  position — f(w) = −KL( Π(w) ‖ target uniform over a chosen region ),
  Π = projection of pool weights onto cube-xyz bins. Searched, rolled
  out, reported as a coverage/occupancy metric + qualitative traj plot.
  NO success-rate claim. This is the capability the encoder cannot
  express at all (its input type is (u_j, r_j) samples).
- **Model-exploitation risk, made a measured quantity**: J is the model's
  own belief; search can exploit model error (adversarial z*). Report
  corr(J, actual rollout return) across the round-0 candidate table for
  ≥2 families (the action-ranking-agreement analog). Search winning J
  but losing rollouts = the coverage-failure branch of the triad, not a
  harness bug.

## 4. Code layout, tests, ops

- `scripts/rescuing_fb/z_search.py` — candidate gen, J evaluator
  (refactored from g293), CEM loop. Pure numpy/jax over the existing
  agent API (`sample_actions_with_z`, `network.select('density_f'/'
  density_b')`). No agent/train.py changes.
- `scripts/rescuing_fb/eval_spec31_zsearch.py` — battery harness, stages
  `{gate, families, utility, sweep, all}`, `--quick` smoke.
- Tests (new `tests/test_z_search.py`):
  1. CEM recovers a planted argmax on a synthetic quadratic J on the
     sphere (tolerance in cos);
  2. exact affine-invariance of round-0 argmax with a fixed candidate
     set (single-head w), and a quantified bound under the min
     convention;
  3. encoder seed present in round-0 candidates;
  4. ‖z‖=√d preserved through every refit round;
  5. refactored J == the inlined g293 J on a pinned random ckpt/batch
     (the numerics control gate).
- Ops: run in tmux; results → `results/spec31_zsearch/`; poll the dir
  (no watcher trust); afterwards add the group to `scripts/results_map.py`
  and `.venv/bin/python scripts/compile_results_csvs.py --refresh`.
- Cost: K≈1k forwards × 16 states per family — minutes per family on
  GPU; the whole battery is an evening, not a campaign.

## 5. Deliverables and decision rules

Deliverables: `families.json` (per-family encoder-vs-search J + rollout),
`sweep.json` (pool-size robustness), `utility.json` (KL-matching cell),
`corr.json` (J-vs-return), spec-31 section in DENSITY-FB-EXPLAINED.md,
paper paragraph positioning vs Soft FB (cite bagatella2026softfb: they
search because general utilities break linearity; we search because
NORMALIZATION breaks linearity — same cure, different disease; our
categorical w makes f(M̂_z) a literal finite sum).

Decision rules:
- Search ≥ encoder on the family battery AND pool-robust ⇒ the search is
  the shipped reward→z interface; encoder demoted to warm-start +
  ablation row. G29-3's general-interface claim is licensed through the
  search path.
- Search ≈ encoder, both good ⇒ ship encoder (cheaper at test time),
  search kept as the general-utility extension + robustness argument.
- Search fails where encoder fails + reconstruction good ⇒ policy-family
  coverage failure (triad branch 2) — a finding about the z-indexed
  policy family, not the interface; report as such.
- corr(J, return) low ⇒ flag model-exploitation; CEM capped at round-0
  (random shooting) in shipped numbers if refinement measurably
  overfits J.

Out of scope (explicitly): soft/max-ent training arm (future spec if the
non-soft landscape proves ragged); FRE/DeepSets encoder comparison (spec
30 P3 rule unchanged — mandatory only if arbitrary-reward prompting
becomes a headline claim); any training-side knob.
