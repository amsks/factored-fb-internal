# Spec 17 — Delta (temporally-centered) successor measure

Status: WRITTEN 2026-08-06. Parent: spec 15 (dwell empirics), spec 16
(multistep), external review 2026-08-06 ("centered successor measure /
discounted net flow of occupancy" proposal).
Implementation specs (self-contained, for the implementing agent):
- `specs/17a-delta-measure-impl.md` — variant A (`measure_delta`, region
  net-flux target on the existing hit path).
- `specs/17b-delta-partition-impl.md` — variant B
  (`measure_delta_partition`, fixed-partition simplex head).
Runs head-to-head against the spec 16 multistep arms in one joint wave (§6).

Naming: "delta" = the target is the temporal difference Δh of the region
indicator. The external proposal calls the same object the *centered* or
*net-flow* measure; we avoid "flow" (collides with the FlowBC actor
everywhere in this repo) and "centered" (collides with T10
`centered_measure`, which centers over ACTIONS, not time — record: aux
0.823 (−8pts), only-mode scale runaway → anchored dead band 0.06–0.10).

## 0. One paragraph

Every failed occupancy cell regresses the raw region measure
`M(s,a,g) = E[Σ_t γ^t h_g(x_{t+1})]`, whose targets are dominated by the
action-independent dwell mass (ledger §9.16: batch-column phase 2 pins q_abs
to the dwell constant 43 and the actor decays; §9.15c: even a *correct* mass
anchor costs Spearman .68→.48). The delta measure replaces the target reward
with the temporal difference of the indicator:

    J(s,a,g) = E[ Σ_t γ^t ( h_g(x_{t+1}) − h_g(x_t) ) ]
             = (1−γ)·M(s,a,g) − h_g(x_0)            (exact identity)

At fixed s, `h_g(x_0)` is action-independent, so **argmax_a J = argmax_a M**:
a reparameterization of occupancy, not a new objective. What changes is the
supervision: a 50-step rest contributes ONE +1 entry event and ONE
discounted −1 exit event instead of 50 identical positive labels — dwell is
encoded as the temporal gap between entry and exit (γ^T(1−γ^L) telescopes to
exactly the discounted dwell mass) and never appears as a repeated target.
`J ∈ [−1, 1]` is bounded by construction. Degenerate-constant RELOCATION,
not elimination: the lazy constant for J is C* = E[Δh]/(1−γ) ≈ 0 (entries ≈
exits over the dataset), so collapse now looks like *zero net-flux*
(visible, actor-dead) instead of a plausible calibrated-looking dwell
constant (§5 NULL read).

## 1. Two variants (see 17a / 17b for the code-level contracts)

**A — `measure_delta`.** Minimal estimator change on the existing
ball-visitation cells: same columns, same bilinear M, same tpd
batch/hindsight; the target reward becomes the signed event indicator
`hit(x') − hit(x)` (outside→outside 0 · enter +1 · dwell 0 · exit −1),
non-absorbing, squared-diagonal residual. Isolates *centering alone* —
nothing else moves.

**B — `measure_delta_partition`.** The proposal's §6 simplex form:
one-stage, no goal-region queries, no hindsight, no phase switch. Fixed
partition φ(x) ∈ Δ^K (softmax-RBF over K dataset-drawn prototypes of the
hit-spec object blocks, per-block ε scaling); `delta_head` MLP([s,a,g]) → K
logits, p = softmax (validity free by construction — no hinge, no anchor);
CE fit to `sg[(1−γ)·φ(x') + γ·p̄(s', a⁺∼π(s',g), g)]`; actor maximizes
pᵀφ(goal) at frozen head params (= J readout + action-independent constant).
Zero-shot readout: z_r,k = r(proto_k); Q_r = (r(x₀) + (p−φ(x₀))ᵀz_r)/(1−γ).

Readout algebra for A (eval side): `M̂ = (h_g(x₀) + Ĵ)/(1−γ)`. Any
calibration (L1fix) or reward readout (L2) against a delta checkpoint MUST
apply this conversion — quote nothing raw.

## 2. Learnings applied (the checklist this design answers to)

1. **Actor Q-term never backprops into any critic head** (crl f7b623d) —
   enforced by a grad-hygiene TEST, not convention (17a §4 / 17b §5).
2. **No ortho** (fp-noortho 0.967 vs with-ortho 0.225) — neither variant
   adds it; B has no scale degeneracy to fix.
3. **No pessimism added** (T7) — stock min over the 2 heads only.
4. **No loss mixing / dose cliffs** (criticmix ×7, sgphi λ1→3) — A rides the
   existing measure-only slot; B is measure_only with its own head.
5. **Mass-conservation ordering risk** (§9.15c) — B's hard simplex is the
   untested cousin of the mass anchor; Spearman ≥ 0.65 is a pre-registered
   GATE (§6), not a hope. Fallback if it fails: bounded-unnormalized head.
6. **Grounding at actor-goal pairs is load-bearing** (§9.16k) — A runs
   hindsight-column arms alongside batch; B trains at value_goals and its
   actor reads its own goal's cell directly.
7. **Freeze is a diagnostic, not a fix** (§9.16j) — the phase-2 slots
   warm-start exactly like the §9.16 anchors so decay-vs-anchor is
   attributable; no reliance on actor_freeze_steps.
8. **0→10k shock** (dense wave) — eval_interval=10000 on all cd/sc arms.
9. **z-manifold / train-eval match** (rz law) — A: no z sampling changes;
   B: the actor consumes z_g built EXACTLY as eval builds it (φ of goal obs).
10. **Scale identities tested up front** (L1fix/L2 lessons) — the
    J = (1−γ)M − h identity is a unit test, not an eval-time surprise.
11. **ml_collections list trap** — every new knob is scalar/string.
12. **Group names lie** — every arm gets a distinct run_group; audit by
    recorded config.
13. **T10-b scale lesson** — J bounded / p simplex: the parameterization
    supplies what T10-b2's bolt-on anchors tried to (and don't re-add them).

## 3. What each variant can and cannot show (pre-stated)

- A vs one-step visitation, same slots: isolates centering alone. If
  delta-batch survives where visitation-batch collapsed, the §9.16
  batch-column death was level competition. If delta-batch still decays
  while delta-hindsight holds, grounding-at-actor-goals is independent of
  the level ⇒ fork (B) confirmed as the ingredient, centering as hygiene.
- A vs nstep (spec 16), same slots: two rival mechanisms for the same
  collapse — "dwell mass starves the bootstrap" (nstep moves it into the
  observed term) vs "dwell mass dominates the regression" (delta deletes
  it). Whichever holds the phase-1 anchor on cd/sc wins the slot.
- B: the cleanest estimator-vs-semantics deconfounder we have. Exactly
  equivalent to occupancy, dwell-free targets, dense vector supervision, no
  hindsight, one stage. If B floors on cube-single anyway, the remaining
  suspects are occupancy semantics itself (proposal §9) and vector-TD
  collapse (SF-feature-TD 0.09) — the entropy/calibration reads separate
  those two.
- NOT shown by either: delta fixes nothing about conjunctive per-cell
  sparsity (a cd joint cell is entered as rarely as a region is hit); B's
  bet is trunk generalization across cells, stated as such.

## 4. Failure signatures (pre-registered interpretations)

- A: `msq → 0` with flat loss + success → BC floor = **zero-flux collapse**
  (the relocated constant). Reported as: centering relocates, does not fix.
- A: success tracks the §9.16 batch-column decay curve ⇒ level competition
  was NOT the batch-column mechanism; grounding is (fork B strengthened).
- B: p → uniform (entropy → log K) = B's zero-flux analogue.
- B: calibrated p (MC cell masses match) + dead actor = the **semantic**
  verdict (occupancy ranking ≠ first-hit ranking) — proposal §9, our anneal
  iql-fade, now deconfounded from estimator failure.
- Either variant healthy on cs but not cd/sc: conjunctive sparsity, not
  dwell level, is the binding constraint ⇒ multistep/hindsight territory.

## 5. Anchors (from the record; do not re-run)

- cube-single: fp-noortho 0.967 · visitation-from-scratch floor ~0.004–0.5
  unstable (cellw0-noortho) · phase-2 batch ≈ 0.94.
- cube-double: phase-1 ≈ 0.549 · phase-2 batch collapse ≈ 0.358 (q_abs pins
  to dwell constant 43; d128 27 vs 43) · phase-2 hindsight ≈ 0.53.
- scene: phase-1 ≈ 0.484 · phase-2 batch ≈ 0.342 · phase-2 hindsight ≈ 0.927.
- BC floor ≈ 0.06–0.08 multi-object (stack-divergence audit).

## 6. Pre-registered joint wave (delta vs multistep, one launch)

All d50, standing recipe elsewhere; seed 0 (phase-1 warm ckpts
`exp_spec08/factored-fb/GCIQL-{CubeDouble,Scene}-Flow-Measure-FirstPassage-
NoOrtho-d50/{cd,sc}_fp_s0` @500k — the §9.16 anchor ckpts); 500k steps;
eval_interval 10000 on cd/sc arms (shock window), 50000 on cs;
eval_episodes 50; wandb entity amsks; tmux; XLA_PYTHON_CLIENT_PREALLOCATE=false.

cube-double (phase-2 slots, warm-start, measure_only visitation base):
- `d17_cd_delta_b`  delta + batch columns       (PRIMARY vs §9.16 batch collapse)
- `d17_cd_delta_h`  delta + hindsight columns   (vs hindsight anchor ≈ .53)
- `d17_cd_ns64_b`   nstep=64 visitation + batch (spec 16 arm, same slot)
scene (same three): `d17_sc_delta_b`, `d17_sc_delta_h`, `d17_sc_ns64_b`.
cube-single (sanity/anchors, from scratch):
- `d17_cs_delta`    delta visitation from scratch (vs floor ~0.1 / fp 0.967)
- `d17_cs_part`     variant B, K=256 protos       (PRIMARY: ≥0.5 = a
  one-stage occupancy-semantics method that trains an actor at all)
- `d17_cs_fpk64`    fp (absorbing) + nstep=64     (spec 16 hold gate: ≈0.9+)

Gates:
- G1 (A, cd/sc batch): success ≥ phase-1 anchor − 0.05 at 500k AND no
  §9.16-shaped decay after 100k.
- G2 (A): no zero-flux collapse (msq bounded away from 0, event-rate-scaled).
- G3 (B, cs): success ≥ 0.5 AND p-entropy < 0.9·log K at 500k.
- G4 (ordering, both): Spearman of MC region/cell masses ≥ 0.65 on the
  post-hoc calibration read (delta: through the J→M conversion).
- NULL: no arm passes its gate ⇒ both estimator repairs dead on arrival;
  the semantic hypothesis (first-passage + dwell head) becomes the program.

Post-wave eval battery (both eval axes):
- Goal-based: training-loop `evaluation/overall_success`.
- Multi-reward zero-shot: spec 14 L2 harness on the 500k ckpts — delta arms
  need the `M̂ = (h+Ĵ)/(1−γ)` conversion (harness flag, separate small
  change); partition arms use the z_r = r(protos) lookup (small standalone
  script). Quote conversion/identity conventions next to every number.

After collection: add every run_group to `scripts/results_map.py` and
refresh CSVs (`.venv/bin/python scripts/compile_results_csvs.py --refresh`).

## 7. Open questions carried forward

- B on cube-double (6-D joint protos, K=512?) — only if G3 passes on cs.
- Delta × nstep composition (telescoped k-step window = γ^k-anchored
  two-point target `γ^{k}h(x_{t+k}) − h(x_t)` + bootstrap) — refused in
  round 1; derivation is one line if A wins.
- Support-aware truncation (cut windows at the first object-motion frame) —
  spec 16 §7 territory; shared with the delta derivation note.
- Torch-stack port: never (stack decision: jax is the method artifact).
