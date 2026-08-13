# Spec 07 — capability on top of first-passage FB: task-prior actor + renewal measure

*2026-07-22. Pre-registered BEFORE implementation, per program discipline. Builds directly on
the FirstPassage-NoOrtho win (`GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50`: 0.956 /
0.964 / 0.980 → mean 0.967, 3 seeds @500k; memory `firstpassage-measure-works-20260722`).
Companion docs: `../first-passage-derivation.tex` (what the learned object is),
`../phenomenon.tex` §7–§8, spec 06 (esp. the PROGRAM RE-CENTER and its "occupancy
completion" bullet, which this spec turns into a concrete item).*

**User decisions (2026-07-22, binding for this spec):**
- **Target = capability**: reward semantics at test time (dense, composite, region, linear
  rewards — beyond multi-goal Dirac generalization) PLUS a genuine successor-measure object.
- **The independent-eval verification of the 0.967 result is NOT bundled here** — it runs
  separately. Every read below is provisional until that gate lands; nothing in this spec
  is claim-grade before it.
- **Amendment to the spec-06 re-center**: "multi-goal GPI is out of scope" is superseded —
  GPI re-enters as the *inference-time composition layer* for composite rewards, sitting on
  top of the measure-native critic (which itself is unchanged). The re-center's
  measure-native definition is still satisfied: the trained object is the M(s,a,s⁺) matrix,
  the actor consumes only readouts of M, task entry is zero-shot via the z/B readout.

Environment invariants (do not deviate): run in tmux; wandb entity `amsks`; jax venv
`/dev/shm/.venv-jax` (volatile — recreate if wiped), repo `.venv` for torch + tests;
`XLA_PYTHON_CLIENT_PREALLOCATE=false`; datasets under `/dev/shm/factored-fb/datasets` get
WIPED — restore from `~/bridge_ckpt_backup/datasets/`. After every campaign: add the group to
`scripts/results_map.py`, run `.venv/bin/python scripts/compile_results_csvs.py --refresh`,
add a DECISIONS.md entry with the pre-registered read applied verbatim.

---

## 0. The design principle (why the shape of this spec is forced)

The winning critic is `Q_FP(s,a,g) = E[γ^{T_g−1}]`, factorized as `F(s,a)ᵀB(g)/√d` — a field
of goal-specialist first-passage values, not a successor measure and not reward-linear
(`first-passage-derivation.tex` §7). Two structural facts dictate the design:

1. **The critic has no off-Dirac degrees of freedom.** `Q(s,a,z) = Fᵀz/√d` is linear in z by
   architecture, so its value at any mixture or random z is fully determined by the Dirac
   columns. Moreover first-passage values do NOT compose linearly over goal sets
   (`E[γ^{T_{A∪B}}] ≠ E[γ^{T_A}] + E[γ^{T_B}]`), so grounding the critic at `z_mix` with a
   mixture-task FP target would fight the Dirac columns *through* the linearity.
   ⇒ Capability work goes into the **actor** (item 1) and **inference** (items 2–3), not the
   critic loss. (Item 4 is the cheap falsification probe of this argument.)

2. **The successor measure factorizes through first passage** (renewal identity, potential
   theory; the spec-06 "occupancy completion" bullet):
   `M^π(s,a,B_ε(g)) = E[γ^{T_g−1}] · R^π(g)`, where `R^π(g)` = expected discounted revisit
   mass of the ball after first entry — **a function of the goal alone**. All the
   action-dependence of the measure lives in the FP factor (learned, works); the
   mixing-dominated 82% of the γ-mass (phenomenon Prop. 2) lives in the action-free renewal
   factor. Learning R as a separate goal-only head reconstructs the measure at provably zero
   control risk, and its product form explains the whole failure record in one line: joint
   occupancy TD spends tower capacity on R, which owns the mass, while control lives only in
   the FP factor.

Reward readout that follows: for reward samples `{(g_j, r_j)}`,
`z_r = matchnorm( mean_j [ r_j · R(g_j) · ψ(g_j) ] )` — FB's moment inference, renewal-
weighted. The plain moment `z_r = matchnorm(mean_j r_j ψ(g_j))` is the mandatory ablation.

Honest caveat to carry in every writeup: the reconstructed object is a *measure field* over
goal-specialist policies (each column's policy is π_g), not one policy's measure. The
single-policy object M^μ is item 2's second deliverable (fully offline, MC-supervised).

## 1. `task_prior_actor` — actor trained across a z-prior in the FP corner (the capability arm)

**What.** New knob on `tools/wandb_mode_shim/gciql_flow.py`, default-off ⇒ the winning FP
cell is bit-identical. When on, ONLY the actor's conditioning z changes; the FP matrix
critic loss, its bootstrap, and its rng stream are untouched.

- `task_prior_actor ∈ {'dirac' (default), 'mix'}`, `task_prior_actor_probs = [0.6, 0.2, 0.2]`
  (kept equal to v1's for comparability — user question (c) resolved to comparability),
  `task_prior_actor_mix_k = 2`.
- Per-sample z for the actor (flow BC target sees goals as now; the actor's q term and
  distillation consume the sampled z):
  - **Arm 0 Dirac (p=0.6):** the existing actor-z path, unchanged (ψ of the actor goal,
    sg'd — v0/FP-cell parity; D8 trajectory-future split preserved).
  - **Arm 1 Mixture (p=0.2):** K goals from batch next-states, w ~ Dirichlet(1),
    `z = Σ w_k ψ(g_k)` renormalized to match-data norm with sg SCALE (live direction is
    irrelevant here — see gradient discipline). No reward, no mask: **an actor consumes only
    z and the critic; the arms are purely a z-distribution.** This is the decisive
    simplification vs v1 — none of `_taskprior_vc_loss` is used.
  - **Arm 2 Random (p=0.2):** z uniform on the sphere × sg[mean‖ψ(s')‖].
- **Gradient discipline:** the actor's z is FULLY sg in every arm (matches the current FP
  cell where `z_g = sg[ψ(g)]`, `gciql_flow.py` ~425/467). Unlike v1 there is no vc-loss
  needing live ψ-grads through z — ψ trains only through the FP matrix loss, exactly as in
  the winning cell. This kills v1's hardest bug class (spec 06 item 4 gradient-discipline
  corrections) by construction.
- **rng discipline:** all new randomness (arm draw, mixture indices/weights, sphere z) comes
  from a split consumed ONLY inside the knob-on branch, so knob-off is bitwise identical
  (follow the param-tree-preserving pattern of spec 06).
- **Assert lattice:** requires `measure_only + value_arch='bilinear' + flow actor + state
  obs`; mutually exclusive with `task_prior='mix'` (the scalar-corner knob — different
  grounding, keep the namespaces disjoint); refuse with `measure_first_passage=false`
  (occupancy floors — there is nothing to extend).

**Known risk (pre-registered):** the FP matrix's bootstrap actions come from this same actor
conditioned per-column on Dirac z's; if off-Dirac actor training degrades the Dirac policy,
critic targets degrade with it (coupled). This is exactly what the non-regression band
detects.

**Tests before launch** (`tests/test_gciql_fp_taskprior_actor.py`):
(a) default knob-off bitwise ≡ the FP-NoOrtho cell update on a fixed batch;
(b) knob-on: the critic loss value on a fixed batch is EXACTLY unchanged (actor-only knob);
(c) the actor's z is fully sg in all three arms (jax.grad w.r.t. ψ-params through the actor
loss = 0 for the z path);
(d) mixture renorm hits match-data norm; arm frequencies match probs;
(e) assert-lattice refusals.

**Launch.** 3 seeds, 500k, d50 (matched to the breakthrough config; d512 only as a follow-up
rung), group `GCIQL-State-Flow-Measure-FP-NoOrtho-TPActor-d50`. Config
`configs/gciql/cube_single_state_measure_fp_noortho_tpactor_d50.yaml`.

**Pre-registered reads (binding; all via item 3's eval):**
- **Dirac non-regression ≥0.90** (anchor 0.967; user question (b) resolved to 0.90). Below
  0.90 ⇒ the actor-coupling risk is real — retry with probs 0.8/0.1/0.1 ONCE, else the arm
  is dead and capability rests on inference-only (item 3 on the frozen 0.967 ckpts).
- Two-goal max-mixture WITHOUT GPI: ≥0.60 success / 0.30–0.60 partial / ≤0.30 dead
  (anchor: v0-no-GPI 0.260).
- Two-goal max-mixture WITH GPI: expect ≥0.75 (anchor: GPI@d512 scalar corner 0.752).
- Region goals: ≥0.672 (non-regression vs T6).

**Comparison arm (zero new code):** `cube_single_state_rgfb_v1_taskprior_d50.yaml` as
already spec'd (06 item 4) — the "reward semantics via scalar grounding, no measure"
baseline. The FP+TPActor cell must match or beat it on every item-3 task family, else the
scalar corner remains the capability recommendation and the FP corner's claim stays
control-only.

## 2. Renewal head — the successor measure, reconstructed (CPU-friendly, frozen ckpts, run first)

**What.** New script `scripts/rescuing_fb/renewal_head.py` (jax CPU or 1 GPU; sibling of
`measure_readout_head.py`, spec 06 item 8 — reuse its checkpoint-restore and feature-dump
plumbing). No agent surgery; towers are structurally frozen. Two deliverables:

**2a. The renewal factor and the field measure.**
- MC targets, pure offline, no TD, no bootstrap, units pinned by construction (satisfies
  phenomenon §6's binding scale rule): for test goals g sampled from dataset next-states,
  find every trajectory entry into `B_ε(cube(g))` (ε = 0.4 obs units, the 4cm radius —
  decimeter scaling, verify against `cube_env.compute_observation` as T1 did); target per
  entry `R_tgt = Σ_{t≥t_entry} γ^{t−t_entry} 1[cube(s_t) ∈ B_ε]`; average over entries.
- Head: scalar MLP (2×256 GELU) — TWO input variants, both cheap, both reported:
  `R(sg[ψ(g)])` (the deployable one — needs only the trained ψ) and `R(cube(g))`
  (position-only — the interpretable control).
- Reconstruction: `M̂(s,a,g) = sg[Q_FP(s,a,g)] · R(g)` — by construction R cannot touch the
  control ordering (action-free scalar per column).
- **Calibration eval (the paper figure):** held-out episodes (last 100), empirical
  discounted-occupancy targets on an (s,a)×g grid; Spearman + calibration slope, broken down
  by horizon bucket K≤5 / 5–20 / >20 — same protocol as spec 06 item 8, so the two heads
  (occupancy refit vs FP×R reconstruction) are directly comparable.

**2b. The honest single-policy measure M^μ (appendix object).**
- Both factors MC-computable offline under μ: `FP_μ(s,a,g)` by within-trajectory
  first-hit regression (target γ^{T−1}, censored-as-zero beyond the episode — γ=0.99 makes
  the censoring bias small only for T ≪ episode length; REPORT the censoring fraction and
  restrict claims to the uncensored-dominated region), times `R_μ(g)` from 2a.
- Product = a bona fide successor measure of the behavior policy: density queries, OPE of
  arbitrary rewards (demo: OPE of a random dense reward vs MC ground truth on held-out
  episodes — the spec 06 item 8 panel, now through FP×R), horizon-bucketed calibration.
  This converts T8's dead SARSA arm into the object it was always right for: evaluation
  semantics without improvement.

**2c. Renewal-weighted task inference (feeds item 3).** Export
`infer_z(reward_samples, weighting ∈ {'renewal', 'plain'})`:
`z_r = matchnorm(mean_j r_j · R(g_j) · ψ(g_j))` vs `matchnorm(mean_j r_j ψ(g_j))`.

**Pre-registered reads:**
- Control untouched: structural (nothing retrained; assert checkpoint hash unchanged).
- Fidelity(FP×R reconstruction) ≥ fidelity(spec-06-item-8 occupancy refit on the same
  frozen towers) on Spearman in EVERY horizon bucket ⇒ print "the measure was never lost —
  it factorizes; the failure was estimating the product jointly." If it loses in the K>20
  bucket only, print exactly which bucket and keep both heads.
- Flat-R side-prediction (diagnostic, from the renewal framing): R̂ variance over goals
  should be LARGE under μ (re-grasping revisits) — this is the quantitative form of "R owns
  the mass." Report var[R̂]/mean[R̂]².
- Position-only R within noise of ψ-input R ⇒ the renewal factor is a cube-position
  functional, as the quotient-goal-space story predicts (derivation §7).

## 3. Zero-shot reward-capability eval (`scripts/rescuing_fb/eval_zeroshot_rewards.py`)

One harness, run against (i) the frozen 0.967 checkpoints (inference-only — isolates what
items 2c + GPI buy WITHOUT new training) and (ii) item 1's checkpoints (adds actor
coverage). Protocol: match-data norm rule throughout (`eval_rgfb_zeroshot.py` conventions);
GPI path forked from `t6_gpi_eval.py` (spec 06 item 1, already coded). 25 eps/cell, H=200,
batched lockstep envs.

Task families and pre-registered reads (all [0,1] reward convention, matching the FP cell —
NOT gc_negative):
1. **Dirac** (the band gate for item 1; for inference-only runs this must reproduce ~0.967).
2. **Two-goal max-mixture**, ± GPI (anchors above). Success = either goal reached.
3. **Region reward** `r = 1[cube ∈ region]` (T6 region protocol; ≥0.672).
4. **Dense negative-distance** `r(g) = exp(−‖cube(g) − c*‖/σ)` for a target c*, σ ≈ 2ε.
   Success metric = the OGBench criterion at c*. Read: within 5 points of the same-c* Dirac
   cell ⇒ dense readout works; the renewal-vs-plain weighting delta is the headline
   ablation number for R.
5. **Random-linear-ψ reward** `r(s') = sg[ψ(s')]·z̃/√d` (train/test-matched for the v1
   comparison arm; for FP cells this is the hardest case — no goal-set structure). Report
   the achieved reward under rollout vs the BC-flow floor; no success-rate read exists.
6. **Reach-avoid stretch** `r = 1[cube ∈ A] − 1[cube ∈ B]`: report success-at-A and
   fraction-of-time-in-B. Exploratory — no binding read, first negative-reward evidence.

Every cell reports z-inference diagnostics alongside: cos(z_r, ψ(g*)) where applicable, and
‖z_r‖ vs match-data norm.

## 4. Optional diagnostic — the linearity-conflict probe (1 seed, cheap)

`task_prior_critic_mix=true` (separate knob, off by default, guarded): ADD a scalar FP TD
grounding at z_mix (r = max_k hit, absorbing mask, bootstrap through target critic at
sg[z_mix]) on top of the untouched matrix loss. **Prediction (pre-registered, from item 0
fact 1): this HURTS the Dirac anchor** — the true mixture target `E[γ^{T_{A∪B}}]` (≥ max_k of the
per-goal values, and not any fixed linear combination of them) cannot be represented by the
z-linear critic without moving the Dirac columns it is built from. 1 seed, 500k. If it does NOT hurt (Dirac ≥0.93 and
mixtures improve over item 1), the actor-only design argument is wrong — promote to 3 seeds
and re-open the critic-grounding design. Either outcome is a paper sentence.

## 5. B-generalization gates (shared with the derivation doc's diagnostics)

The renewal-weighted readout assumes ψ interpolates over goals never hindsight-paired.
Two cheap probes from `first-passage-derivation.tex` §7 double as gates for items 2c/3:
position-only-B retrain (B on cube(g) only — prediction: ortho-vs-no-ortho gap shrinks) and
the d-sweep d ∈ {3, 5, 10, 20, 50}. If these are already queued in the other session, do not
duplicate — check `git log` + tmux registry first (shared-box rule).

---

## Ordering & stopping rule

Item 2 FIRST (frozen ckpts, no code risk, feeds 2c into everything) in parallel with item 1
implementation+tests; item 3 inference-only pass as soon as 2c exists (capability numbers
with ZERO new training — the cheapest headline); item 1 launch after tests green; item 3
full pass when item 1 lands; item 4 only after item 1's read; item 5 gated on session dedup.
Stop condition: inference-only item 3 already clears mixtures(+GPI) ≥0.75, dense within 5
pts of Dirac, and regions ≥0.672 ⇒ the capability claim needs NO new training — freeze,
write, and item 1 becomes an ablation, not a dependency.

## Success bars (pre-registered, capability claim)

- **Headline:** Dirac ≥0.90 AND mixture+GPI ≥0.75 AND regions ≥0.672 AND dense within 5 pts
  of Dirac, all zero-shot from reward samples ⇒ "an FB-style agent with first-passage
  grounding delivers zero-shot reward semantics on manipulation."
- Measure claim (item 2): FP×R ≥ occupancy-refit fidelity ⇒ "the successor measure
  factorizes into a control part and a renewal part; learn them separately."
- Any bar missed: report per-family; the negative space is publishable against the v1
  comparison arm either way.

## Do-not-build (inherited + new)

- Everything on spec 06's list (shared-parameter measure aux, conservatism on measure cells,
  warm-start curricula, whitened inference — NOTE: renewal weighting is a scalar reweight
  per goal, not a covariance inverse; it is NOT whitening and does not touch κ).
- **Critic grounding at off-Dirac z in the FP corner** — except item 4, which exists to
  falsify this rule once, at 1 seed.
- Ridge/regression task inference (`(E[ψψᵀ]+λI)⁻¹E[ψr]`) — it is whitening by another name;
  the do-not-build stands until someone shows the moment readout failing for a reason ridge
  would fix.
- Retraining R jointly with the towers, at any λ (the entire point is the quarantine).
