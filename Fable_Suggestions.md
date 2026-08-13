# Fable_Suggestions — stabilizing and extending the FB-rescue program

*2026-07-20. A brainstorm of mathematically principled modifications to the two directions in
`PAPER/rescuing-fb/fb-variants-and-grounding.tex` (§1 Bellman-supervised FB interface = RG-FB;
§2 centered successor measure = refuted), given everything the bridge experiments have shown.
Motivation: scalar TD works very well on cube (bilinear GC-IQL 0.904; RG-FB zero-shot 0.825),
measure TD fails in every tested form (0.06–0.10 ceiling). The question is: how do we either
**combine** the two so FB-the-interface performs, or repair FB-the-objective in a way that is
principled rather than another regularizer.*

---

## 0. The constraint set — what any proposal must respect

These are the findings a new idea is not allowed to contradict (each has a source in
`results/*.csv`, `analysis/rescuing-fb/*`, or DECISIONS.md):

| # | Fact | Number | Source |
|---|---|---|---|
| C1 | Scalar hindsight expectile TD on the bilinear head works | 0.904 (bilinear-d50) | bridge anchors |
| C2 | The same interface trained ONLY by scalar TD is already a working zero-shot agent | RG-FB v0 **0.825 ± 0.014**, inference gap 2.7 pts (clean) | DECISIONS 07-16 |
| C3 | The measure objective's own fixed point is bad on cube — even from the 0.93 basin | iql-fade 0.913 → 0.000 in lockstep with iql_w | anneal campaign |
| C4 | Centering + scale anchors + split bootstrap fix the *path*, not the fixed point | T10-b2: 0.18–0.26 @50k → 0.068/0.060/0.104 @500k (= uncentered ceiling) | T10 close-out |
| C5 | A measure term as auxiliary never helps and can cost | T10-a 0.823 vs 0.904 anchor (−8 pts); E2 λ≤1 band 0.86–0.90 "free" at best | T10, E2 |
| C6 | FB's failure is in-distribution, post-grasp steering — not generalization, not grasping | C1-cell FB 0.43 vs GCIQL 0.86–0.99; grasp = 1.000; P(success\|secure) = 0.437 vs 0.982 | oodgrid + phase diagnostic |
| C7 | Fitting the measure well is insufficient | fbz cut measure TD-residual 24× → success only 0.255 | fbz probe |
| C8 | The learned B is a bottleneck under measure training; ψ trained through scalar TD is not | z=g identity-B ≈0.44–0.51 vs learned-B ≈0.20 | z=g ladder |
| C9 | Every *explicit* vector/feature-TD estimator of the SF structure collapses; scalar TD on the same structure works | sf-joint 0.12, feature-TD ≤0.16, rewreg ≈0, SimpleSF-grounded ≈0 vs scalar-TD 0.38–0.90 | SF ladder |
| C10 | Zero-shot inference is NOT the weak link; z-space **coverage of the actor** is | E1a 0.955; T6 novel region-goals 0.672 but two-goal max-mixtures **0.260** with cos(z,ψ_g) still 0.77–0.82 | E1a, T6 |
| C11 | z-zeroed one-step FB is dead | 0.058 (10 seeds) | bridge_grafts.csv:107 |
| C12 | The rank-d ψ(g) bottleneck is load-bearing twice: it pools advantage across goals at training time AND keeps the backup stable | ψ(s,g): 2/3 seeds diverge; MLP dGOAL −16…−24 vs bilinear ≈0; eval-time smoothing does NOT substitute (T1) | bilinear_sg, T1, T2 |

**In flight (2026-07-20, tmux `t78`, ~5.5 h):** T7 conservatism dose-response
(pessimism 0.5 ≡ exact min, 1.0 = beyond-min; anchor plain FB 0.42) and T8 faithful one-step
FB (`onestep_z_cond=true`; matched control = C11's 0.058). Contingencies in §5.

---

## 1. A unifying frame: the FB → GC-IQL gap factorizes into four axes

This is the proposed mathematical spine for both the tex and the next experiments. Write the
measure TD loss as what it actually is: **scalar TD performed simultaneously on the family of
Dirac rewards** `r_g(s') = δ(s'≈g)/ρ(g)`, with specific choices along four independent axes:

| Axis | FB choice | GC-IQL choice | Evidence on this axis |
|---|---|---|---|
| **A. Task weighting** | test points/tasks ~ ρ⊗ρ (occupancy) | hindsight mix (0.2 current / 0.5 geometric-future / 0.3 random) | misalignment doc: 82% of γ-discounted mass beyond the control horizon; §2.1 of the synthesis |
| **B. Bootstrap policy** | a′ ~ π_z (learned, non-contractive; Zheng et al. 2602.11399) | in-sample expectile over dataset actions (contractive) | T8 tests the SARSA half; the expectile half is untested on the measure |
| **C. Value semantics** | discounted **occupancy** (counts re-visits, no absorption at the task) | **first-passage** (mask 0 at goal ⇒ V ≈ affine of E[γ^T_hit]) | the 0/−1+mask convention IS first-passage; never transplanted to the measure |
| **D. Task conditioning of F** | φ(s,a,z) (z-conditioned forward map) | φ(s,a) shared, z only at the readout | fbz costs ~6 pts (0.844 vs 0.904) — GC-IQL's choice wins |

Every bridge cell run so far sits at a corner or edge of this cube, and every failed measure
cell shares **all three** of A=occupancy, B=learned-π, C=occupancy-semantics. T10 varied none
of these axes — centering and scale anchoring are conditioning fixes *within* the FB corner,
which is exactly why they changed the path and not the fixed point (C4). The principled
program is therefore: **move one axis at a time from the FB corner toward the GC-IQL corner
and find the minimal toxic subset.** §3 turns each single-axis move into a concrete loss.

A corollary worth printing: GC-IQL *is* FB after moving all four axes — so "FB fails, GC-IQL
works" is not a comparison of two unrelated methods but a statement that some subset of
{A,B,C} is fatal (D is a ~6-pt detail). The strong negative claim ("no known form of
measure-TD grounding supports cube manipulation") survives §3's variants semantically, because
a first-passage or hindsight-weighted object is no longer *the* successor measure — but running
them is what turns the claim from "everything we tried" into "and here is which axis kills it."

---

## 2. Direction I — the combine: RG-FB v1, "task-prior scalar grounding"

RG-FB v0 proved the interface: FB's API (z-space, moment inference, z-conditioned flow
policy) trained purely by scalar TD gives a clean zero-shot agent (C2). Its two open problems
are (i) an 8-pt training-side gap to the supervised anchor and (ii) T6's finding that the
actor only covers the Dirac slice of z-space (C10). Both have principled fixes that *borrow
FB's machinery without borrowing its estimator*.

### 2.1 The headline idea: sample tasks from a prior, ground every one with scalar TD

FB's real asset was never the measure loss — it was that the **actor and critic are trained
across a distribution over z**, so the agent is a function of the task. RG-FB v0 trains both
only at z = ψ(g_hindsight). T6 shows the consequence precisely: inference produces a good
mixture z (cos 0.77–0.82) and the actor fails on it (0.260), because moment inference is
linear while the optimal policy of a mixture is not — the actor has simply never seen an
off-Dirac z.

**v1 objective.** Per batch, sample a task from a prior `T` over *computable* rewards:

- with prob p₁ — **Dirac goal** (current v0): `r(s') = 1[‖s'−g‖ ≤ ε]`, `z = ψ(g)`;
- with prob p₂ — **K-goal mixture**: goals g₁..g_K from the hindsight/random mix, weights w
  on the simplex, `r(s') = max_k 1[‖s'−g_k‖ ≤ ε]` (or Σ_k w_k·1[·]),
  `z = Σ_k w_k ψ(g_k)` renormalized by the match-data rule;
- with prob p₃ — **random linear reward**: draw z̃ (sphere × typical-ψ-norm),
  `r(s') = sg[ψ(s')]·z̃ / √d` — FB's own z-sampling, but now the reward is *evaluated*, not
  implicit.

Then apply the **same GC-IQL expectile machinery at that z**: `V(s,z) = φ_V(s)·z/√d`,
`Q(s,a,z) = min_k φ_k(s,a)·z/√d`, expectile regression of sg[Q] onto V, scalar Bellman
regression of Q onto `r + γ·m·V(s',z)` (mask m per §3.3's semantics for ball rewards), and
the flow actor + distillation conditioned on sg[z]. Nothing else changes.

**Why this is the principled combine.** The measure TD is scalar TD marginalized over Dirac
tasks at ρ-weighting with a learned-π bootstrap. v1 is scalar TD marginalized over a *chosen
task prior* with the in-sample expectile bootstrap: it keeps FB's "train on a distribution of
tasks" (the actual source of zero-shot-ness), and replaces every diagnosed pathology — the
weighting (axis A) is now task-shaped, the bootstrap (axis B) is contractive, the scale is
identified by construction because every sampled reward has units (this kills the entire
T10-b runaway class), and the linear-in-z critic structure is now *supervised* at off-Dirac
z instead of assumed by extrapolation.

**Cost & risks.** One new agent knob-set on `gciql_flow.py` (task sampler + reward evaluator);
same compute as v0. Risk 1: mixture rewards are sparse in-batch (few states inside any ball) —
mitigate by hindsight-selecting g_k near batch next-states, exactly as GC-IQL already does.
Risk 2: p₃'s dense pseudo-rewards could dominate the loss scale — normalize per-task-type as
the actor loss already normalizes by sg|Q|. Risk 3: multi-task interference degrading the
Dirac anchor — pre-register a non-regression band.

**Pre-registered reads (suggested).** Dirac anchor ≥0.80 (non-regression vs v0 0.825);
two-goal max-mixture (T6 protocol) ≥0.60 = success, 0.30–0.60 partial, ≤0.30 dead;
region-goals should not fall below 0.672.

### 2.2 GPI at inference — the zero-training fix for mixtures (run this first)

Before any retraining: T6's mixture failure has a classical remedy that needs *no new
gradients*. For `r = max_k r_k` with per-goal embeddings z_k and mixture embedding z_r, act by
**generalized policy improvement** over the candidate set Z = {z_r, ψ(g₁), …, ψ(g_K)}:

    a_j = π(s, z_j)  for each j;   execute  a_{j*},   j* = argmax_j Q(s, a_j, z_r).

Q(·,·,z_r) is exactly the linear combination the bilinear head is good at, and each candidate
policy is a Dirac policy the actor *was* trained on — GPI's guarantee (max over evaluated
policies ≥ each) applies verbatim because Q is linear in z. Cost: K+1 critic passes per step
in the existing batched-rollout harness (hours, CPU).

**Prediction:** two-goal max-mixtures 0.260 → near single-goal level (~0.8+), since reaching
the nearer goal satisfies a max-mixture and per-goal success is ~0.9. If GPI fixes it, v1's
p₂ component becomes optional for max-compositions and the paper gains an "SF/GPI machinery
transfers to RG-FB for free" section. If GPI does *not* fix it, that is strong evidence the
mixture failure is not actor coverage but something in inference — re-open E1a's ladder.

### 2.3 Closing the 8-pt training gap (mechanical ladder, in order)

1. **d512 + z-actor rerun** — the G4 anchor at d512 was 0.921; v0 is d50. Known next rung
   (DECISIONS 07-16). Cheapest points on the board.
2. **Selection-based extraction at deployment** (IDQL-style): N≈32 BC-flow proposals, argmax
   frozen Q, optionally distill. The rerank instrument says the bilinear critic's ordering is
   worth 0.947–0.96 vs the one-shot actor's 0.904 — and RG-FB inherits that critic. Also
   immune to both known extraction failure modes (off-support peak; anchor fights).
3. **Conservatism on the scalar side**: v0 already uses min over φ_k towers; if T7 shows
   pessimism moves plain FB, try the same dose on RG-FB's target (expectile already provides
   some — expect a null, cheap to close).
4. **vf conditioning**: the vf-none control (~0.84) beat vf-z (0.829) on cube — keep `none`
   on cube, but keep the z-vf knob for antmaze where the goal-agnostic anchor is the prime
   collapse suspect. Portability, not cube points.

### 2.4 Keeping an honest successor-measure reading (optional, low priority)

If the paper wants to claim the trained φ·ψᵀ still *means* something measure-like: don't add
the measure loss back (C5). Instead, **probe** it post-hoc — fit the trained representation's
implied measure against empirical discounted occupancy on held-out trajectories, report the
fidelity, and let it be bad. The scientific point ("scalar grounding produces control-useful
but occupancy-unfaithful F·B") is stronger than a cosmetic auxiliary that costs 8 points.

---

## 3. Direction II — principled measure-native repairs (one axis at a time)

These are the three untested single-axis moves from §1. Each is a small, well-defined delta
to the existing loss in `agents/fb/agent.py` / the jax measure cells — not a new regularizer.
Given C3/C4/C7, the prior on each individually rescuing FB is low; their value is that they
convert "measure TD fails" into "axis X is the toxic one," which is the difference between an
empirical observation and a claim with a mechanism. Run order = cheapest-diagnosis first.

### 3.1 Axis C — first-passage FB (mask the bootstrap at the test point)

The successor measure counts *re-visits forever*; the task pays only for *first arrival* (the
0/−1+mask convention). On cube these diverge enormously — post-drop re-visitation mass and
the 82%-beyond-horizon finding both live in the gap. The first-occupancy representation
(Moskovitz et al., ICLR'22) is the measure-side object with GC-IQL's semantics:

    F(s,a,s⁺) = E[ γ^{T_hit(s⁺)} ]  instead of  Σ_t γ^t P(s_t ≈ s⁺).

**Implementation is one masking line.** In the batch measure loss, the test points are the
batch's own next states, so "the test point is hit" is (a tolerance ball around) the
diagonal: multiply the bootstrap term by `(1 − hit_ij)`, `hit_ij = 1[‖s'_i − s⁺_j‖ ≤ ε]`
with ε = the 4 cm success radius on the cube-xyz coordinates:

    δ_ij = M_ij − hit_ij − γ·(1 − hit_ij)·sg[M̄_ij].

This makes each column of M exactly a masked sparse-reward value function — the *same
fixed-point semantics* GC-IQL trains — while keeping the bilinear factorization, the z-space,
and FB inference intact. It also removes the non-contraction's worst amplifier: unbounded
re-visitation mass is what lets columns of M inflate together.

**Read:** ≫0.10 ⇒ semantics was a/the toxic axis; ≈0.10 ⇒ occupancy-vs-first-passage is not
what kills the measure on cube (and the misalignment doc's 82% number needs reframing as
correlate, not cause).

### 3.2 Axis B — expectile (in-sample) measure backup, "IQL-ified FB"

T8 tests the SARSA half of axis B. The full GC-IQL move is stronger: SARSA evaluates μ, the
expectile approximates an in-sample *max*. Since each test-point column of M is a scalar
value function, the IQL construction applies pointwise: introduce `V_M(s, s⁺)` (bilinear:
φ_V(s)·ψ̃(s⁺)/√d), train it by expectile regression onto sg[M(s, a_data, s⁺)], and back up

    M(s,a,s⁺) ← r_δ + γ · sg[V_M(s', s⁺)],

with no a′ of any kind — no learned-π bootstrap (contraction restored *and* policy
improvement retained, unlike SARSA which caps you at μ's measure). This is the exact
transformation that took GC-Bellman from unstable to 0.904 on the scalar side, applied to the
measure. It composes with 3.1's masking (and probably should — run C+B jointly as the
"GC-IQL-semantics measure" cell if singles are both partial).

**Read:** if T8-zcond lands ≈0.058 but this cell lifts, the story is "one-step evaluation of
μ is not enough — in-sample improvement is the missing bootstrap"; if this also floors at
~0.1, axis B is exonerated and A/C carry the blame.

### 3.3 Axis A — hindsight-weighted test points

Replace the off-diagonal test-point distribution (batch next states ~ ρ) with the goal mix
that scalar grounding uses: 50% geometric-future states *from the same trajectory as (s,a)*,
30% random, 20% current. The diagonal (positive) term likewise moves from "this transition's
next state" to "a future state of this trajectory" — which turns the measure loss into a
temporal-contrastive objective pointed at *controllable* futures rather than occupancy mass.
(Honest caveat: at this end of the axis the density interpretation needs an importance
correction ρ_future/ρ, or you accept that the object is no longer the ρ-referenced measure —
that's fine; say so.)

**Read:** this is the direct test of the synthesis §2.1 claim ("measure TD is
occupancy-weighted; manipulation needs task-weighted supervision"). If A alone moves the
needle, the paper's central mechanism sentence is validated at the loss level.

### 3.4 What NOT to build in Direction II (already refuted — do not resurrect)

- Centering / dueling-style baseline splits of M (T10-b, T10-b2: path not fixed point).
- Scale normalization of both measure legs (T10-b v1 runaway, q→10¹³).
- Measure as auxiliary to scalar TD at any λ, centered or not (T10-a −8 pts; E2 at best free).
- Curricula/annealing into a measure-only endpoint (iql-fade: collapse tracks iql_w→0).
- z-zeroed one-step (0.058, 10 seeds — C11). Only the z-conditioned reading (T8) is open.
- Eval-time goal smoothing as a substitute for the ψ(g) bottleneck (T1: monotone decline).
- Full-rank ψ(s,g) heads without new stabilization (2/3 seeds diverge; if ever revisited, it
  needs an output bound / V-target / norm control per the T2 diagnosis, not a rerun).

---

## 4. Cross-cutting ideas worth one paragraph each

- **"SFs without feature TD" as the theory frame.** Bilinear GC-IQL implicitly learns a
  d-dim reward basis ψ and value features φ by scalar TD (C9). v1 (§2.1) makes the implicit
  explicit: it supervises the linear-in-z structure on a task prior. This is the cleanest
  positioning sentence for the tex §1 and connects to Ollivier's "which features for SFs."
- **Trajectory-level z sampling (FB-CPR / behavioral task sampling).** If p₃'s sphere-z tasks
  are too alien, replace with z inferred from random dataset *trajectory windows* (moment
  inference over a window's next-states). Keeps the prior on achievable tasks; literature
  (Bendib et al.) reports gains on OGBench cube specifically.
- **Occupancy-corrected inference is refuted on cube-single (E1a whitening) but** keep the
  least-squares form `z = (E[ψψᵀ]+εI)⁻¹E[rψ]` in the toolbox for *mixture* rewards if GPI
  (§2.2) fails — under scalar-TD-trained ψ (no orthonormalization), moment ≠ least-squares,
  and mixtures are where the difference would bite.
- **Phase-targeted evaluation, not phase-targeted architecture.** C6 says all FB failure is
  post-secure steering. Whatever variant runs next, report P(success|secure) alongside
  success — it is the sensitive instrument (0.437 vs 0.982 dwarfs the 0.42-vs-0.90 headline)
  and cheap to compute with the existing phase probe.
- **Scale identifiability as a design principle.** Every objective in this doc either has
  computable rewards (units pin the scale) or a masked/bounded fixed point. Any future
  measure variant should pass the question "what pins the absolute scale?" *before* launch —
  T10-b's runaway was predictable on this criterion alone.

---

## 5. Contingencies on the in-flight arms (T7/T8) and a priority queue

**T7 (pessimism 0.5/1.0 vs plain-FB 0.42):** if min-aggregation ≫0.42, conservatism becomes
a *mandatory axis* added to every §3 cell before any is declared dead (and the FB-failure
claim is restated against conservative FB). If ≈0.42, the VC-FB confound closes and §3 cells
run without it.

**T8 (z-conditioned one-step vs 0.058 control):** if ≫0.058, one-step FB is a live baseline
and §3.2 (expectile backup) is the natural escalation of the same axis — promote it. If
≈0.058, spec 04 §3 licenses the strong negative claim; §3's cells then become *optional
mechanism work* — run them only if the paper wants "which axis is toxic" rather than "all
tested forms fail." (They do not threaten the claim's wording: first-passage/hindsight
objects are not the successor measure.)

**Suggested queue (cost-ordered):**

| Priority | Item | Cost | What it buys |
|---|---|---|---|
| 1 | GPI over goal candidates at inference (§2.2) | hours, CPU, no training | likely fixes T6 mixtures; free paper section |
| 2 | RG-FB d512 + z-actor (§2.3.1) | ~1 day GPU | the known next rung toward 0.90+ zero-shot |
| 3 | Selection extraction on RG-FB (§2.3.2) | hours + optional distill | +3–5 pts per the rerank instrument |
| 4 | RG-FB v1 task-prior grounding (§2.1) | ~2 days impl + 3 seeds | the headline combine; makes "zero-shot RL" (not just goal-reaching) claimable |
| 5 | First-passage FB (§3.1) | 1 masking line + 3 seeds | cheapest single-axis diagnosis |
| 6 | Expectile-measure FB (§3.2) | ~1 day impl + 3 seeds | completes axis B beyond T8 |
| 7 | Hindsight-weighted measure (§3.3) | ~1 day impl + 3 seeds | loss-level test of the weighting thesis |

Items 1–4 are Direction I (the paper's method); 5–7 are Direction II (the paper's mechanism
section) and should be triaged after T7/T8 land. Everything in §3.4 stays dead.
