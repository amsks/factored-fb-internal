# Spec 10 — Brainstorm: promoting the phase-2 object to a full successor measure

Companion to `two-phase-successor-measure-note.tex` (§7 object, §8 dense) and spec 08.
Input: the 7-gap capability list (z/x independence; compromise; signed rewards; linear
rewards; calibrated integration; arbitrary sets; continuous GPI), which reduces to two
structural deficits — (a) `z` and `x` do not vary independently (being closed by the ZX
form), and (b) `B(x)` is a hand-placed **ball mass**, not a **density w.r.t. a reference
measure**. This spec is the plan for (b), plus the consequences that fall out for
(1)–(7). Nothing here abandons the training discipline that made the method work:
**every training target stays a bounded, representable residual; no linear terms; no
orthonormality; one axis per cell.**

---

## 0. What we actually have, stated exactly

The phase-2 (ZX) object is the **ε-mollified successor measure evaluated at dataset
centers**:

    M̃(s,a,z,g) = ∫ 1[x ∈ B_ε(g)] dM^{π_z}(s,a,·)(x) = (K_ε * M^{π_z})(g)

i.e. the convolution of the true measure with the 4 cm indicator kernel, read at
cube-position centers. Three consequences that the note under-uses:

1. **The TD fixed point does not depend on the dataset density over g.** Regression is
   pointwise: ρ(g) determines *where M̃ is accurate*, not *what it converges to*. The
   "inherits the dataset density" defect lives entirely in the **inference-time
   integral** (sampling goal centers from ρ), not in the learned object.
2. **"Double counting of overlapping balls" is exactly kernel smoothing, not an error.**
   For any weight w over centers, ∫ M̃(s,a,z,g) w(g) dg = ∫ (w * K_ε)(c(x)) dM(x).
   Overlap re-counts each future state once per covering ball — which is precisely what
   the same overlap does to the *data* distribution. So the correction is a **ratio**,
   and the smoothing biases cancel to first order (next section).
3. The mollification bias on a reward r is O(ε²·∇²r): for rewards smoother than 4 cm
   the ε-ball object already supports genuine integration.

So the honest gap is smaller than the note's phrasing: we are one **denominator** away
from a density, and one **readout route** away from consuming it.

---

## 1. KEYSTONE: the density as a ratio of two bounded regressions

FB's object is m = dM/dρ. Its estimator fails here because it fits a *ratio with a
singular immediate term* inside one TD loss. Instead: estimate numerator and
denominator **separately**, each with our stable machinery, and take the ratio only at
readout:

- **Numerator** (have it): M̃(s,a,z,g) = M^{π_z}(s,a,B_ε(g)) — phase-2 TD, bounded,
  γ-contraction.
- **Denominator** (trivial to get): N(g) = ρ(B_ε(g)) = P_{x~ρ}(x ∈ B_ε(g)).
  This is **not even a TD problem**: per batch, the column-mean of the existing hit
  matrix is an unbiased MC target; fit a tiny head (or even kNN/post-hoc MLP on frozen
  ψ(g) — no training-loop change at all). Bounded in [0,1], plain regression.

Then

    m̂(s,a,z,x) = M̃(s,a,z,c(x)) / N̂(c(x))  ≈  dM^{π_z}/dρ (s,a,x),   ε-smoothed,

and the **same-kernel smoothing cancels in the ratio** — including the boundary/manifold
effects that worried us (ball clipped by the table: the cube's reachable set is ~2D
on-table, 3D in transport; a raw Lebesgue volume normalization would be wrong in a
state-dependent way, but N̂ uses the identical clipped kernel, so the ratio is correct
everywhere).

What this buys, immediately:

- **Calibrated integration (gap 5).** Q̂_r(s,a,z) = (1/m) Σ_j [r_j / N̂(g_j)] ·
  M̃(s,a,z,g_j) is self-normalized importance sampling of ∫ r dM with samples from ρ.
  This is E4, but with the measure's *own kernel* as the importance weight instead of an
  alien kNN estimate — consistent by construction.
- **The corrected FB moment readout (gaps 1, 4).** The moment z_r = mean r_j ψ(g_j) is
  wrong under ρ-sampling of a mass object; the corrected version is simply
  **z_r = mean_j [r_j / N̂(g_j)] ψ(g_j)** — one reweighting. This is the principled
  version of "renewal z-weighting" and subsumes it.
- **A decisive, control-free validation harness.** Before any rollout: for random query
  sets X (boxes, unions), compare Σ_j w_j 1[g_j ∈ X] M̃(s,a,z,g_j) against direct MC
  occupancy of X under logged trajectories at matched (s,a). Pure evaluation
  calibration; cheap; kills or licenses "∫ r dM" as a claim with no actor confound.

Risk: 1/N̂ inflates variance where data is thin (classic IS). Mitigate: self-normalize,
clip weights at a quantile, and report effective sample size.

---

## 2. Route change at inference: stop inferring z open-loop; search it against the measure

Fresh evidence (E3, landed): the ridge readout does **not** fix the dense split — s1
moment 0.128 → ridge ≈ 0.46 (partial), and ridge *hurts* the good seed (s0 moment 0.896
→ 0.47–0.65 across λ). Covariance anisotropy is at best half the story; open-loop
embedding-space inference is fragile per se.

But the separated form gives us something FB never had here: a **differentiable
evaluation oracle over the continuous policy family**,

    z  ↦  Q̂_r(s,a,z) = Σ_j w_j r_j F(s,a,z)ᵀB(g_j)/√d,     w_j = r-independent 1/N̂ weights.

So select z by **maximizing the measure's own estimate**, not by trusting a moment
formula:

- Candidates: {ψ(g) for top-k reward states} ∪ {corrected moment z_r} ∪ gradient-ascent
  refinements of z from those inits (a few Adam steps on Q̂_r(s₀,·,z)).
- Per-episode (pick z* at reset) and per-state (GPI over the continuum = gap 7) variants.
- This one route serves **every** family uniformly: dirac (reduces to retrieval),
  mixture/compromise (gap 2 — at fixed z, M̃(z,g₁)+M̃(z,g₂) IS one policy's value of the
  union reward; the sum-of-slices pathology is gone by construction in ZX), region,
  linear (no linearity-in-ψ assumption used — r is consumed as a black box sampled at
  dataset points), signed (weights can be negative; linearity of the integral is free).

Guardrails: z-search can exploit critic error off-manifold (the OOD-z analogue of the
oodgrid finding). Constrain to a trust region around the candidate set / the ψ(data)
hull; log Q̂_r at z* vs at candidates; if search-found z* wins the readout but loses the
rollout, that gap itself is the measurement of off-manifold critic optimism.

---

## 3. The policy-family gap nobody has closed: π_z is untrained off the goal manifold

Code fact (`gciql_flow.py` actor_loss, ~line 959): under the ZX form the actor's greedy
term is q = F(s,a,ψ(g))ᵀB(g) — the **diagonal readout only**. Random-z rows train the
*critic family* (with z-consistent bootstrap a′~π(·|s′,z)), but the **actor is never
optimized at any z off the ψ(goal) manifold** — π(·|s,z_random) is pure generalization.
Direct-z extraction therefore leans on an actor input mode with zero training signal.
This is a live suspect for the dense split and for whatever E1's region/linear payoff
shows.

Candidate fixes (one per cell, in increasing ambition):

- **3a. FB-faithful actor extension:** for random-z rows add the greedy term
  q = F(s,a,z)ᵀz/√d (the reward direction z's own Q — FB's actor objective). Meaningful
  exactly to the extent B-space directions are reward directions; becomes principled
  once §1's density calibration holds.
- **3b. Mixture-matched greedy:** sample a K-goal mixture, set z = norm-matched
  Σ w_k ψ(g_k), greedy on Σ w_k F(s,a,z)ᵀB(g_k) — trains the actor on the *compromise*
  readout it will be asked to execute (gap 2's actor side). Note `_sample_tasks` already
  builds exactly these arms for the (dead) scalar-TD path — the arms can be recycled
  with the greedy readout swapped in for the scalar TD. Warning from the record:
  TPActor's 0.395 was attributed to the z-*prior*, and zctl showed z-conditioning itself
  is fine (0.960/0.996). So the prior must stay dominated by ψ(goal) samples; anneal the
  mixture/random fraction in (E6's knob generalizes).
- **3c. Actor-free fallback:** if actor training off-manifold stays fragile, decode z*
  to its nearest goal-manifold candidate for *execution* while using the measure for
  *selection* — keeps gains 1/2/7 at the eval level without touching the actor.

---

## 4. Making the query argument a measure: regions, additivity, multi-scale

Everything so far keeps B_ε(dataset-state) columns. Three escalating moves on the
**query side** (gap 6):

- **4a. Random-region columns [+knob].** Generalize `measure_hit_spec` columns from
  balls at dataset next-states to sampled regions: random centers (not only dataset
  states), random radii, axis-aligned boxes, unions of 2 balls. h_ij = 1[c(s′_i) ∈ X_j]
  stays a bounded 0/1 event; the backup is unchanged. B's input becomes a region
  parameterization (center ⊕ log-radius ⊕ type). Then "occupancy of an arbitrary set" is
  *trained*, not derived — and the training distribution over X is ours to design
  (task-agnosticism improves: no single privileged ε).
- **4b. Additivity by construction (the deepest single change).** Parameterize the
  region embedding as an **average of point features**:
  B(X) = λ̂(X) · mean_{x ∈ S_X} b(x), with S_X a sample of X. Then
  M(s,a,z,X) = F(s,a,z)ᵀB(X) is finitely additive over disjoint unions *by
  architecture*, and F(s,a,z)ᵀb(x) **is a successor density** — obtained without ever
  writing a Dirac into a TD target. The singular term never appears because supervision
  is always a finite region hit; the density emerges as the ε→0 limit of ball-averages
  (classic mollified regression). This is the cleanest answer to "make B a density": FB's
  API (moment z_r = mean r_j b(g_j)-style readouts, linear rewards, signed rewards) all
  reattach to `b`, but every gradient ever taken was a bounded squared residual.
  Rank caveat: densities plausibly need d > 50; start by holding d and measuring the
  residual floor vs the ball form.
- **4c. Multi-scale ε heads.** Train ε ∈ {2, 4, 8} cm sharing φ (per-ε ψ head or FiLM).
  Gives: (i) a Richardson step toward the ε→0 density; (ii) a measured smoothing-bias
  curve (how wrong is 4 cm integration for each reward family); (iii) the region-width
  ablation the note wants for the conjunction-vs-sparsity confound — **and** (iv) the
  fix for E7/E8 sparsity: choose ε per environment by a **target positive rate**
  (quantile-calibrated ε matching cube-single's ~14 off-diag positives/row) instead of
  hand-porting 4 cm. Adaptive-ε-by-quantile is probably the single most practical idea
  here for the multi-env suite.
- **4d. (Speculative) learned quotient.** Replace the privileged cube-coordinate kernel
  by a ball in a *learned* metric: h = 1[‖ψ̄(s′) − ψ̄(g)‖ ≤ ε_q] with ψ̄ an EMA copy and
  ε_q quantile-calibrated per batch (fixes the hit rate, which blocks the collapse mode
  of a shrinking metric). Removes the "not fully task-agnostic" caveat. High risk
  (self-referential kernel); run last, as a headline-hardening ablation.

---

## 5. Signed rewards and compromise: mostly an evaluation-design problem now

Once §1 holds, ∫ r dM is linear in r and negative weights cost occupancy automatically.
The current avoid-side null (time-in-B ≤ 0.003 for *every* policy) says the avoid region
never binds — no candidate trajectory passes through it. Fixes:

- Place the avoid region **inside the play corridor** (high-ρ region between start and
  goal, e.g. the table center the cube gets dragged across), so the unpenalized optimal
  path crosses it; metric = deflection (path detour) + success retention, not raw
  time-in-B.
- Compromise test with teeth: reward = union of two balls at intermediate separation;
  compare max_z [M̃(z,g₁)+M̃(z,g₂)] rollouts vs best single-goal policy vs GPI-over-two.
  A compromise z that beats both singletons is the first behavior *only* a (z,x)-measure
  can produce — a headline figure if it lands.
- Behavior-anchoring caveat: the flow-BC actor cannot express avoidance the data never
  shows; report the BC-floor control alongside.

---

## 6. The R-structure, upgraded from nuisance to object (post-E5)

E5 (landed) **refutes strict action-independence of R**: argmax_a Q vs argmax_a M
agreement 0.295/0.299, ICC(R|g) 0.242/0.292, within-goal CV 1.3–1.8. Phase 2 works
anyway (it never assumed the factorization — TD learns full M; the split only motivated
the curriculum). But the refutation says R carries real control information
(secure-placement vs pass-through — the interaction variable of the CLAUDE.md thesis).
Two upgrades:

- **6a. Interaction-conditioned renewal head (phase 3).** Renewal identity: splitting at
  first arrival, M(s,a,g) ≈ Q_fp(s,a,g) · (1 + γ·M_ret(s_T, g)), where M_ret is
  occupancy started *from the arrival state*. Train R(s_T, g) = 1 + γM_ret on arrival
  states only (a small, dense subproblem; MC or TD on frozen ψ, arrival states are
  h=1 rows we already index). Readout M̂ = Q_fp · R(s,·) restores the arrival-mode
  information the diagonal split discards, and gives calibrated occupancy on envs where
  full phase-2 TD is unstable (cube-double/scene, if E8's phase 2 wobbles).
- **6b. Renewal-form density.** In renewal-theory terms the occupancy density is the
  first-passage kernel convolved with the renewal measure. Architecturally: keep the
  phase-1 F,B as the first-passage kernel and multiply a per-(region, arrival-mode)
  renewal factor — a low-rank correction, trained by bounded regression. This is the
  factorization-as-inductive-bias made honest after E5.

---

## 7. Optional endgame: distill to the vanilla FB API

Once the ZX visitation object + N̂ ratio is validated: **distill** (F_dens, b) by plain
supervised regression to match m̂ = M̃/N̂ on sampled (s,a,z,x). No TD, no stability
risk. Output: a bona fide FB pair with the standard moment-inference API, trained
end-to-end by nothing but bounded regressions. Paper-level story: *"FB's object is
recoverable on manipulation — but only by estimating the numerator and denominator of
its density separately, each as a bounded first-passage/visitation regression, and never
by the density TD."*

---

## 8. Experiment ladder (leverage ÷ cost, one axis per cell)

| # | Cell | Type | Gaps | Pre-registered read |
|---|------|------|------|---------------------|
| L1 | N̂ head (post-hoc on frozen ψ) + calibration harness: Σw_j M̃ vs MC occupancy on random boxes | eval-only | 5 | rel. error ≤ ~15% across box sizes ⇒ "∫r dM" licensed; systematic bias ⇒ measure mis-calibrated, fix before anything else |
| L2 | 1/N̂-corrected readouts on E1's ZX-visitation ckpts: region, linear, mixture — moment-corrected AND z-search/GPI-continuum | eval-only | 1,2,4,7 | region ≥ 0.80 (clears retrieval ceiling) or linear beats control with an edge ⇒ first measure-does-work evidence; z-search ≫ corrected-moment ⇒ inference was the boundary, not the object |
| L3 | Compromise + corridor reach-avoid eval redesign | eval-only | 2,3 | compromise-z beats best singleton ⇒ headline figure |
| L4 | Actor off-manifold probe: rollout π(·|s,z) for random/mixture z, score coherence | eval-only | — | incoherent ⇒ 3a/3b required before direct-z claims |
| L5 | Multi-ε heads (2/4/8 cm) + quantile-ε | +knob | 5,6 + E7 sparsity | smoothing-bias curve; quantile-ε un-floors cube-double conjunction |
| L6 | Actor extension 3a or 3b (one, not both) | +knob | 1,2 | dense split closes / direct-z region improves; watch for TPActor-prior regression |
| L7 | Random-region columns | +knob | 6 | arbitrary-box eval matches ball eval ⇒ query generalization real |
| L8 | Additive B(X) = λ·mean b(x) | new arch | 5,6,(4) | residual floor vs ball form at d=50; then density readouts |
| L9 | Phase-3 renewal head R(s_T,g) | +head | 6 (§6) | recovers E5's lost control info; portable occupancy for E8 envs |
| L10 | Distill to (F_dens, b) | offline | endgame | vanilla-FB API numbers on the zero-shot grid |

L1–L4 are **eval-only and immediately runnable** on existing checkpoints (ZX phase-1
zx_s0/s1, ZX visitation visz_s0/s1, FP-NoOrtho s0–s2). L5–L9 are training cells for
after E8 settles GPU pressure.

## 9. Anti-patterns (from the negative record — do not reintroduce)

- No linear/unregressed loss terms anywhere (density-TD divergence, 154→5e4).
- No orthonormality (0.967→0.225).
- No scalar task-TD grounding into the critic (criticmix late-collapse ×7) — grounding
  goes through the matrix residual only.
- No stacking of new axes onto a working cell (full-corner 0.00); one axis per cell.
- Sparse-conjunction cells need a sparsity control or quantile-ε (E7: 27× sparsity).
- Off-manifold z consumption without either actor training there (§3) or a trust region
  (§2) is the current top suspect for every direct-z anomaly.
