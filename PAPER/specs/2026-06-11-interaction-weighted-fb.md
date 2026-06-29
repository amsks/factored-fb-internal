# Interaction-weighted FB (IWR-FB) — derivation + experiment spec

**Date:** 2026-06-11 · **Status:** derivation + implementation; first runs 3 seeds @500k.
**Context:** the FB-side port of IWR (Factored_URL_in_Manipulation), motivated by the
gap-analysis program (B1: the FB measure loss allocates basis capacity by ρ-persistence
and deletes the interaction variable; B3: the actor's z-distribution defines the policy
index — eval-time fixes fail). This is a TRAINING-level intervention.

## 1. Derivation

### 1.1 What the goal measure does in FB
FB minimizes, over rank-d factorizations m(s,a,g) = F(s,a,z)ᵀB(g),

  L_ρ(F,B) = E_{(s,a,s⁺)∼ρ} E_{g∼ρ_G} [ (F(s,a,z)ᵀB(g) − γ F̄ᵀB̄(g))² ] − 2 E[F(s,a,z)ᵀB(s⁺)]

with E_{ρ_G}[B Bᵀ] = I. The goal marginal ρ_G enters in three places:
(i) the **off-diagonal integral** — where in goal space approximation error is charged;
(ii) the **orthonormalization** — the inner product under which B is whitened;
(iii) via z = B(g), g∼ρ_G — the **task distribution the actor is trained on**.

Changing ρ_G → ρ' with density w(g) = dρ'/dρ_G is *exactly* FB in L2(ρ'):
the optimal rank-d (F,B) becomes the top-d singular space of M under the ρ'-weighted
inner product. The diagonal (Dirac) term is measure-invariant (density ratio cancels),
so a pure-ρ' FB leaves it unweighted (already implemented: `fb_successor_terms`,
`ortho_cov`, the coverage-balanced machinery).

### 1.2 Why persistence-ordering buries interaction, and what w must do
The rank-d L2(ρ') optimum orders singular functions of M by σ_k² ∝ (temporal
persistence under the behavior process) × (ρ'-mass of the mode's support).
In play data the interaction variable ω (free/held/slipping) toggles fast and its
boundary (the contact set I) has small ρ-mass ⇒ under ρ'=ρ it is truncated first
(measured: lift R²(B→d)=0.02). A weight w concentrated near I raises the ρ'-mass
factor of exactly the modes supported on the interaction boundary — re-ordering the
spectrum so interaction modes survive truncation. **This is the formal content of
IWR translated to successor measures:** IWR tilts the contrastive positive
distribution toward I; here we tilt the L2 base measure (and the task distribution).

### 1.3 The weight (IWR's Eq., ported)
With effector position u_p(g) and cube position y_p(g) read from the goal state
(obs frame, slices 12:15 / 19:22 — same frame, so ‖u_p−y_p‖ is hand–cube distance):

  d(g) = ‖u_p(g) − y_p(g)‖₂,
  w(g) ∝ ε + exp( −(d(g) − c)² / (2σ²) ),   then mean-normalized (E[w]=1).

c = contact distance (set from data: mode of d over `grasped`-labeled states),
σ = interaction window, ε > 0 keeps global coverage (ρ' ≫ 0 everywhere, so the
FB factorization remains valid on all of ρ's support).

**Difference from the killed coverage-balanced reweight:** that w was inverse cube-
position density (up-weights *rare object placements* — a position-space tilt, the
same subspace FB already prefers). IWR-w is a tilt toward the *interaction boundary*
— the subspace FB deletes. The kill showed ρ'-position doesn't fix the readout; the
B1 mechanism predicts ρ'-interaction reshapes which modes B keeps (testable via
z-decode lift R² and separability — *representation-level* endpoints, not only success).

### 1.4 The z-distribution leg (required by B3)
B3 showed z is a policy-family index: the actor only understands z's from its
training distribution. So reweighting only the measure (i, ii) without the task leg
(iii) trains a better basis the actor never exercises on interaction tasks. Hence the
experiment crosses the w with `weight_z` (resample goal-derived z = B(g) by w), which
makes the actor *practice interaction-conditioned tasks*. Predictions:
- w only (pure-ρ'): lift R²(B→d) ↑, separability ↑; success gain modest.
- w + weight_z: same basis gains + actor coupling ⇒ the success-relevant variant.
- (variant C, later) zHER: z = B(s_{t+k}) same-trajectory future near contact — the
  exact analog of IWR's trajectory-future positives, touches the diag/task semantics.

## 1.5 Derivation: trajectory-future goals (zHER) — what the coupling does

### (a) The induced distribution
Sampling a transition (s,a,s′) ∼ ρ and a goal g = s_{t+k}, k ∼ Geom(1−γ) truncated
at the episode end, induces the joint

  Γ(s,a,g) = ρ(s,a) · (1−γ) m^β(s,a,g) ρ_G(g),

i.e. **g is drawn from the normalized successor measure of the behavior policy**,
(1−γ)M^β(s,a,·). This is exactly CRL's positive distribution. Standard FB instead
uses the product ρ(s,a) ⊗ ρ_G(g) (independent uniform-buffer goals). zHER therefore
replaces the *task-leg coupling* — which (s, z=B(g)) pairs train together — from
"any state × any task" to "any state × a task feasible from it within ~1/(1−γ) steps."
Nothing else changes: the measure loss, ortho, and the 50% random-sphere z mix are
untouched, so the FB identity is still enforced; only **where in z-space accuracy is
allocated** moves (from the uniform goal-embedding cloud to the feasible-task manifold).

### (b) Why the coupling carries the controllability signal (the B2 mechanism)
Consider two anchor states that alias under the basis (e.g. hold vs about-to-slip,
small ‖Δs‖) and a goal distribution q(g | s,a). The value-regression pressure to
distinguish them is governed by

  D_q(hold, slip) = E_{g∼q} [ ( m(hold, g) − m(slip, g) )² ].

Under q = ρ_G (uniform), most goals are unreachable from either state within the
horizon — m(hold,g) ≈ m(slip,g) ≈ 0 — so D ≈ 0: the regression target is *constant*
along the controllability direction and TD faithfully learns to ignore it. This is
the measured B2 verdict (MLP + uniform goals → cf-AUC 0.50: the form didn't matter,
the target had no signal). Under q = (1−γ)M^β(s,·), mass concentrates precisely on
goals reachable from hold but not from slip (transported-cube futures), so
D is large and the distinction enters the loss with the data's natural bridge mass.
**The trajectory coupling is an importance weighting of the controllability
discrepancy.** IWR's contact bump composes with it by further concentrating q near
the interaction boundary.

### (c) Renewal/bridge view
Goals beyond the anchor's current mode (a lifted-cube future sampled from a pre-grasp
anchor) force the z-conditioned value to represent the first-passage factor of the
renewal decomposition M = M_pre + E[γ^τ M(s_τ, ·)] — the cross-mode coupling enters
the loss with weight (1−γ)E[γ^τ] (the natural interaction frequency of the play data)
instead of the thin product mass ρ(s)ρ_G(g). zHER is the *data-side* approximation of
the bridge term; Bridge-FB is its architectural counterpart.

### (d) Zero-shot validity and the B3 alignment
FB's inference z_r = E_ρ[rB] is unchanged and the identity is still trained over a
full-support z mix (random-sphere half remains). Eval tasks are achievable tasks, so
their inferred z's lie in (or near) the feasible-task cloud — zHER trains the actor's
z→behavior map *exactly where eval queries it*, repairing the B3 decoupling at its
source rather than at inference time.

### (e) Reconciliation table (model form × goal coupling)
| | uniform goals | trajectory-future goals |
|---|---|---|
| linear density, bilinear (FB/PSM) | cf-AUC ≈ 0.49–0.52 (measured) | **zHER — running** |
| log density, bilinear (CRL) | — | 0.68 (paper) |
| MLP value (GCIQL-style) | 0.50 (measured, B2 control) | 0.89 (GCIQL) ; in-house cell running |
CRL sitting *between* FB and GCIQL is exactly what this framework predicts: right
coupling, restrictive form. The two running cells complete the table.

### (f) Falsifiable predictions (dissociation structure)
- zHER fixes the **value/task legs, not the basis spectrum** (the measure loss still
  orders B by persistence): predict cf-AUC ↑ and success ↑, but B's lift-R² stays low.
- IWR-ρ′ fixes the **basis spectrum, not the coupling**: predict lift-R²(B) ↑ and
  separability ↑, with smaller success gain.
- If both hold, the full fix is the **composition** (IWR-ρ′ + zHER) — next run.

## 2. Implementation (smallest diff, default-off)
- `agents/fb/agent.py`: add `interaction_weights(eff_xyz, cube_xyz, c, sigma, eps)`
  (mean-normalized, detached); `_coverage_weights` → dispatch on `reweight_mode`:
  `"coverage"` (existing inverse-density) | `"interaction"` (new). Weight flows through
  the EXISTING ρ' plumbing untouched: `fb_successor_terms(w)`, `ortho_cov(B,w)`,
  q-cov, `weight_z`.
- `configs/agent/fb.yaml`: `reweight_mode: coverage`, `iwr_c`, `iwr_sigma`,
  `iwr_eps` (alpha>0 still the master switch; byte-identical when off).
- obs-frame distances from `batch["next"]["observation"]` (no physics needed).
- tests: peak at d=c, ε floor, E[w]=1, w=None paths byte-identical.

## 3. Runs (GPUs 4–7, free since fb_z128 finished)
3 seeds @500k, ortho_coef=1000, z50 (= fb_state_plain config + reweight):
  A `fb_iwr`      : reweight_mode=interaction, alpha=1, weight_z=false
  B `fb_iwr_wz`   : same + weight_z=true
Readouts at 500k: success vs fb_state_plain@500k; **z-decode lift R² + separability
AUC on B** (the mechanism endpoints); then cf-AUC probe if promoted to 1M×10.
