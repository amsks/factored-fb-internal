# Spec 30 — Response to the method review: gauge-mean correction, TD InfoNCE positioning, and the isolation baselines

Status: WRITTEN 2026-08-16 from a three-way verification of the external
method review (code audit, literature check, and a RUN geometry
diagnostic — results/spec30_geom/, script scripts/rescuing_fb/
eval_spec30_geom.py, uncommitted). House rules as specs 27–29.

**Rev 1 (2026-08-16, post-assessment).** The reviewer assessed this spec
as the correct research program, approved for execution after four
factual corrections and five technical changes. All were re-verified
(primary-source fetches + the M=1 algebra) and all hold; they are folded
in below. The consequential ones: the Finer-BFM rebuttal is RETRACTED,
the TD InfoNCE offline positioning is corrected, and the centered
reward-sample encoder has NO one-sample (Dirac) limit — point prompting
and reward-sample prompting are now separate APIs, and P1/P2/P3 are
revised accordingly.

## 0. Verified verdict on the review

**Confirmed, decisive:**
- **TD InfoNCE (arXiv 2310.20141, ICLR 2024) relation is real.** With
  hard immediate labels and matched query supports, Density-FB's measure
  loss **reduces to the categorical Bellman update underlying TD
  InfoNCE** ((1−γ)·CE(observed successor) + γ·CE(stop-grad target
  softmax), marginal negatives, bilinear two-tower); their appendix
  proves the classifier is the normalized discounted future-state
  distribution. Do NOT say "update-for-update identical": implementation
  deltas are the positive-slot supports (their immediate term labels the
  dynamics next state, the bootstrap term labels a marginal draw, with
  ONE shared negative pool s^(2:N) — our refinement of the reviewer's
  own account), ℓ2-normalized towers, a goal-contrastive actor CE, an
  MSE goal-BC regularizer, and their relabeling distribution.
  **Corrected positioning (Rev 1): TD InfoNCE IS an offline GC-RL
  method** — offline D4RL AntMaze with a GC-BC MSE regularizer
  (their App. D.3; actor = contrastive CE over the commanded goal vs
  marginal negatives, their Eq. 11). The honest delta: it learns a
  raw-goal-conditioned policy and occupancy estimator; it has **no
  reward-prompted latent policy family, no general reward→policy-index
  inference mechanism, and no validated zero-shot control for arbitrary
  rewards** (its arbitrary-reward note is fixed-policy, appendix-level).
  ⚠ Our flagship cells are the SOFT variant (measure_density_hard:
  False in tab:main .929/.736 and the R-A screen); only the hard
  ablation rows are the direct-reduction candidate.
- **§4.4 geometry: reviewer's scenario A HOLDS (measured).** ‖μ_B‖ is
  33–103× the centered signal (z25 1087.8 vs 10.6). eff_rank: uncentered
  Σ_B 1.000–1.002 (archived) vs **centered Cov(B) 6.3–20.8 of 50**. The
  "B collapses onto one direction" claim in spec 28 / tex §8.2 /
  compendium §6.6 OVERSTATES: the dominant direction is a softmax-
  invariant additive gauge (B→B+b leaves every density unchanged), not a
  collapse of the informative geometry. **Centering alone rescues the
  point index** (pairwise cos .9995→.001–.004; reconstruction AUC
  .427–.858 → .990–1.000); ridge adds participation-ratio headroom
  (28–30 vs 6.6–19.8) and a small AUC edge. No in-sample inflation
  (disjoint fit/eval split: out-of-sample ridge .990–.999).
- **§4.5 shift instability: confirmed and severe.** The TRAINED R-A
  encoder (uncentered ridge): cos(enc(kg), enc(kg+1)) = .035/−.000/−.044 —
  a policy-preserving constant reward shift rotates the index to
  near-orthogonal. The centered encoder reads 1.000000 exactly.
- **§4.8: the .812 used the analytic point prompt** ẑ ∝ (Σ̂+λI)⁻¹B̄(φ(g))
  at eval (gciql_flow.py:3289–3304, 1134–1135). It validates the dual
  GOAL code. The reward-sample encoder (`density_z_from_reward_samples`)
  is built, unit-tested, trained-through — never rolled out (G29-3).
  (Rev 1: never call this a "Dirac M=1 limit" — see P1; the centered
  encoder has no such limit at all.)
- **§4.2: plain FB's B sees the full observation** (agents/fb/model.py:117,
  184), Density-FB's sees φ(s′) — the dwell "same successor repeated"
  narrative describes φ-space, not the failing baseline's input. Also our
  query support is the batch's own next states (gciql_flow.py:2476), so
  dwell states appear as negatives in proportion — the review's
  double-count objection applies BY CONSTRUCTION.
- Also confirmed: pool-normalized kernel semantics (_delta_phi, no
  density compensation); no tabular ground-truth test for the zero-shot
  de-biasing correction; duplicate-query false negatives structurally
  possible in hard mode (magnitude uninstrumented).

**Review errors (rebuttal notes; Rev 1 pruned after re-verification):**
SF2 is online sample-efficiency work, not zero-shot; DVFB is online
unsupervised RL, not offline zero-shot. ⚠ **RETRACTED (Rev 1): the
Finer-BFM rebuttal.** The explicit contrast "z = E[r(s)B(s)] (FB) or
z = (E[φ(s)φ(s)ᵀ])⁻¹E[r(s)φ(s)] (SFs)" IS in their §3.1 main text (our
first fetch missed it). Covariance-inverse task regression is therefore
established prior art; our novelty is strictly the **softmax-gauge ×
task-inference interaction** — the density objective leaves B
identifiable only up to transformations invisible to the density, so
reusing B directly as a task prompt conditions the policy on an
arbitrary gauge; we identify the dominant additive mode and build the
translation-invariant, reward-shift-invariant encoder. Real must-engage
baselines: **TD InfoNCE** (estimator), **TD-JEPA** (ICLR 2026 oral,
OGBench manipulation — our own codebase forks its framework), **ZSOPL**
(arXiv 2602.01962, evaluated on cube-single/double-play), and (Rev 1)
**OpTI-BFM** (arXiv 2510.20264, ICLR 2026 oral; Rupf, Bagatella,
Vlastelica, Krause): formalizes BFM task inference as reward regression
z_r = argmin_z E[(r−zᵀφ)²] via UNCENTERED second-moment normal
equations, in an online reward-feedback / label-efficiency setting with
confidence-ellipsoid optimism. Not a matched quantitative baseline
(online feedback), but direct prior art for reward-sample task
inference and covariance-corrected regression; our delta vs OpTI-BFM =
the softmax gauge makes B not automatically a valid task basis, and our
encoder is the CENTERED regression theirs is not.

**Already-ours (no new action):** calibration≠extractability wording,
one-seed caveats, failed synthetic reported, dwell-causal retreat
(compendium Part 7) — the review independently endorses positions the
docs already hold; tighten wording only.

---

## P1 — Centered interface (Rev 1: two separate APIs, not one encoder)

**The Rev-1 fact that forces the split:** the centered reward-sample
moment m̂ᶜ = (1/M)Σ_j (B̄(u_j)−μ)(r_j−r̄) with r̄ = (1/M)Σ r_j is
IDENTICALLY ZERO at M=1 (r₁−r̄=0). The centered encoder has no
one-sample / Dirac fallback. Therefore:

- **Point-prompt API** (native goal reaching):
  z_point = √d·norm(B̄(φ(g))−μ) — call it the **analytic centered point
  prompt** (the population moment of a Dirac-style goal reward), NEVER
  the "Dirac M=1 limit" of the sample encoder.
- **Reward-sample API** (the general interface):
  {(u_j,r_j)}_{j=1..M} → ẑ = √d·norm((C+λI)⁻¹ m̂ᶜ), requires M>1,
  realistically M≥256. Native `sample_actions(goal)` must NOT silently
  substitute the point shortcut when a row is reported as
  general-encoder evidence.
- **Constant-reward fallback:** ‖c_r‖ < ε_z ⇒ return z_neutral (or the
  behavior policy); always log ‖m̂ᶜ‖, ‖c_r‖, Var_ρ[r] (near-constant
  rewards otherwise amplify numerical noise through the normalization).

Encode mode `measure_density_z_encode=dual_centered`, implementation
pins (reviewer §2.3–2.4, adopted):
- μ and C under the SAME EMA schedule, stored as serialized param leaves
  (extend the uncentered-EMA pattern; test_sigma_is_ema_uncentered…
  gets a sibling); symmetrize C ← ½(C+Cᵀ); eigenvalue floor / Cholesky
  jitter; log min-eig(C) and the EMA-vs-probe sync errors
  ‖μ̂−E_probe[B̄]‖ and ‖Ĉ−Cov_probe(B̄)‖_F every eval interval.
- ONE pinned base measure ρ_φ = the batch next-state marginal (the
  query distribution) for ALL of μ, C, r̄, m̂ᶜ; the interface states
  that reward samples are expected from ρ_φ (no arbitrary-proposal
  importance correction yet).
- Terminology (binding): **translation-gauge-invariant (additive-gauge)
  centered dual encoder**. The full linear gauge B→AB, F→A⁻ᵀF is NOT
  fixed — isotropic ridge and sphere normalization are not invariant
  under general A. Population encoder exactly invariant to B+b, r+β,
  αr (α>0); the EMA implementation approximates it during joint
  training (say both, in that order).

Arms (cube-single 500k, s0 screen; reviewer §5.1 table adopted):

| arm | prompt construction | actor reward | establishes |
|---|---|---|---|
| ref | uncentered dual point (spec 29) | kernel | .812 reference |
| A | centered point prompt | kernel | mean subtraction alone repairs goal prompting? |
| B | centered dual, M=256 | kernel samples | small-pool general encoder |
| C | centered dual, M=1024 | kernel samples | intermediate pool |
| D | centered dual, M=16384 | kernel samples | high-accuracy general encoder |

Native goal success for the P1 winner is reported under BOTH APIs
(point shortcut AND M∈{256,1024,16384} sample pools) — otherwise P1
validates only another goal shortcut. Pre-registered reading: if A
matches B–D on goals, the conclusion is "mean subtraction suffices for
point-goal prompting", NOT "the inverse covariance is unnecessary for
general reward inference".

Gate: centered ≥ uncentered − noise ⇒ SHIP CENTERED (principled
object); .812 uncentered becomes an ablation row. Seeds s1/s2 + cd
promotion on the winner only.

## P2 — Isolation baselines (Rev 1: two TD InfoNCE controls, renamed LSIF)

1. **TD InfoNCE controls — TWO arms, not one:**
   - **paper-exact TD InfoNCE**: ℓ2-normalized towers, their supports
     (dynamics next state in the immediate positive slot, marginal draw
     in the bootstrap slot, one shared negative pool), goal-contrastive
     actor CE + MSE goal-BC regularizer, their relabeling distribution.
     Answers: how does the published method do in our setting?
   - **matched-extractor TD InfoNCE**: their critic, our φ-space
     outcome representation + external kernel readout + flow/DDPG-BC
     actor. Answers: estimator or extraction?
   Calling one hybrid arm "exact TD InfoNCE" would conflate the two.
2. **Goal-conditioned outcome-space LSIF** (renamed from "plain FB same
   φ"): m = F(s,a,g)ᵀB(φ(x)) — raw-g conditioning matched to the main
   table, LSIF loss, external kernel reward readout
   q_g = E_ρ[m·κ(g,·)], same actor. Everything matches Density-FB
   except LSIF-vs-CE and unnormalized-ratio-vs-normalized-log-ratio.
   This is the privileged-coordinates confound killer. Standard FB
   (native z interface, full obs) stays in the table separately as the
   literature method.
3. **Index×readout matrix, seeded (Rev 1: 3×2, not 2×2)**: index ∈
   {uncentered point, centered point, centered dual} × readout ∈
   {latent B̃ᵀz, external kernel/samples}, cs, ≥3 seeds (5 for camera).
   Centering and dualization must not be bundled — the geometry result
   says centering alone nearly repairs the point index, and the old 2×2
   could not attribute between them. Indispensable new cells if compute
   binds: uncentered-point+external, centered-point+latent,
   centered-point+external, centered-dual+external (uncentered-point+
   latent = archived failure; uncentered-dual+latent ≈ spec-29 R-A/
   latent .368, note the encoder differs).
4. **Affine-invariance rollout probe** (eval-only, after P1 screen):
   r, 2r, .1r, r+1, r−5 → latent cos, E‖Δπ‖, action-value ranking,
   success. Spec-29 arm is the negative control (predicted
   shift-unstable); the centered arm must pass.

## P3 — G29-3, upgraded (Rev 1: reward-sample API only, + the z-search oracle)

Through `density_z_from_reward_samples` + `sample_actions_with_z`,
centered encoder, on the P1 winner — never through the point shortcut:
- families: existing region/dense/hard + max/sum mixtures over two
  goals, target−λ·avoid, per-object conjunctions, negative rewards, and
  SHIFTED copies (r+β) of each — the review's §4.10 list;
- per-reward reporting (the failure-separation battery): disjoint-pool
  reward reconstruction (AUC, Spearman, MSE); index stability
  cos(z_256,z_1024), cos(z_1024,z_16384); policy sensitivity
  E_s‖π(s,z_r)−π(s,z_r′)‖; action-ranking agreement vs candidate-action
  estimates; rollout success/return;
- **search-over-z oracle**: z_search = argmax_z Ĵ_r(z) via CEM/Sobol
  over Z using Density-FB's own reward integral. Interpretation triad:
  closed-form fails + search succeeds ⇒ task-encoder failure; both fail
  + reconstruction good ⇒ policy-family coverage failure;
  reconstruction fails ⇒ task-basis failure;
- **soft-smoothing caveat (Rev 1, from P-soft below):** sharp/max/
  avoid/conjunction families are evaluated under the smoothed reward
  K_ρ,ε r by the soft model — run these cells in HARD mode as the
  semantically exact interface, or report both requested-r and
  effective-K_ρ,ε-r performance and say which;
- **FRE upgrade rule:** FRE (or a matched functional reward encoder)
  becomes MANDATORY if arbitrary-reward prompting is a headline claim
  (title/abstract/primary contribution); minimum comparison set
  otherwise: DeepSets reward-sample encoder + fixed RBF features with
  centered ridge. Appendix-diagnostic G29-3 ⇒ discussion-only is
  acceptable.

## P-soft — The soft variant is a kernel-smoothed occupancy (state it)

Population operator: with K_ρ,ε(du|x) = k_ε(x,u)ρ_φ(du)/∫k_ε(x,v)ρ_φ(dv),
  T_K^π ν = (1−γ)·E_{x′∼P_φ}[K_ρ,ε(·|x′)] + γ·E[ν(·|s′,a′)],
whose fixed point ν_K^π is the kernel-smoothed discounted occupancy —
a valid successor object, but reward integration under it satisfies
  ∫ r dν_K^π = ∫ (K_ρ,ε r) dν^π:
the soft model evaluates (and its actor optimizes) the SMOOTHED reward,
not r. Sums stay linear; sharp boundaries, conjunctions, max-mixtures
and avoid-terms are altered. Consequences: (i) present soft as a
kernel-smoothed successor formulation, not "label smoothing"; (ii) the
base measure of K is kernel×pool density (already stated); (iii) hard
mode is the semantically exact reward interface — see the G29-3 caveat.

## P4 — Cheap hygiene items (Rev 1 refinements folded in)

- Tabular ground-truth unit test for the zero-shot de-biasing correction
  against the exact matrix successor measure ((I−γP^π)⁻¹ convention as
  appropriate) on a toy MDP, covering all SIX conventions: measure start
  index (s_t vs s_{t+1}), reward attachment (s_t / s_{t+1} /
  transition), terminal & absorbing handling, the 1−γ normalization,
  the finite-pool correction, hard AND soft immediate kernels.
- Duplicate queries — instrumentation FIRST, no engineering yet:
  exact-duplicate and ε-duplicate fractions per batch, irreducible-CE
  estimate, mass aggregated over ε-equivalence classes. Rationale
  (reviewer §6.1, correct): identical queries get identical logits, so
  gradients can aggregate to the right state-level mass; multi-positive
  targets only if the measured rates/gradients show a real issue.
- G4 naming: "occupancy-ranking fidelity", never "calibration".
- Simulator MC action-ranking probe (§4.12) — Tier-1, spec only for now.

## P5 — Paper/compendium edits (immediate, no compute; Rev 1 additions ★)

- **Positioning**: hard estimator reduces to the TD InfoNCE categorical
  update (cite, proposition with the "matched conventions" hedge + the
  CORRECTED residual-delta list — TD InfoNCE is offline GC-RL; the
  delta is the reward-prompted latent family / reward→index inference /
  validated arbitrary-reward control); soft = kernel-smoothed successor
  formulation (P-soft fixed point ★); contributions = outcome-space
  successor density + offline reward-integrating extraction + gauge
  analysis + corrected interface.
- **Geometry restatement** (tex §8.2, compendium §6.6), reviewer's §7
  wording adopted: the uncentered second moment is dominated by a large
  additive mean direction (33–103× the centered signal), invisible to
  the density softmax but dominant in normalize(B(g)); after centering
  the covariance has effective rank 6–21/50 and point prompts separate;
  the inverse covariance additionally expands low-variance task
  directions for reward regression. Propositions: reduction-to-TD-
  InfoNCE (hedged) + translation-gauge/affine-reward invariance of the
  centered encoder (★ scoped: additive gauge only; full linear gauge
  not fixed; population exact, EMA approximate).
- **★ API split in the text**: analytic centered point prompt vs
  reward-sample encoder; the r₁−r̄=0 no-one-sample-limit fact; the
  constant-reward fallback; replace every "Dirac M=1 limit" phrasing.
- **★ Citations**: add OpTI-BFM (2510.20264, with the task-inference-
  as-reward-regression relation and our gauge/centered delta); DELETE
  any Finer-BFM "not found" rebuttal remnants (tex never had one; check
  compendium/rebuttal notes).
- **Scope sentence**: rewards measurable on the chosen outcome
  coordinates and supported by ρ_φ — not "any reward"; B(s′) vs B(φ(s′))
  stated for every baseline; explicit φ/ε privileged-information
  statement (§4.9) pending P2.2.
- **Dwell**: Option-B wording in tex §2 (done in the first P5 pass).

## Priority (Rev 1: first GPU wave is PARALLEL)

P5 edits (now) → **first GPU wave, one seed each, concurrent:**
(a) P1 centered arms on cube-single; (b) paper-exact TD InfoNCE on
CUBE-DOUBLE; (c) goal-conditioned outcome-space LSIF on CUBE-DOUBLE —
cd for (b)/(c) because cube-double is the load-bearing separation.
**Promotion wave** (→3 seeds) only if: centered stays competitive AND
exact TD InfoNCE does not trivially reproduce the soft result AND
matched-φ LSIF stays substantially below Density-FB. → **Attribution
wave**: the P2.3 minimal matrix. → **General-interface wave**: G29-3
through the reward-sample API with M-curves, affine copies,
reconstruction, action ranking, z-search oracle, rollouts. Q4 topline
(spec 29) stays fill-in. NOT adopted: 5-seed-everything now (3 now, 5
at camera-ready), SF2/DVFB/InFOM as quantitative baselines (protocol
mismatch — cite and discuss), simulator MC ranking as a blocker.

## Decision tree (reviewer §9, adopted verbatim)

| Outcome | Interpretation |
|---|---|
| Soft Density-FB beats exact TD InfoNCE; matched-φ LSIF remains poor; centered G29-3 works | Strongest outcome. Defensible new method and analysis. |
| Exact TD InfoNCE matches hard but soft is better; matched-φ LSIF remains poor | Still a viable paper. Position soft kernel occupancy, offline extraction, and gauge-corrected prompting as contributions. |
| Exact TD InfoNCE matches soft; matched-φ LSIF remains poor | Method novelty becomes narrower. Paper is an extension/application of TD InfoNCE plus latent-interface analysis. |
| Matched-φ LSIF closes most of the gap | The original comparison was largely driven by outcome projection and actor readout. The normalization story weakens substantially. |
| Centered point works but centered reward-sample G29-3 fails | Valid goal-conditioned method, but not a general reward-prompted BFM. Remove broad arbitrary-reward claims. |
| Centered reward reconstruction works, z-search works, closed-form fails | Task encoder remains the problem. |
| Reward reconstruction works, but neither closed-form nor z-search succeeds | The learned policy family lacks task coverage. |
| G29-3 succeeds across mixtures, avoidance, conjunctions, negative rewards, and shifts | The general reward interface becomes a central and defensible contribution. |
