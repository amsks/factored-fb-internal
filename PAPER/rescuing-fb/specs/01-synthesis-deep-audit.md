# Deep audit + synthesis — why bilinear-GCIQL wins, why FB fails, and what to build

*2026-07-16. Produced from: full code audits of `tools/wandb_mode_shim/gciql_flow.py` (jax) and
`agents/fb/` (torch), the living CSVs (`results/*.csv`), the experiment inventory, all PAPER/
docs (bridge, atlas, oodgrid, ogbench-sota, rlbrew, misalignment), and two literature sweeps
(FB/zero-shot frontier; bilinear/low-rank critics + offline GCRL + flow actors). All internal
numbers are cube-single-play state @500k 3–13 seeds unless noted.*

---

## 0. TL;DR

1. **The mechanism (Q1).** Bilinear+flow wins because three independently-verified pieces
   compose: (i) scalar hindsight IQL is a *contractive, task-weighted* grounding (the measure
   TD is neither); (ii) the rank-d ψ(g) bottleneck forces one shared action-basis φ(s,a) to
   carry the (tiny, <1%-of-Q-variance) action-advantage signal *for all goals at once*, so it
   accumulates coherently instead of washing out — buying superb on-support action ordering
   (rerank 0.91–0.96 vs MLP 0.57) and free goal generalization (dGOAL ≈ 0 vs MLP −16 to −24),
   at the cost of per-(s,g) value calibration (in-hand AUC 0.39) and a structured off-manifold
   spurious maximum; (iii) the distillation-anchored flow actor consumes exactly the ordering
   and never visits the peak. Every clause has a named literature anchor (below).
2. **The FB failure (Q2).** FB's grounding — policy-conditioned measure TD — is provably
   non-contractive (Zheng, Jayanth & Eysenbach 2026), occupancy-weighted where cube needs
   task-weighted supervision (your own temporal-misalignment doc), and its fixed point on cube
   is bad (your iqlfade result). The fix is not another regularizer on the measure: **keep FB's
   API, swap the grounding.** The assembled method already exists in pieces in your own results:
   bilinear GCIQL (0.904) + z-conditioned actor (G4, 0.921) + dataset-moment inference (E1a,
   0.955) + optional low-λ measure auxiliary (E2, free at λ≤1). Package it as one agent.
3. **Cross-OGBench (Q3).** The recipe is not yet portable because of two *actor-side* artifacts
   (goal-agnostic BC flow anchor breaks antmaze; off-support peak breaks DDPG+BC only where the
   behavior action-manifold is narrow) and one unexplained domain (scene). Unify extraction
   (z-conditioned FQL vf, or selection-based extraction à la IDQL — your reranking result says
   selection is near-optimal), then run the atlas instruments on scene before proposing a fix.

---

## 1. Audit verdicts — design decisions vs established wisdom

### 1.1 Things the pipeline does right (and that the comparison's validity rests on)

- The FB↔GCIQL comparison is genuinely **objective-isolated**: both sides share the goal-agnostic
  BC flow + one-shot distilled actor + conditional Q-normalization pattern; the bridge claim
  "D1 (objective) is the whole gap" survives code audit.
- Reward convention (0/−1, masks at goal), goal mixes (value: 0.2 cur / 0.5 geom-future /
  0.3 random; actor: uniform future), expectile 0.9, τ=0.005 — all stock OGBench, no drift.
- The assert lattice in `gciql_flow.py:1153-1331` makes invalid knob compositions fail loudly.

### 1.2 Deviations from canonical recipes that matter (jax side)

| Finding | Where | Why it matters |
|---|---|---|
| **BC flow vf is goal-agnostic** | `gciql_flow.py:1512,632` | The distillation anchor is the goal-*marginal* behavior flow; ALL goal-dependence comes from the Q term fighting a goal-independent anchor. Canonical FQL conditions the vf on the goal. Benign on cube (unimodal scripted play); the prime suspect for the antmaze flow collapse (0.24–0.33 both critics) — direction-multimodal marginal fights direction commitment. |
| No output bound, V has no target net, expectile pushes V up | `gciql_flow.py:523-544` | The ψ(s,g) divergence (q_mean crossing positive) is structurally unguarded; the ψ(g) bottleneck is plausibly the only stabilizer (see DR3, §2.3). |
| Bilinear ≈ 2× trunk params vs MLP, interaction only at the readout | `networks.py:327-389` | Param-count + interaction-depth confound. Control owed (already flagged in ogbench-sota). |
| −q normalization only when bc_coeff>0 | `gciql_flow.py:684` | bc_coeff=0 ablations change loss scale discontinuously, not just remove distillation. |

### 1.3 Deviations from canonical FB (torch side)

| Finding | Where | Assessment vs literature |
|---|---|---|
| **Twin-head target = mean, not min** (pessimism 0) | `agents/fb/agent.py:829-844` | Optimism delta vs canonical FB. Jeen et al. (VC-FB, NeurIPS 2024) diagnose exactly this: FB catastrophically overestimates M/Q for OOD actions on narrow data — and cube scripted play is narrow. Cheap test (§4 T7). |
| Ortho diagonal term inert (B hard-normalized to √d sphere) | `agent.py:715` + `nn_models.py:404` | ortho_coef=1000 is pure off-diagonal decorrelation. Not a bug, but the "keep norm up" clause of the canonical loss is dead weight; document it. |
| z-mix = 50% sphere / 50% B(permuted buffer next-obs) | `agent.py:380-409` | Matches Touati recipe. Behavioral-task-sampling (Bendib et al. 2026) says sampling z from *achieved trajectories* helps on OGBench cube — your zher tried the goal-relabel half (null), but not trajectory-level z-embedding (FB-CPR style). |
| `onestep` cell feeds **zeros for z** + SARSA | `agent.py:447-451` | This is NOT one-step FB (Zheng et al. 2026: keep z-conditioned F, a′∼π_β target, then one BC-regularized improvement step; +24% avg, validated on OGBench cube). The literature's #1 FB fix is **untested** in this repo. |
| Actor grads leak into F (no no_grad on forward map) | `flow_bc/agent.py:92` | Currently masked by optimizer zero_grad ordering — fragile invariant, worth a guard. |
| Eval actor is stochastic (fresh noise, `mean` ignored) | `flow_bc/model.py:55-57` | Adds eval variance to every FB number; harmless but worth knowing. |

### 1.4 Internal claims that need revision before print

- The CLAUDE.md/rlbrew thesis ("under-resolves the interaction variable") is undermined twice
  by your own results: obs[18] is ~fully redundant with the state (AUC 1.0), and the winning
  critic ranks in-hand reachability *below chance* (0.39) — cross-state value fidelity is
  neither necessary nor what carries control. The misalignment doc's temporal story
  (discount-mass vs gate horizon) is the surviving version; rlbrew's abstract claim as stated
  is not.
- The bridge tex still carries the retracted "occupancy is arm-dominated" mechanism paragraph.
- FB anchors vary by protocol (34@500k / 48@1M / 0.32 own-harness / 0.42 published) — always
  cite the protocol.

---

## 2. The mechanism (Q1) — a three-factor compatibility story

Facts any explanation must produce: bilinear+flow 0.904–0.947 > MLP+flow 0.717–0.747;
bilinear+DDPGBC 0.51 < MLP+DDPGBC 0.70 (cube); antmaze flips (bil+flow 0.24, bil+DDPGBC 0.84);
scene bilinear loses everywhere; rerank bil 0.91–0.96 / MLP 0.57 / FB collapses with N;
bilinear in-hand AUC 0.39 (anti-calibrated U) but advantage partially shared with MLP (+0.22);
dGOAL free for bilinear only; ψ(s,g) diverges 2/3 seeds; d5 collapses, d28–512 flat.

### 2.1 Factor 1 — what scalar hindsight IQL trains

Expectile TD on dataset actions only: the training task IS on-support ordering + magnitude,
under a goal distribution concentrated on controllable futures (50% geometric on-trajectory)
plus random negatives (30%). Contrast the measure loss: unweighted density regression under
ρ⊗ρ. On cube, the two weightings diverge enormously — your misalignment doc shows 82% of
γ=0.99 discounted mass lies beyond the grasp gate's control horizon, and grasp-stability
counterfactuals (17 drops vs 12,127 places) are unsupported in ρ. On locomotion they roughly
coincide (any state is a plausible goal; occupancy is spread) — which is why FB is respectable
on ExORL/antmaze (0.59) and bad on cube (0.42). **Measure TD is occupancy-weighted; hindsight
scalar TD is task-weighted. Manipulation is where they disagree.**

### 2.2 Factor 2 — what the rank-d ψ(g) bottleneck does: advantage pooling

Your action-allocation probe: action-dependence is <1% of Q's variance for every critic. For a
monolithic MLP the expectile loss is therefore ~99% about the baseline U(s,g); the action
residual can be fit lazily or with degenerate features — and TD's implicit rank collapse
(Kumar et al., ICLR 2021) predicts exactly an MLP whose returns are fine but whose
action-ordering is noise. Measured: MLP rerank 0.57–0.59, barely above the 0.06 floor's
argmax-over-32 baseline.

The bilinear head cannot do this. With Q = φ(s,a)·ψ(g)/√d, *every* goal's advantage must be a
linear functional of the same d-dim φ(s,a). The tiny per-sample action signal from *all* goals
accumulates in one shared basis — a massive cross-goal parameter-sharing regularizer on the
advantage. Three measured consequences, all confirmed:

1. **Superb on-support ordering** (rerank 0.91–0.96 ≈ the trained actor; fresh-actor-vs-frozen
   0.88 — intrinsic, not co-adaptation).
2. **Sacrificed baseline calibration**: U(s,g) = E_μφ·ψ(g) is *constrained to the same rank-d
   form*, so the head trades U-fidelity for advantage structure — exactly your Finding 7a
   dissociation (MLP U AUC 0.82 / bilinear U 0.40, uncorrelated baselines, partially shared
   advantages). The dissociation is not a paradox; it is the factorization's budget allocation.
3. **Free goal generalization**: a new goal is a reweighting of the same basis (BVN's
   compositional-generalization mechanism, Hong et al. ICLR 2022) — your dGOAL ≈ 0 finding.

Literature anchors: BVN (same form, same OOD-goal transfer, online); FLAP (linear-in-shared-
basis heads extrapolate); Yang et al. 2020 (control Q-matrices are near-low-rank; imposing rank
is a structured denoiser); Stop Regressing + SimbaV2 (bounded/structured critic outputs beat
raw MSE MLPs under TD). Note BVN's working factor was φ(s,g)·f(s,a) *online*, while your
offline result inverts (ψ(g) stable, ψ(s,g) diverges) — regime-dependence worth a paragraph.

### 2.3 The two costs of the bottleneck, and why they're survivable

**Cost A — off-manifold spurious maximum** (Finding 1: DDPG action valued above every reference
79% of the time, +2.14; equal gradient norms — a peak, not a slope). Principled origin:
−V* of goal-reaching is a quasimetric (QRL); a symmetric rank-d inner product provably cannot
represent it (MRN/IQE), and low-rank fits of distance-like matrices carry an irreducible
*additive, spatially structured* residual that inflates where embedding norms are large — the
far field / off-manifold region. The representation error doesn't vanish; it *relocates* to
where the flow actor never looks.

**Cost B — backup stability lives on the bottleneck.** ψ(s,g) (bilinear_sg) removes the
low-rank goal bottleneck and 2/3 seeds diverge late (value_loss 0.15→1–5, q_mean crosses
positive on a ≤0-bounded value); the stable seed lands at MLP level (0.69). DR3 (Kumar et al.
2022) is the named mechanism: TD implicitly grows the dot product between features of
bootstrap-adjacent pairs; putting s in both factors opens a shared-s co-adaptation channel on
Q = φ·ψ, and DR3's predicted signature on this form is literally q_mean drifting positive.
ψ(g) closes the channel. This upgrades "the goal bottleneck is what keeps the backup
contractive" from observation to mechanism — and it is testable (§4 T2).

### 2.4 Factor 3 — extraction determines which side of the trade you get

- The FQL-style actor (BC-flow anchor + distillation) consumes the critic as a **ranker over
  near-support actions** — the IDQL/SfBC selection operator, which your reranking probe
  measures directly. One-step-RL ≡ critic-regularization (Eysenbach et al. 2023) says this
  extraction end is the safest under distribution shift.
- DDPG+BC consumes the **landscape** and climbs to the off-support peak (BCQ extrapolation
  error) → bilinear+DDPGBC 0.51 on cube, intrinsic (fresh DDPG+BC vs frozen bilinear 0.52).
- **Why DDPG+BC is fine on antmaze with the same bilinear head (0.837)**: hypothesis — the
  peak is only *reachable/fatal* when the behavior action-manifold is narrow (cube scripted
  pick-place). Antmaze data comes from noisy-expert locomotion with broad action coverage, so
  Q-ascent stays effectively on-support. Untested; cheap to test (§4 T3).
- **Why flow fails on antmaze for BOTH critics (0.24/0.33)**: the audit found the BC vf is
  goal-agnostic. Antmaze's behavior marginal is direction-multimodal; the anchor fights the
  direction commitment the Q term must impose. Cube's marginal is unimodal so the anchor is
  benign. Untested; cheap to test (§4 T4). If confirmed, the domain actor-flip is an artifact
  of this port, not a fact about bilinear critics — and fixing it is the road to Q3.

### 2.5 One-sentence answers

- **Why bilinear?** It reallocates a fixed representation budget from value calibration to
  cross-goal-pooled advantage structure, which is the only part of Q the extraction consumes.
- **Why only with flow?** Flow (= anchored selection) reads only the on-support ordering;
  gradient-ascent actors read the off-manifold residual, which is where the factorization
  parks its irreducible error.
- **Why cube?** (i) The benchmark's binding shift is the goal shift — free for the
  factorization; (ii) success hinges on post-grasp per-step action selection (advantage
  geometry), not long-horizon value shaping; (iii) narrow play-data action manifold makes the
  competing extraction (DDPG+BC) fail, making the flow×bilinear cell uniquely visible.

### 2.6 A reframing worth putting in the paper

With z = ψ(g) and scalar TD over the hindsight-goal reward family, φ(s,a) is forced to satisfy
φ·ψ(g) ≈ r_g + γ φ′·ψ(g) for all dataset goals simultaneously: **bilinear GCIQL is implicit
joint learning of a d-dim reward basis (ψ) and its value features (φ) by scalar hindsight TD —
"successor features without feature TD."** Your own SF ladder shows every *explicit* vector-TD
estimator of the same structure collapses (sf-joint 0.12, feature-TD ≤0.16, rewreg ~0.00,
SimpleSF-grounded 0.00; SimpleSF-arch scalar-TD 0.38): the structure is fine, the estimator is
what fails. This connects to Ollivier 2025 (*Which features are best for SFs?*) and gives the
paper its thesis: **the successor-measure/SF API is right; the density-style TD estimator is
wrong (for manipulation). Ground the same representation with scalar hindsight TD.**

---

## 3. Why FB specifically fails on cube — the consolidated indictment

1. **Non-contraction**: the FB operator with a′∼π_z(F) targets is provably not a γ-contraction
   and has non-unique fixed points (Zheng, Jayanth & Eysenbach, arXiv 2602.11399). Your
   anneal-iqlfade result is the empirical twin: even *starting from the 0.93 basin*, pure
   measure TD destroys it (→0.05) — the fixed point itself is bad, not the optimization path.
2. **Occupancy-vs-task weighting** (§2.1): supervision lands where ρ has mass, not where the
   task needs resolution; cube's control-relevant futures (grasp-stability counterfactuals)
   are unsupported (17/1M events).
3. **Rank-d over the wrong object**: exact FB needs d ≥ |S×A| (same paper); the *fixed-policy*
   measure is benignly low-rank but FB needs one rank-d family covering the optimal policies
   of all rewards. Your d-sweep on the IQL side (d28–512 flat) shows the *task family's*
   demands are low-rank — it's the measure object, not the task, that outgrows d.
4. **Anti-informative action signal**: FB Q is 99.8% state-driven and its rerank *collapses*
   with N (0.52→0.17) — the mirror image of bilinear. The measure loss's action-dependence is
   diluted across ρ⊗ρ instead of pooled per-goal.
5. **Port-level optimism**: mean-not-min targets + no conservatism, on narrow data — the exact
   VC-FB failure mode.
6. **Learned-B bottleneck** (your z=g result: identity-B + IQL 0.51 vs learned-B FB-IQL 0.20)
   — under scalar grounding, ψ/B trained through the TD loss fixes this (that IS bilinear
   GCIQL's ψ).
7. Fitting the measure well is insufficient anyway: fbz cut the measure TD-residual 24× and
   moved success only to 0.255.

**Conclusion: stop trying to rescue the measure estimator on manipulation. Keep the API.**

---

## 4. Quick tests (ranked by information/cost; forward-pass ones first)

| # | Test | Cost | Prediction / what it decides |
|---|---|---|---|
| T1 | **Goal-smoothed MLP rerank**: rerank with Q̄(s,a,g) averaged over K nearby goals (goal-grid neighbors) | hours, forward passes | If MLP rerank jumps toward bilinear, the advantage-pooling mechanism (§2.2) is confirmed causally-ish: MLP's action signal exists but is per-goal noise. If unchanged, MLP's action info is absent (pure rank collapse). Either way names the MLP failure. |
| T2 | **DR3 statistic across the exp5 50k-checkpoints**: normalized E[φ(s,a)·φ(s′,a′)] for bilinear / bilinear_sg / MLP + q_mean overlay | hours, forward passes | bilinear_sg shows runaway feature co-adaptation tracking the q_mean sign-crossing; bilinear flat. Turns the "bottleneck = stabilizer" claim into a mechanism figure. |
| T3 | **Off-ref peak + action-coverage on antmaze/scene checkpoints** (rerun Finding-1 instruments) | hours, forward passes | Antmaze bilinear has no fatal off-support peak (or the peak lies within the broad data manifold) → explains the DDPG+BC domain flip. Scene numbers become diagnostic. |
| T4 | **Goal-conditioned BC vf on antmaze** (bilinear+flow, 3 seeds, 500k) | ~1 day GPU | If it recovers toward DDPGBC-level (0.8+), the actor flip was a port artifact and the universal recipe is bilinear + goal-conditioned FQL. Highest-leverage single training run on the board. |
| T5 | **Param/fusion controls**: (a) MLP with matched params; (b) late-fusion two-tower (same towers, MLP readout instead of inner product) | ~1 day GPU, 3 seeds each | Separates "factorized towers" from "inner-product readout" from "param count". The bridge paper needs (a) anyway. |
| T6 | **Zero-shot generality of E1a**: infer z for synthetic rewards (two-goal mixtures, lift-anywhere, avoid-region) and roll out | hours | Decides whether the method claim is "zero-shot RL" or "zero-shot goal-reaching". Determines the paper's scope honestly. |
| T7 | **FB min-vs-mean + light conservatism** (pessimism_penalty>0 / VC-FB-style) | ~1 day GPU, 3 seeds | The port's optimism delta is the literature's #1 diagnosed FB failure on narrow data. Cheap, publishable either way. |
| T8 | **Faithful one-step FB** (z-conditioned F, a′∼dataset, advantage-weighted extraction — Zheng recipe; the repo's `onestep` zeroed z and never tested it) | ~1–2 days | The literature's main contraction fix, validated on OGBench cube elsewhere (+24% avg). If it works here it's a strong baseline for the paper; if not, strengthens the "grounding swap" argument. |
| T9 | **Rank of the true advantage**: estimate rank of A*(s,·,g) across the goal grid from counterfactual rollouts (atlas banks + sim) | ~1 day CPU-heavy | Is the low-rank advantage assumption *true* on cube (justifying the architecture) or merely useful? Feeds the theory section. |

---

## 5. The algorithmic contribution (Q2): **RG-FB — Reward-Grounded Forward-Backward**

Keep everything that makes FB *zero-shot* (the z-space, the task encoder B/ψ, moment inference,
z-conditioned policy). Replace the one thing that is broken (measure-TD grounding) with the one
thing you've proven works (hindsight scalar expectile TD). Every component is individually
validated in your own tables:

| Component | Spec | Validation |
|---|---|---|
| Critic | Q(s,a,z) = φ(s,a)·z/√d, z = ψ(g) during training (φ NOT z-conditioned — fbz costs ~6 pts: 0.848 vs 0.904) | bilinear-d50 0.904 |
| Grounding | GC-IQL expectile TD, OGBench goal mix, 0/−1 rewards | the whole bridge program |
| Task encoder | ψ(g) learned through the TD loss (this is what fixes FB's learned-B bottleneck) | z=g vs learned-B ladder |
| Actor | flow (FQL-style) conditioned on z=sg(ψ(g)); **goal/z-conditioned BC vf** (fix the port's goal-agnostic anchor) | G4 z-cond 0.921; vf fix = T4 |
| Inference | z = E_dataset[r·ψ(s′)], rescaled to ‖ψ(g)‖-scale (raw-ψ hygiene, not √d) | E1a 0.955 (= exact-goal anchor) |
| Measure semantics (optional) | low-λ scale-normed measure auxiliary (λ≤1, stopgrad-φ variant) to keep an honest successor-measure reading of φψᵀ | E2: 0.86–0.92, free |
| Conservatism (optional) | min-aggregation / small pessimism on targets | T7 |

What is genuinely new to run: the **single end-to-end agent** (one training run producing one
checkpoint that is evaluated only zero-shot on all 5 tasks), the generality probe (T6), and the
cross-domain replication. Expected headline: **zero-shot ≈ 0.95 on cube-single, ~2× the best
published zero-shot number (FB 62.0 RGB / your FB 0.42 state), matching or beating the
supervised GCIQL SOTA it is grounded by.**

Positioning: "Successor measures are the right interface and the wrong estimator on
manipulation. We keep the FB API and swap its grounding for task-weighted scalar TD; the
resulting representation is an implicitly-learned reward basis + value features (SFs without
feature TD), and it inherits bilinear GCIQL's advantage-geometry win." The FB-side ablations
(one-step FB T8, conservative FB T7, BREEZE-style extraction) become the baselines table.

---

## 6. Cross-OGBench SOTA plan (Q3)

1. **Unify extraction first** (it's the portability blocker, not the critic):
   - Fix the goal/z-conditioned vf (T4). If antmaze flow recovers → one actor everywhere.
   - Fallback/alternative: **selection-based extraction** (IDQL-style: N≈32 BC-flow proposals,
     argmax frozen Q, distill into a one-shot actor for deployment). Your rerank numbers say
     this is near-optimal on cube (0.947–0.96) and it is immune to both failure modes (peak,
     anchor). The in-flight antmaze/scene rerank audits decide this empirically.
2. **Scene** (bilinear loses both actors, 0.26–0.29 vs 0.52): diagnose before fixing — run the
   atlas instruments (rerank N-sweep in flight, off-peak probe, a scene OOD-grid). Candidate
   fixes in order: (a) Schaul-style split so bilinear TD doesn't train a shared rep (validated
   on HIQL: 41.3 vs pub 15); (b) factored per-object goal heads (scene goals differ from start
   in 1–2 factors; a single rank-50 ψ over full-state goals may under-resolve combinatorial
   goals); (c) asymmetric residual head (MRN/IQE) where the symmetric form's expressivity gap
   binds.
3. **Multi-cube / puzzle**: run bilinear-d50 + fixed actor as-is first (no published method
   works there; even +10 pts on cube-double is a frontier claim).
4. **Owed controls** before any SOTA claim: param-matched MLP (T5), cube DDPGBC α rerun,
   8-seed/1M confirmations, pixel bilinear (DrQ path exists).
5. **Paper split**: (i) mechanism/atlas paper (the reranking instrument + compatible
   policy-learning system story — no published competitor per the lit sweep); (ii) RG-FB
   zero-shot method paper; (iii) the SOTA recipe as the empirical backbone of (ii) or separate.
   Revise the rlbrew interaction-variable framing per §1.4 before anything goes out.

---

## 7. Literature index (for the related-work sections)

Bilinear/low-rank critics: BVN (Hong et al. ICLR'22, 2204.13695) · MRN (Liu et al. AAAI'23,
2208.08133) · Harnessing Structures (Yang et al. ICLR'20, 1909.12255) · FLAP (2101.04750) ·
Stop Regressing (Farebrother et al. ICML'24, 2403.03950) · SimbaV2 (2502.15280).
TD pathologies: DR3 (2112.04716) · implicit under-parameterization (2010.14498).
Extraction: FQL (2502.02538) · IDQL (2304.10573) · SfBC (2209.14548) · one-step≡critic-reg
(2307.12968) · Brandfonbrener one-step (NeurIPS'21) · BCQ (1812.02900) · EXPO (2507.07986).
GCRL: OGBench (2410.20092) · HIQL (2307.11949) · value-learning-bottleneck (2406.09329) ·
GOAT (2305.18882) · CRL (2206.07568) · QRL (2304.01203) · IQE (2211.15120).
FB/zero-shot: Touati & Ollivier '21 · Does ZS RL Exist (2209.14935) · FB-CPR/Meta Motivo
(2504.11054) · **Zheng, Jayanth & Eysenbach 2026 (2602.11399, non-contraction + one-step FB)**
· BREEZE (2510.15382) · VC-FB (2309.15178) · Cetin et al. AR-features (2412.04368) · ZOL
(2602.01962) · behavioral task sampling (2604.25496) · HILP (2402.15567) · PSM (2411.19418) ·
TD-JEPA (2510.00739) · DiSPOs (2403.06328) · Ollivier SF-features (2502.10790) · switching
successor measures (2605.13207).
Published FB-on-cube anchor: TD-JEPA reports FB 62.0 / cube-single RGB, 1.2 cube-double —
nobody has a mechanism-level diagnosis; the atlas program is unduplicated as of 2026-07.

---

## POST-HOC CORRECTIONS (probe results against this synthesis)

- **2026-07-16, T2**: §2.3 Cost B named DR3 φ-co-adaptation as the ψ(s,g)-divergence mechanism.
  REFUTED as operationalized: normalized φ·φ′ is saturated (~0.99) for all archs and highest for
  stock bilinear. The measured divergence channel is **ψ(s,g)-leg norm/alignment runaway**
  (E‖ψ‖² 1083/2355/3789 monotone in divergence severity; q_max +778/+1177; φ sane). Revised
  mechanism: **global coupling as implicit regularization** — ψ(g) forces any Q-inflation
  through globally-shared factors; ψ(s,g) opens a local (s,g) pocket for expectile-TD's upward
  pressure (no bound, no V-target). SimbaV2/norm-bounding family, not DR3. See
  `analysis/rescuing-fb/t2_notes.md`.
- **2026-07-16, T3**: §2.4's off-support-peak boundary condition CONFIRMED (cube frac 0.639 vs
  antmaze 0.003) but the *data-coverage* explanation REFUTED (dim-corrected narrowness ratio
  only 1.11×). The peak's domain-specificity needs a value-geometry account. See
  `analysis/rescuing-fb/t3_notes.md`.
- **2026-07-16, T1**: §2.2's advantage-pooling mechanism REFUTED in its eval-time form: MC
  goal-smoothing does not rescue the MLP rerank (0.616 → 0.408–0.580, monotone DECLINE in σ;
  pre-registered success was ≥0.70) while bilinear is invariant at 2 cm; random-goal control
  valid for both (0.148/0.040). Sharpened reading: a 2–4 cm goal perturbation is already
  distribution shift for the MLP head (same pathology as its dGOAL −16…−24), and the MLP's
  action-ordering signal is absent/systematic, not noisy. If pooling operates, it operates at
  TRAINING time via the ψ(g) parameter-sharing constraint — T5 (late-fusion + param-matched
  controls) is the discriminating test. See `analysis/rescuing-fb/t1_notes.md`.
