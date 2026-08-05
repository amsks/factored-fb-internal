# ICLR roadmap: picking apart the bilinear value form

**Date:** 2026-08-04. **Anchor prior work:** Bilinear Value Networks (BVN), Hong, Yang,
Agrawal, ICLR 2022, arXiv:2204.13695. **Our state:** PAPER/decoding-bilinear/ (theory +
metric probes + selector experiment + transfer analysis, commits through `f94049e`).

---

## 1. What BVN is

- **Form:** `Q(s,a,g) = f(s,a)^T phi(s,g)` — the second field conditions on **s and g**,
  not g alone. Their input-grouping ablation says processing s with g is *important* and
  the `f(s,a)^T phi(g)` factorization (Schaul et al. 2015 — **our form**) is "too
  restrictive."
- **Setting:** online DDPG+HER, Fetch + Shadow-hand suites, sparse reward, gamma=0.98,
  16-dim latents (3-dim already beats monolithic).
- **Evidence:** end-to-end success curves (better in 4/8 Fetch, 6/8 hand tasks), OOD-goal
  splits (left/right, near/far) with faster finetuning, a toy U-maze PCA visualization,
  composition ablation (dot ~= L2 head, linear worse), input-grouping ablation.
- **Their explanation** (informal, one paragraph): combinatorial generalization — unseen
  tuples `(s_A, a_A, g_B)` are covered because each factor saw its own marginal. This is
  the matrix-completion intuition, asserted but never isolated or measured.
- **Their admitted gaps:** disentanglement is incomplete (freezing f underperforms);
  scalar ambiguity `(lambda f, phi/lambda)`; no theory; and verbatim: they do not explain
  *why certain tasks benefit more than others*.

## 2. What we did that BVN did not

| Axis | BVN | Us |
|---|---|---|
| Setting | online DDPG+HER | offline expectile TD (GCIQL), no exploration confound |
| Theory | none | negated value = quasimetric; bilinear head = rank-d factorization of K*=gamma^d*; unified links to CRL/VIP/QRL (log link, polarization/Schoenberg, self-subtraction) |
| What the embedding learns | toy PCA picture | probe battery: temporal corr, triangle violations, symmetry, cosine/norm, RSA across 5 method families → **the winner is the geometric outlier; metric story refuted** |
| Mechanism isolation | success curves only | best-of-N selector (no actor training): ordering alone reproduces the trained policy at N=8; dose-response in N; candidate-source crossing; CRL control separating *form* from *target semantics* |
| Mechanism elimination | none | boundedness (tanh-kappa: ≤3/19.5 pts), norm concentration (psi-LN: nil), value calibration (anti-correlated with success), off-distribution coherence interaction (P2 2x2) |
| Grounding | latent-dim ablation only | environment kernel spectrum measured from data (eff-rank ~10); learned value matrix expresses the same spectrum capacity-independently; success saturates at d† ≈ dims95 |
| When/why it fails | unexplained | headroom censoring, goal-observable channel (with causal eval-side repair, +0.23), kernel range compression / dwell — each with a confirmed kill-test |
| Baselines | monolithic UVFA | monolithic + CRL + QRL + VIP + FB across 3 benchmarks, 2+ datasets each |

## 3. What BVN did that we have not (the gaps our next arms fill)

1. **The (s,g)-conditioned second field.** BVN's central architectural claim directly
   contradicts our theory: phi(s,g) breaks the fixed rank-d factorization of the
   state-goal matrix (each state gets its own goal-map), which our mechanism says is the
   *source* of off-distribution ordering coherence. Their finding is from online HER;
   ours is offline. This is a genuine, falsifiable disagreement — the best kind.
2. **Systematic OOD-goal splits.** They partition goal space (left/right, near/far); we
   only have the eval-tasks-vs-play-distribution contrast.
3. **Composition ablation under one objective.** They compare dot vs L2 heads under the
   same DDPG loss. Our L2 evidence comes from VIP — a different objective entirely.
4. **Multi-task/dexterous breadth** (Shadow hand). Not a priority for our question.

## 4. The ICLR paper: "What bilinear value functions actually learn"

**Core narrative (already in hand):** the field's implicit story — factored critics win
by learning a metric/similarity (VIP/QRL/CRL lineage) — and BVN's informal story —
"disentangling dynamics from goals" — are both wrong in a measurable way. The bilinear
critic wins by **rank-restricted spectral completion**: it estimates the ~low-dim
spectrum of the value family and therefore ranks actions correctly on states it never
fit, which a policy-free selector reproduces at N=8. The advantage is predicted by one
number: the rank of the environment's reachability kernel.

**Novelty claims vs BVN + metric line:**
- N1. Refutation: best factored critic is the *worst* metric (probes + RSA consensus).
- N2. Instrument: best-of-N selector as a causal, actor-free measure of guidance value
  (dose-response; separates ordering from calibration — calibration is anti-correlated!).
- N3. Mechanism: spectral completion, grounded in the *measured* environment kernel;
  d† ≈ dims95 law; capacity-independent expressed spectrum.
- N4. Predictive scaling law: the bilinear-vs-monolithic gap tracks goal-latent
  dimension. (BVN explicitly could not say why tasks differ.)
- N5. Boundary conditions: headroom / goal-observability / kernel degeneracy — with
  causal repairs, not just correlations.
- N6. Head-to-head with BVN's form offline, where our theory predicts the *opposite* of
  their online conclusion.

## 5. Experimental roadmap (scoped to our infra)

Every arm reuses: `impls/` critics + `scripts/vista_train.slurm` matrices,
`eval/metric_probe.py`, `eval/kernel_spectrum.py`, `eval/bon_probe.py` (selector),
`eval/rollout_probe.py` (P2). Cost basis: cube 500k run ≈ known; selector row ≈ <2h dev.

### Arm A — rank scaling across environments (the headline; tests N3/N4)
- Envs: OGBench `cube-single-play` (done), `cube-double-play`, `scene-play`, and one
  near-rank-1 control (e.g. `pointmaze`-style or a 1-target env). All are stock OGBench:
  no new env code, datasets download.
- Per env: kernel spectrum from val data (cheap, CPU) → predicted d†; then
  {biql d in {10, 20, dims95, 2*dims95, 512}, gciql} x flowbc x 3 seeds; selector probe
  biql vs gciql at N in {1,8,64}.
- **Pre-registered predictions:** (i) success saturates at d† ≈ dims95 per env;
  (ii) the biql−gciql gap (trained AND selector) grows with dims95: cube-double >
  cube-single >> 1-target control ≈ 0; (iii) expressed-spectrum capacity-independence
  replicates per env.
- Scale: ~40 training runs + ~20 selector rows. ~2 campaign-days.

### Arm B — head ablations under one objective on cube (tests N6 + composition)
Same GCIQL expectile objective, same data, swap only the head:
1. `bvn`: `f(s,a)^T phi(s,g)` (BVN's form; ~1 new critic file cloning iql_bilinear).
2. `l2`: `V = -||phi(s) - psi(g)||` (metric-constrained head, VIP's form, GCIQL's loss).
3. `iqe`: IQE quasimetric head with GCIQL's loss (reuse impls/utils/quasimetrics.py).
4. `bottleneck-mlp`: monolithic MLP with a rank-d linear bottleneck on (s,a) before
   concatenating g — separates "low rank" from "bilinear in g".
Each: 3 seeds x 500k + selector probe + P7 spectrum.
- **Predictions:** bvn ≈ monolithic on selector coherence (s-conditioned goal map loses
  completion; likely fine on-distribution) — *contradicting their online result*; l2/iqe
  land near biql iff cube's kernel is near negative-type (P4 says asymmetry is small);
  bottleneck-mlp recovers part of the gap iff rank is the active ingredient independent
  of the bilinear form.
- Scale: ~12 runs + probes. 1 campaign-day. **Highest insight per GPU-hour; start here.**

### Arm C — train-side goal-channel repair on MW (closes Ph.II; tests N5)
Mask context dims (obs[36:39]) from psi input + p_randomgoal=0, retrain the pnh square,
selector with synthesized goals.
- **Prediction:** selector curves become fully cube-shaped (the synth arm already flipped
  N=8; the residual poisoning is train-side).
- Scale: 12 runs + 12 selector rows. 1 campaign-day.

### Arm D — offline OOD-goal splits (BVN's protocol, our setting; tests N2/N4)
Filter cube play data: value/actor goals restricted to a half-space of the workspace;
eval on the held-out half. biql vs gciql vs bvn-head x 3 seeds; selector probe on held-out
goals; measure ordering degradation vs goal distance from the training support.
- **Prediction:** biql's selector curve degrades most slowly with OOD-ness (completion);
  bvn-head degrades like the monolithic critic.
- Scale: ~18 runs (needs a small GCDataset goal-filter flag). 1-2 campaign-days.

### Arm E — writing + theory tightening (no GPU)
Finite-sample completion statement for the fixed factorization (matrix-completion
guarantee under incoherence + marginal coverage, contrast with phi(s,g) which has no such
guarantee); relate d† ≈ dims95 to low-rank-MDP representational results; fold
phenomenon.tex remarks into the paper's appendix as the pre-registration log (reviewers
reward this).

**Priority order: B → A → C → D** (B is cheapest and resolves the BVN contradiction;
A is the headline; C closes the one open phenomenon; D is the direct BVN-protocol
comparison reviewers will ask for).

## 5b. Arm A spectra landed BEFORE training (pre-registration order preserved)

Centered-value statistic (the one that matched cube-single: dims95=16, eff-rank ~10),
default estimator (n_anchors=2500, radius_q=0.9):

| env | dims90 | dims95 | dims99 | eff-rank | d* median |
|---|---|---|---|---|---|
| scene-play | 9 | **22** | 398 | 9.3 | 40 |
| cube-single (prior) | 11 | **16** | — | 10.1 | ~40 |
| cube-double | 5 | **9** | 55 | 5.1 | 75 |
| pointmaze-medium | 5 | **5** | 13 | 5.3 | 75 |

Two pre-registration updates, recorded before any Arm A training run finishes:

1. **cube-double under-resolution caveat.** dims95=9 < cube-single's 16 contradicts the
   naive "double the blocks, double the rank" guess — but knn_radius had to stretch to
   2.20 to keep the graph connected at 2500 anchors in the ~46-dim obs space, exactly the
   distance-collapse failure mode the estimator notes warn about. Hi-res re-runs
   (n_anchors=6000, radius_q=0.75; jobs 889249/889250) decide which number is real. If
   dims95 stays ~9, the honest revised prediction is that cube-double's biql-gciql gap is
   *comparable to or smaller than* cube-single's — the rank hypothesis then predicts gap
   from the *measured* spectrum, not from intuitive latent-dimension counting.
2. **pointmaze is not the rank-1 control we assumed.** A maze's shortest-path metric is
   the textbook non-Euclidean distance; measured dims95=5, not 1-2. The revised control
   reading: pointmaze is *low* rank (5) rather than *near-rank-1*, so the predicted
   ordering of the bilinear advantage by dims95 is now
   **scene (22) > cube-single (16) > cube-double (9, pending hi-res) ≥ pointmaze (5)**,
   with pointmaze additionally headroom-limited if BC is strong there.

## 5c. Estimator robustness (2026-08-04, before Arm A training lands)

The single-setting spectra are NOT stable across estimator resolution: cube-double dims95
went 9 (a2500, q0.9) -> 58 (a6000, q0.75); scene went 22 -> 9; antmaze reads 24 at its
first setting (knn radius 16.7 — the ant's proprio dims dominate distances; antmaze is
*not* a low-rank control either, and pointmaze was dropped as anomalous per OGBench).
Consequence: the cross-env prediction must be stated over estimator **brackets**, not
points. An 8-cell robustness grid (envs x {a2500,a6000} x {q0.75,0.9}, incl.\ cube-single
recalibration) is running (`specgrid_*.json`); the pre-registered comparison for Arm A is
gap-vs-**median-bracket** dims95, with envs whose brackets overlap treated as unordered.
Honest current state: cube-double plausibly >= cube-single (the low first reading was
distance collapse), scene and antmaze unresolved until the grid lands.

## 6. Risks

- BVN head might *win* offline too → then the completion story needs refinement (their
  s-conditioning may act as a learned gating, not a rank increase); still a publishable
  finding either way, and the selector probe will say *where* the orderings differ.
- cube-double at 500k may need longer training (harder task); budget a 1M-step pilot.
- OOD-goal filtering interacts with p_randomgoal relabeling — implement the filter at
  the GCDataset level, not collection level, and validate with a goal-histogram audit.
### 5c addendum — grid complete (locked pre-training)

| env | dims95 bracket [min..med..max] | eff-rank (med) |
|---|---|---|
| cube-double | [9 .. **33** .. 62] (n=4) | 9.0 |
| antmaze | [24 .. **24** .. 24] (n=3, stable) | 8.9 |
| scene | [9 .. **15** .. 23] (n=4) | 7.3 |
| cube-single | [8 .. **13** .. 19] (n=2) | 9.2 |

Locked ordering prediction for the Arm A gap (bracket medians; overlapping brackets
unordered): **cube-double > antmaze > {scene ~ cube-single}**. Two sharp claims: (i)
antmaze — a maze, whose shortest-path family is high-rank — should show a *large*
bilinear advantage if headroom is open, even though its goal latent is only 2-D torso
position (this cleanly divorces kernel rank from goal-coordinate dimension); (ii)
scene ~ cube-single predicts comparable gaps despite very different task semantics.
