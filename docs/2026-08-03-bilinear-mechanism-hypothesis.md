# The leading hypothesis for bilinear GCIQL, and its validation on OGBench cube

*Status: living document, 2026-08-03. Companion to `PAPER/decoding-bilinear/phenomenon.tex`
(the pre-registered version with full math) and `decoding-bilinear.tex` (kernel formalism).
All numbers below are from our own runs on `cube-single-play-v0` (the metric-probe matrix in
`$SCRATCH/factored-fb-metric-probe/`, 3 seeds × 500k unless noted).*

## 1. The claim

**Bilinear GCIQL's advantage is not a better learned metric. It is better *actor guidance*:
the bilinear head restricts the critic's hypothesis class to bounded, goal-uniform,
low-rank functions, and that restriction (i) caps how much phantom value the actor can
manufacture by ascending critic errors, and (ii) forces matrix-completion-style
generalization to eval-time (state, goal) pairs never co-observed in the play data —
which is exactly where a monolithic MLP critic breaks.**

The mechanism is a **conjunction**: the guidance *target* must be in-sample optimal
(expectile TD), and the *parametrization* must be bounded-low-rank (bilinear). Either
ingredient alone plateaus far below both together:

|                                       | monolithic head | bilinear head |
|---------------------------------------|-----------------|---------------|
| behaviour-policy target (NCE)         | —               | CRL: 28–34%   |
| in-sample-optimal target (τ = 0.9)    | GCIQL: 73%      | **biGCIQL: 93%** |

(Pure-BC floor: 8–10%. So every point above the floor is critic guidance.)

Formally, with `Q(s,a,g) = ⟨φ(s,a), ψ(g)⟩/√d` and LayerNorm-concentrated tower norms:

- **Boundedness**: |Q| ≤ ‖φ‖‖ψ‖/√d =: κ, uniformly over (s,a,g). Measured on biql-d50:
  κ ≈ 75, realized V range [−63, +6] — while the true V* range under γ=0.99 is [−100, 0].
  The bilinear critic *cannot even express* the far-from-goal values; actor ascent can
  collect at most 2κ of phantom value, versus an unbounded amount from a monolithic MLP.
- **Goal-uniform gradient scale**: ‖∇ₐQ‖ ≤ ‖J_φ‖·‖ψ(g)‖/√d with ‖ψ‖ concentrated
  (CV ≤ 0.09), so no goal region dominates the actor's batch gradient.
- **Rank restriction / completion**: Q has rank ≤ d. A low-rank factorization is pinned
  down by marginal coverage (each state seen with *some* goals, each goal with *some*
  states) and therefore fills in the unseen (s, g) blocks of the reachability kernel —
  the pairs eval actually asks about.

## 2. What the hypothesis had to survive (validation on OGBench cube)

### 2.1 The headline ordering it must explain (E1)

biql-d50 + FlowBC = **92.7 ± 1.4** (fresh reproduction of the historical 90.7);
biql-d512 = 89.2; gciql = 73.2 ± 1.3; crl = 28–34; pure-BC floor 8–10.

### 2.2 The "better metric" story is dead (probe battery, established negative)

If biql won by learning a more faithful temporal geometry, it should win the geometry
probes. It loses them:

- **P1 temporal-distance rank correlation** (val-set pairs, partialed on raw-obs
  distance): gciql 0.63/0.54 > biql 0.54/0.41.
- **RSA across methods**: gciql↔qrl 0.88, gciql↔crl 0.54 — a three-method consensus
  geometry; biql agrees with nobody (≤ 0.21) and is closest to *raw observation L2*
  (0.45–0.52). The outlier wins by 20 points.
- **Geometry saturates by 100k steps while success climbs 65 → 93** — value-geometry
  fidelity on the data distribution is neither necessary nor sufficient for guidance value.
- Caveat we log honestly: the cosine-ness of the head (dot≈cos, corr 0.996) is already
  present in the untrained net — norm concentration is architectural (LayerNorm), so the
  *learned* asset is the direction field, not cosine-ness per se.

Conclusion: whatever biql buys, it buys at the **actor interface**, invisible to
(s,g)-geometry probes on the data distribution. That is what forced the guidance-centric
hypothesis in §1.

### 2.3 Direct positive test: the P2 eval-rollout probe (2026-08-03, decisive)

Pre-registered prediction (phenomenon.tex Pred. mech.3): gciql's value-ranking quality
degrades from val-pairs to *eval-task* pairs strictly more than biql's — an interaction,
not a main effect. Full 2×2 (scoring critic × rollout-generating policy), 3 seeds × 50
temperature-0 eval episodes; entries are per-episode Spearman ρ(V̂(s_t, g), t), median over
episodes, mean ± std over seeds:

| scored by ↓ on rollouts from → | biql policy (0.95 succ.) | gciql policy (0.73 succ.) |
|---|---|---|
| biql critic  | **0.983 ± 0.003** | 0.871 ± 0.060 |
| gciql critic | **0.355 ± 0.239** | 0.883 ± 0.015 |

Three readings, all in the hypothesis's favour:

1. **The interaction is confirmed.** gciql wins on val-pairs (P1) and collapses on
   eval-task rollouts (0.355, per-task medians down to −0.80: its value *falls* as the
   policy approaches the goal).
2. **The sharper form is a transfer asymmetry.** Each critic is coherent along its own
   policy's trajectories (diagonal 0.98 / 0.88). Only the bilinear critic transfers to the
   *other* policy's state distribution (0.87 vs 0.36). The monolithic critic's ranking is
   intact exactly up to the states its own worse policy reaches, and breaks on the states
   the better policy actually visits.
3. **It closes the causal loop on 73 → 93**: ascent on the bilinear Q can push the policy
   *beyond* the fidelity frontier of the monolithic critic because the bilinear value
   stays coherent there — which is the completion property (§1, iii) doing the work.

### 2.4 The rank axis (d-sweep, 2026-08-02/03)

Success vs latent dim (3 seeds × 500k): d=5: 0.06 ± 0.05 · d=10: 0.62 ± 0.07 ·
**d=20: 0.92 ± 0.02** · d=30: 0.90 ± 0.02 · d=50: 0.93 ± 0.02 (d=512: 89.2).

Learned effective rank at the same checkpoints: 4.1 / 8.0 / 14.7 / 19.6 / 28.0
(d = 5/10/20/30/50; 48 at 512).

- The pre-registered *strong* reading ("success transitions sharply at the unconstrained
  solution's eff-rank ≈ 28") is **falsified** — saturation is already at d = 20.
- The surviving, sharpened reading: the guidance signal is a genuinely low-rank object of
  intrinsic dimension ≈ 10–20; eff-rank tracks the container sublinearly (compression
  0.82 → 0.56 → 0.09 as d grows) and is *not* an invariant — the invariant is the success
  saturation point d† ∈ [10, 20]. Rank is a capacity budget, not a learned constant.
  (d = 15 runs in flight bracket d† tighter.)
- d = 5 collapses to 6% — *some* rank is necessary; boundedness alone with rank 5 is not
  sufficient. This is independent pressure toward the conjunction claim.

### 2.5 What is still pending (launched, not yet landed)

- **gciql-qb75** (κ·tanh(·/κ), κ = 75, on cube): if bounding the monolithic critic
  recovers a large fraction of 73 → 93, boundedness (i) shares the credit; if none, the
  completion property (iii) carries the phenomenon alone. Either way the hypothesis
  refines rather than dies — that is what the P2 result already secured.
- **biql-d50-nopsiln** (ψ tower without LayerNorm): tests whether norm concentration (ii)
  is load-bearing; prediction: success drops toward gciql at identical rank.
- Cross-env arms (MW play-noise-hard, MW play-resample) test the *other* half of the
  project — why the mechanism is invisible on MW/MS (headroom censoring, broken goal
  channel, kernel degeneracy); see phenomenon.tex Ph. I–III.

## 3. Grounding: the intrinsic RL property being captured (2026-08-03)

"Low-rank completion works" is a description until we say what property of the *MDP* makes
the value function low-rank. The claim: **the family of optimal goal-conditioned value
functions {V\*(·, g) : g ∈ G} spans a low-dimensional function space, because the hitting
time d\*(s, g) depends on (s, g) only through the low-dimensional controllable latent (the
block pose), not the 28-dim observation.** This is the structure assumed by linear /
low-rank MDP theory (transition kernel of rank d ⇒ every value function linear in d
features), observed empirically by UVFA's goal-value matrix factorizations, and exploited
by successor-measure / forward-backward methods; the bilinear critic is a spectral method
for the optimal-value kernel.

Measured, with no learning involved (`eval/kernel_spectrum.py`: shortest paths on a
stride-5 graph over the validation data — temporal edges carry demonstrated step gaps, kNN
shortcuts are radius-limited to the 5-step displacement quantile — then SVD of the
centered value matrix V = (γ^D − 1)/(1 − γ)):

| estimator setting | 90% energy | 95% energy | eff. rank |
|---|---|---|---|
| stride 5, radius q0.9 | 7 dims | 19 dims | 11.1 |
| stride 5, radius q0.5 | 5 dims | 7 dims | 5.5 |
| stride 3, radius q0.9 | 5 dims | 6 dims | 5.0 |

Robust conclusion: the environment's own reachability kernel has ~5–7 dominant components
(the top 3 carry 44/21–30/12–19% — the three spatial coordinates of the block latent, plus
harmonics), effective dimension ≈ 5–11.

**The learned critic converges to the same object, independent of capacity.** SVD of the
trained value matrix Φ Ψᵀ/√d on the probe states (centered):

| | 90% energy | 95% energy | eff. rank | top components |
|---|---|---|---|---|
| biql-d50 | 11 | 16 | 10.1 | 0.31 / 0.23 / 0.11 |
| biql-d512 | 12 | 18 | 10.6 | 0.29 / 0.23 / 0.12 |

Identical spectra at 10× different container sizes. So the d-sweep's "no invariant" verdict
was about the wrong statistic: the *embedding spread* grows with d (28 → 48), but the
*value function the critic expresses* is a fixed ≈ 10-dimensional spectral object matching
the environment kernel — and success saturates exactly when the container covers it
(d = 10: 0.62, d = 15: 0.84, d = 20: 0.92 ≈ the kernel's 95%-energy dimension).

Cross-environment prediction this makes: the saturation dimension should track the
dimension of the goal-relevant latent, not the observation size — e.g. cube-double should
need roughly twice the rank; a 3-D-target Metaworld task should have a very low-rank
kernel, which is precisely the regime where a monolithic MLP suffices and the bilinear
advantage should shrink. This ties RQ1 and RQ2 together: the bilinear form pays off
exactly when the task family's value spectrum is rich enough that unstructured critics
fail to extrapolate it.

Honest caveats: the graph estimator's absolute numbers move with its knobs (5–19 for the
95% dimension); the critic factorizes Q(s, a, g), whose action conditioning adds rank over
V; and matching spectra is consistency, not identifiability — the decisive causal test
remains the d-sweep saturation, which the kernel spectrum now *predicts* rather than
merely describes.

## 4. Honest limits

- The exploitation bound (§1, i) is an upper-bound argument, not an identity; its
  kill-test is gciql-qb75, not the algebra.
- The P2 probe scores the value tower V(s, g); the actor ascends Q(s, a, g). The two share
  ψ and the TD objective, but an action-conditional version of P2 is a possible follow-up.
- VIP rows in the probe matrix are uninformative (embedding collapsed to eff-rank 1.0) and
  must not be cited as a reference point until its objective is fixed.
- All of this is one environment (cube-single) + one dataset (play). The cross-env arms
  exist precisely because "works on cube" ≠ "property of the algorithm".
