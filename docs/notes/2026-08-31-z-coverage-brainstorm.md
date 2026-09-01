# Does p(z) cover the rewards we care about? — brainstorm for a z-coverage analysis

**Status: design only. No implementation, no training. All proposed work is eval-only on
existing checkpoints.**

## 0. The question

FB-style methods promise: π(a|s,z) = argmax_a F(s,a,z)ᵀz is optimal for reward r once
z = encode(r). The promise is only redeemed for rewards whose inferred z falls where the
actor was actually trained — i.e., inside the region the training prior p(z) actually
samples. The analysis: **for each class of "reasonable" reward functions (k-hot, bounded
[-1,1], dynamics-smooth), sample many rewards from the offline dataset, run each through
the method's own z-inference, and compare where the resulting z's live against the z's
the actor was trained on** — a known 50/50 mix of hindsight goal-derived z's and uniform
√d-sphere samples.

Two prior measured facts make this urgent rather than academic:

- **Flat-landscape finding (spec-29/30 line):** a zero-order search over the √50-sphere
  found the objective flat everywhere except an isolated spike at the encoder's ẑ.
  Uniform-sphere z's map to generic/BC-like behavior. That is exactly what "p(z) wastes
  almost all its mass off the reward-relevant set" would look like. This analysis
  quantifies it as a statement about whole reward classes instead of a single anecdote.
- **Gauge finding (spec-28/30):** density-FB's softmax normalization leaves a per-(s,a)
  additive gauge; the uncentered second moment Σ_B is rank-1 dominated by the gauge mean
  (‖μ_B‖ 33–103× the centered signal), and the naive z(g) = norm(B̄(g)) collapsed to
  pairwise cos .998–1.000 until centering/ridge fixed it. Any coverage analysis that
  ignores encoder gauge means will "discover" a cone that is actually an artifact.

And one structural fact: **p(z) is explicitly known.** Actors train on a ~50/50 mix of
hindsight goal-derived z's (project(B(g)) over dataset goals) and uniform √d-sphere
samples (`fb.py::sample_cond`, `rldp.py` analogue). So the coverage question is not
"what is p(z)" but "where do the inferred z's sit relative to a *known mixture*."

## 1. The analysis, plainly

Fix an environment with offline dataset D and inference set ρ (the code pins this:
`INFER_COND_MARGINAL = next_observations`; N = 50k for RLDP, 10k for HILP, per
`scripts/exorl_paper_eval.py`). A reward class C is a recipe for sampling reward
functions r: S → ℝ, which for our purposes means sampling vectors r ∈ ℝᴺ (the reward
evaluated on the N inference states). Each method m has a z-inference procedure

    E_m : r ↦ z ∈ S^{d−1}(√d)      (after its projection convention),

and the object of study is simply **Z_C = { E_m(r) : r sampled from class C }** — a
large sample of inferred z's per class — compared against a matched sample Z_p from the
training prior:

    p(z) = ½ · uniform on S^{d−1}(√d) + ½ · project(B(g)), g from the dataset goals.

All comparisons are **angular** (cos similarity); radial information is recorded
separately *before* projection (§5, pitfall 3) because `project_z` destroys it.

### What "coverage" means operationally — three candidate measures (use all three)

**(M1) Cap-concentration statistics.** For each class: the distribution of
cos(z, z̄_C) to the class mean direction; the mean resultant length R̄_C = ‖mean(Z_C)‖/√d;
the angular radius θ₉₀ containing 90% of the class z's. Compare against the
uniform-sphere null, where cos²(u, v) ~ Beta(½, (d−1)/2), i.e. |cos| concentrates at
O(1/√d) ≈ 0.14 for d = 50, 0.04 for d = 512. Coverage FAILS in the interesting
direction if θ₉₀(Z_C) is tiny (a cone) while the fraction of prior samples inside that
cone, P_{z~p}(cos(z, z̄_C) > cos θ₉₀), is astronomically small — quote it as
"the uniform half of p(z) puts ~10⁻ˣ of its samples where class C lives."

**(M2) Effective rank / participation ratio.** M_C = mean over Z_C of z zᵀ (uncentered,
on the sphere), PR(M_C) = (tr M_C)² / tr(M_C²) ∈ [1, d]. The uniform prior half has
PR = d. Report PR/d per class per method (normalizes across the differing z_dim
conventions). A companion: **energy-in-B-subspace** — the average fraction of ‖z‖²/d
captured by the top-m PCA subspace of the dataset B-rows {B(s'_i)}. This is the direct
"waste" statistic: if the class z's live 95% inside a rank-20 subspace of a d = 512
sphere, the uniform half of p(z) is spending ~all of its samples orthogonally to every
reachable reward.

**(M3) Sample-based support overlap (sphere precision/recall).** With the samples Z_C
and Z_p in hand: *recall* = fraction of Z_C within angular ε of some prior sample (does
p(z) reach the rewards?); *precision* = fraction of Z_p within ε of some class z, for ε
chosen from the within-class kNN scale (does p(z) waste samples?). This is the standard
kNN precision/recall two-sample construction, run with geodesic distance on the sphere.
Optionally a sliced-W2-on-the-sphere number as a single scalar summary, but
precision/recall is more diagnostic (it separates "misses" from "wastes").

(Considered and deprioritized: angular energy spectra via spherical harmonics —
correct in principle but numerically hopeless in d = 50–512 and it answers nothing
M1–M3 don't.)

## 2. Reward-class generators (samplable from the offline dataset)

All classes are defined on the inference set ρ so every encoder sees the reward
exactly the way `infer_cond` would. Every plot also carries the **actual task rewards**
as anchor points — walker stand/run/walk/spin (note: `spin`, not `flip`; see
`exorl_paper_eval.py` header), cheetah tasks, pointmass reach — computed by the
existing re-simulation path (`env/exorl_env.py::relabel_rewards`). Anchors are the
ground truth for "reasonable" and calibrate everything else.

### C1: k-hot
r_i = 1 for i in a hot set S_k of size k, else 0. Sweep k ∈ {1, 4, 16, 64, 256, 1024}.
Two sub-classes, and the distinction is load-bearing (see §5 pitfall 6 and the null in
§5):
- **k-hot-scattered:** S_k drawn uniformly from the dataset indices.
- **k-hot-clustered:** S_k = the k nearest dataset neighbors of a random seed state
  (ε-ball in observation space, or in a fixed random projection of it). This is the
  version that actually resembles "reward for reaching a region."
State-indexed rewards are formally functions of the dataset index, not of s; to keep
them function-like, optionally define hotness by ε-ball membership so duplicates in ρ
are treated consistently. (For d(z)=50 FB on cube, 1-hot-clustered ≈ the goal-reaching
reward, tying C1 to the hindsight half of p(z) — see H4 below.)

### C2: bounded [-1, 1] (three constructions; use ≥2)
- **C2a random-features linear:** r(s) = tanh(wᵀψ(s)/τ) with ψ = random Fourier
  features of the raw observation at a few bandwidths, w Gaussian. Smoothness dial via
  the RFF bandwidth; tanh keeps it in [-1,1] without hard clipping artifacts.
- **C2b signed kernel mixtures:** r(s) = clip(Σ_{j≤J} ε_j K_σ(s, c_j), −1, 1),
  ε_j = ±1, centers c_j drawn from ρ, J ∈ {2, 8, 32}. Rewards with sign structure —
  important because E[rB]-type encoders respond linearly to sign cancellation.
- **C2c white noise:** r_i ~ U[−1, 1] iid across dataset states. The maximally
  dynamics-*un*-smooth member of the bounded class — the far anchor. Under an E[rB]
  encoder its z is pure Σ_B-colored noise (raw norm ~ ‖B‖·σ_r/√N → direction ≈
  top-eigenvector-biased random). It doubles as the permutation noise floor (§5).

### C3: dynamics-smooth (implementable proxies for bounded Lipschitz w.r.t. a dynamics metric)
- **C3a trajectory heat-kernel smoothing:** take any base reward r₀ (e.g. a k-hot),
  define r_W(s_t) = Σ_{|Δ|≤W} λ^{|Δ|} r₀(s_{t+Δ}) along the dataset trajectories,
  renormalized. Exact, cheap, model-free; W ∈ {1, 5, 25} sweeps the smoothness. This is
  "k-step averaged k-hot" and gives a matched pair (r₀, r_W) with identical support
  seeds — the cleanest smoothness contrast.
- **C3b discounted-occupancy smoothing:** r'(s) = (1−γ')·Ê[Σ γ'ᵗ r₀(s_t) | s₀ = s]
  estimated by n-step sums along dataset trajectories (NOT via the model's own
  successor operator — that is circular, §5 pitfall 7; the empirical trajectory version
  is the honest proxy).
- **C3c dynamics-bandwidth RFF:** C2a with the bandwidth set to c × the median one-step
  transition distance median‖ψ(s_{t+1}) − ψ(s_t)‖, c ∈ {1, 4, 16}. "Smooth w.r.t.
  dynamics" operationalized as "varies slowly over one environment step."

## 3. Per-method z-inference and its gauge/centering traps

| method | encoder (as implemented) | trap |
|---|---|---|
| FB | z ∝ rᵀB over ρ, then `project_z` √d (`impls/critics/fb.py::infer_cond`) | z_raw = Cov_ρ(r, B) + r̄·μ_B. Any reward mean offset injects the B-mean direction. Run every class twice: raw r and ρ-mean-centered r; the difference isolates the μ_B leak. |
| RLDP | same moment, ÷N, over 50k next_obs (`impls/critics/rldp.py::infer_cond`) | same μ_B trap; additionally its B is √d-normalized per-row, so B-row norms carry no weighting — hot states contribute equal-norm vectors (changes the k-hot null slightly, see §5). |
| HILP | z = lstsq(φ(s'), r), then √d-normalize (`impls/critics/hilp.py::infer_cond`, 10k rows) | a SOLVE, not a moment: whitens by the φ covariance, so it amplifies along small-singular directions of φ. Sensitivity check: ridge with λ ∈ {0, small, tr/d} — if the class z's move a lot under tiny λ, the lstsq geometry is ill-conditioned and the spread of inferred z's is partly a conditioning artifact. Offset-invariant only if span(φ) contains 1. |
| density-FB | centered dual ridge: m_c = Ĉov(B − μ̂, r − r̄), c = (Ĉ + λI)⁻¹ m_c, re-project √d (`impls/critics/density_fb.py`, spec-30 port; OGBench ckpt is the trained artifact) | this is the one encoder *built* to be exactly invariant to r → αr + β and to the softmax gauge. Use it as the reference: for the other three, report cos(ẑ, μ̂_B/‖μ̂_B‖) per class. If a method's classes all pile onto ±μ_B, that cone is gauge, not reward information (this is precisely how spec-28's cos-.998 collapse would masquerade as "extreme concentration"). |

Prior construction per method must match its own convention: uniform on *its* sphere
(√d for norm_z methods), hindsight half via *its* B/φ and *its* projection. Never
compare raw cos across methods with different d — compare against each method's own
uniform null (M1) and report PR/d (M2).

## 4. Visualization set

Each figure states what it shows under coverage-HOLDS vs coverage-FAILS.

- **Fig A — 2-D PCA scatter.** PCA fit on the pooled class z's (all classes, one
  method), classes colored, prior samples (both halves separately marked) underlaid,
  task-reward anchors as stars. HOLDS: prior points interleave every class cloud.
  FAILS: class clouds occupy a compact region; the uniform-prior half is an
  undifferentiated ring around the origin of the plot (which is what an isotropic
  high-d cloud looks like after projecting onto someone else's top-2 plane). Explicit
  caption warning: this figure is illustrative only; the quantitative claims live in
  B–G (§5 pitfall 2).
- **Fig B — pairwise-cos CDFs.** Three CDFs per class: within-class cos, class-to-prior
  cos (split by prior half), and the analytic uniform null Beta(½,(d−1)/2). HOLDS: all
  three hug the null. FAILS (expected): within-class CDF shifted far right (cone);
  class-to-uniform-half hugging the null (orthogonal waste); class-to-hindsight-half
  intermediate.
- **Fig C — pre-projection raw-norm histograms + noise floor.** ‖z_raw‖ per class per
  method BEFORE `project_z`, with the permutation floor (shuffle r against B rows,
  §5 pitfall 3) as a shaded band. Not a coverage figure — a validity gate: any class
  whose raw norms sit inside the floor has *noise directions*, and its apparent angular
  spread in every other figure must be discounted.
- **Fig D — PR/d bars.** PR(M_C)/d per class per method; reference lines at 1
  (uniform), PR of the hindsight prior half, and PR of the raw B-row cloud. HOLDS:
  class bars near the uniform line. FAILS: class bars at the B-row line or below.
- **Fig E — overlap matrix + precision/recall.** Class × class mean-direction cos
  matrix (including both prior halves and the task anchors as rows), annotated with
  the M3 precision/recall numbers. HOLDS: near-diagonal only. FAILS: a solid block —
  all reward classes mutually aligned — plus recall ≈ 0 for the uniform half.
- **Fig F — the k sweep.** θ₉₀ and mean cos-to-μ_B (and to the top Σ_B eigvector) of
  the k-hot z's vs k, scattered and clustered variants, with the CLT null curve (§5)
  overlaid. HOLDS: flat, wide, null-matching. FAILS: 1/√k collapse onto μ_B — and if
  scattered matches the null exactly while clustered deviates, that deviation is the
  *only* part of k-hot that reflects reward semantics rather than B-row statistics.
- **Fig G — energy-in-B-subspace.** Fraction of class energy in the top-m B-row PCA
  subspace vs m, per class, with the two prior halves as reference curves. The single
  most decision-relevant figure: it says which *fraction of the prior* is spent where
  rewards can possibly land, i.e., it prices the 50/50 mix.
- **(Fig H, OGBench follow-on)** Overlay the flat-landscape anecdote: for the cube
  density-FB checkpoint, histogram cos(z_uniform, ẑ_task) for the uniform prior half
  against the within-class cos scale — connects the coverage statistics to the
  observed behavioral spike directly.

## 5. Controls and pitfalls (what makes this analysis lie)

1. **Gauge-mean domination.** The measured fact: Σ_B is rank-1 dominated by the
   softmax-gauge mean at 33–103×. For moment encoders (FB/RLDP), any class with
   nonzero r̄ acquires a μ_B component that has nothing to do with the reward. Control:
   every class run raw and r-centered; report cos-to-μ_B everywhere; treat density-FB's
   centered ridge as the gauge-clean reference.
2. **2-D projection artifacts.** PCA of points on a high-d sphere fabricates ring/shell
   shapes and hides orthogonality; also cos-structure is signed (z and −z are different
   tasks — moment encoders are sign-definite in r, so keep signed cos, do not fold
   antipodes). Fig A is decoration; B/D/F/G carry the claims.
3. **Projection hides degeneracy (the raw-norm trap).** `project_z` re-imposes √d, so a
   z_raw of norm 10⁻⁶ and one of norm 10² look identical downstream. A tiny raw norm
   means the direction is estimator noise — Σ_B-colored, hence *biased toward the same
   top eigvectors as real signal*, the nastiest possible confound: pure noise can
   reproduce the "cone." Gate: permutation floor from shuffled (r, B-row) pairings
   (analytically ≈ ‖B‖_F σ_r/N per component budget; empirically 100 shuffles). C2c
   white-noise sits at this floor by construction and calibrates it.
4. **Estimator variance vs class variance.** Every encoder integrates over a finite
   inference sample (50k/10k). Decompose: for each fixed reward, re-infer z over B
   bootstrap resamples of the inference set; if the per-reward angular jitter is
   comparable to the across-reward spread, the observed spread of inferred z's is
   mostly Monte Carlo noise and the class geometry is not being measured at all.
   Report the ratio (within-reward θ₉₀)/(across-reward θ₉₀) per class; require ≪ 1
   before quoting any concentration claim.
5. **Norm-convention apples-to-oranges.** √d-sphere vs unit-sphere vs d = 50 vs 512:
   only angular quantities compared, each against its own null; PR always reported /d.
6. **The k-hot matched-null degeneracy (and the CLT prediction).** For an E[rB]
   encoder, scattered k-hot gives literally
   z_raw = (1/N) Σ_{i∈S_k} B(s'_i) — a sum of k rows drawn from the empirical B-row
   distribution. If rows are ~iid (mean μ_B, cov Σ_B):
   z_raw ≈ (k/N) μ_B + (√k/N) Σ_B^{1/2} g, g ~ N(0, I). Consequences: (i) the class
   direction interpolates from "single B row" at k = 1 to "deterministically μ_B" as
   k → ∞, with angular fluctuation ∝ (√λ_top(Σ_B)/‖μ_B‖)/√k around it — an explicit
   1/√k collapse curve to overlay on Fig F; (ii) **scattered k-hot IS its own null**:
   the scattered k-hot z's are determined by B-row statistics alone and contain zero
   bits about dynamics or reward semantics. Therefore the matched null to plot against
   *every* class is "sum of k random B rows, k matched to the class's effective support
   size k_eff = (Σr_i)²/Σr_i²"; only the *deviation* from that null (clustered k-hot,
   smooth classes, task anchors) is evidence about reward structure. An analysis that
   skips this null will proudly report "k-hot rewards concentrate!" when it has
   measured nothing but Σ_B's spectrum — i.e., re-measured finding 1 in disguise.
   (RLDP's per-row √d-normalized B changes the constants, not the structure; HILP's
   lstsq breaks the linearity so its k-hot null must be simulated, not derived.)
7. **Circularity of model-based smoothing.** Smoothing rewards through the model's own
   successor operator would test whether the model covers rewards it generated itself.
   Use trajectory-empirical smoothing (C3a/b) only.
8. **Dataset dependence.** The inferred z's are defined *relative to ρ*. That is not a
   bug (deployment infers z from the same ρ) but it must be said: conclusions are about
   coverage *under this dataset*, and a robustness spot-check with a re-drawn 50k
   subsample (distinct from the bootstrap in 4) belongs in the appendix.

## 6. Prioritized plan

**Where:** ExoRL seam checkpoints first (~/git/Factored-FB-seam, branch density-fb) —
walker (all four critics), then cheetah; state-based, everything is forward passes of
B/φ over ≤50k rows plus per-reward matvecs/solves. Cheapest decisive artifact. The
OGBench cube density-FB centered-ridge checkpoint second (Fig H tie-in to the
flat-landscape anecdote). Harness: reuse the checkpoint-loading + relabel path of
`scripts/exorl_paper_eval.py`; a new eval-only script that never touches training code.

**Order (first three figures):**
1. **Fig B + Fig C together** (cos CDFs, gated by raw-norm floors), walker, 4 methods,
   classes {1-hot-c, 64-hot-s, C2a, C2c, C3a(W=5), task anchors}. One afternoon of
   compute (minutes per method-class on GPU; ≤2 h total CPU). This alone answers
   cone-vs-covered and validates that the cone is signal.
2. **Fig F** (k sweep + derived CLT null, scattered vs clustered). The cleanest
   quantitative curve and the one with an analytic prediction to confirm or refute.
3. **Fig G** (energy-in-B-subspace with the two prior halves). Prices the 50/50 mix
   and converts the flat-landscape anecdote into a mass fraction.
Then D, E, A, H.

**Compute:** all eval-only. Per env × method: one B/φ pass over 50k states (seconds–
minutes), ~10³ reward samples × (matvec or 10k-row lstsq/ridge) — minutes. Bootstrap
×20 for control 4. Everything above: hours, single GPU or even CPU.

**Hypotheses this can falsify:**
- **H1 (cone):** the inferred z's for all reasonable classes lie in a cone of
  angular radius ≪ 90° whose uniform-prior mass is < 10⁻³ — i.e., the uniform half of
  p(z) is ~pure waste. Falsified if class-to-uniform cos CDFs match the Beta null.
- **H2 (CLT collapse):** the k-hot z's contract onto the μ_B/top-Σ_B direction at rate
  1/√k, and clustered k-hot deviates from the scattered null (carrying the only
  semantic content). Falsified if scattered ≠ null (would mean B rows are far from
  exchangeable — itself interesting) or if no 1/√k trend.
- **H3 (smoothness concentrates):** dynamics-smooth classes concentrate MORE than
  their unsmoothed seeds (smoothing = averaging more B rows ⇒ CLT again), with the
  task anchors sitting inside the smooth-class cone, not the k-hot one. Falsified if
  smoothing widens the class spread or anchors land off-cone.
- **H4 (hindsight half = 1-hot class):** for moment encoders, project(B(g)) is exactly
  the 1-hot encoder output, so the hindsight prior half should be two-sample
  indistinguishable from the 1-hot-clustered z's — meaning p(z)'s useful half covers
  only the Dirac corner of reward space, and every multi-support/smooth/bounded reward
  lands in the *interior* cone (sphere-projected convex combinations of B rows, pulled
  toward μ_B) that neither prior half samples. Falsified by the M3 two-sample test.

**Deliverable:** figures + a one-page summary table (θ₉₀, PR/d, prior-mass-in-cone,
precision/recall, noise-floor pass/fail) per method × env, dropped into
`docs/notes/` alongside this file.
