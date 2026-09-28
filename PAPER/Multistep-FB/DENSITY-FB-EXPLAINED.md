# Density-FB, end to end: the problem, the math, the code, and every number

**Status:** written 2026-08-14; updated 2026-08-15 with spec 29's screen,
2026-08-16 with the completed hard cube-double wave, and 2026-08-16 again with
spec 30 (the external method review, verified three ways). Covers
specs 18 (the method), 21/22 (variants and actor ablation), 19 (zero-shot
readout), 24/25/27-Q2 (the z-indexed interface), 26 (the mechanism synthetic),
28 (the z-interface diagnosis — §6.6, which overturns two earlier conclusions
recorded in §6.4), 29 (the R-A repair and its screen — §6.7), and 30 (three
review-driven corrections: the hard estimator is TD InfoNCE's categorical
update — §3.6; the "rank-1 collapse" is a softmax-gauge mean, and the centered
geometry is rich — §6.6.1; the R-A encoder is reward-shift-unstable and the
centered form is the identified fix — §6.6.1, §6.7). Spec 30 Rev 1
(2026-08-16, post-assessment) added four corrections recorded in place: TD
InfoNCE is offline-capable (§3.6), the Finer-BFM rebuttal is retracted and
OpTI-BFM added as prior art (Part 8), the centered encoder has no one-sample
Dirac shortcut so point and reward-sample prompting are separate APIs (§4.4),
and the soft variant is a kernel-smoothed occupancy that evaluates smoothed
rewards (§3.6).

**Who this is for:** someone who has never seen this codebase and wants to
understand what Density-FB is, why it exists, what the code actually computes,
what it scores, and exactly where it fails. Every number in here is either
recomputed from an artifact in `results/` or read from a run's `eval.csv`; where
a number is quoted from someone else's paper it is labelled as such.

---

# Part 0 — The one-paragraph version

A **successor measure** predicts *where a policy will spend its future*, which
lets one trained model evaluate any reward you invent later. **Forward-Backward
(FB)** represents it as an unconstrained bilinear score `F(s,a,z)ᵀB(x)`. On
robot manipulation data that representation breaks, because the object sits
still for long stretches and the objective piles repeated positive updates onto
whatever configuration the object is resting in. **Density-FB** fixes the
representation rather than the data: it predicts the *normalized* measure — a
probability distribution over future object configurations — via a softmax over
a resampled query set, trained with cross-entropy against a TD target. On
OGBench manipulation this takes cube-double from **0.005 to 0.736** and beats
published GC-IQL at half the training budget. It also transfers zero-shot to
rewards it never trained on. FB's *latent* task interface initially failed
outright — swapping goals for a latent `z` collapsed the method to the
behavior-cloning floor — and the cause turned out to be geometric: `B`'s
covariance is rank-1 dominated, so the index collapsed to a single ray and FB's
primal reward embedding pointed nearly orthogonal to the dual coefficient the
readout actually needs (§6.6). Correcting that, and letting the actor integrate
real reward samples, restores the latent interface to **0.812** on cube-single
against a 0.111 floor (§6.7, one seed, reward-transfer eval still pending).

---

# Part 1 — Background: what a successor measure is and why anyone wants one

## 1.1 The object

For a policy `π`, discount `γ`, and a measurable set `X` of future states:

```
M^π(s,a,X) = E_π [ Σ_{t≥0} γ^t · 1[ s_{t+1} ∈ X ] | s_0 = s, a_0 = a ]
```

This is the **discounted future-state occupancy**: how much discounted time the
policy spends in `X`, starting from `(s,a)`.

The reason it is interesting is a single line of algebra. For *any* reward `r`:

```
Q_r^π(s,a) = ∫ r(x) dM^π(s,a,x)
```

Train `M` once, and you can evaluate **any** reward afterwards by integrating it
against the measure. No retraining, no new critic. This is the "zero-shot RL"
promise (Dayan 1993; Blier et al. 2021).

## 1.2 How FB represents it

FB factorizes the *density* of `M` against the dataset marginal `ρ` into two
towers:

```
m_z(s,a,x) = F(s,a,z)ᵀ B(x)
```

`F` sees the current state, action, and a task index `z`. `B` embeds a candidate
future state. A reward is embedded into the index space linearly,

```
z_r ∝ E_{x∼ρ}[ r(x) B(x) ]         and then      Q_r(s,a) ≈ F(s,a,z_r)ᵀ z_r
```

and the estimator is a TD-style loss with an orthonormality regularizer:

```
L_FB = ½ E_{x∼ρ}[ ( m_z(s,a,x) − γ m̄_z(s⁺,a⁺,x) )² ]        (a)
       − E[ m_z(s,a,s⁺) − γ m̄_z(s⁺,a⁺,s⁺) ]                  (b)
       + λ_ortho · L_ortho
```

Term **(a)** is a squared Bellman residual on *random* future-state queries.
Term **(b)** supplies the observed one-step successor — every transition gives a
positive update at the state it actually landed in.

---

# Part 2 — Why manipulation play breaks this

## 2.1 The data

OGBench "play" datasets are long unscripted robot trajectories. The arm is
always moving; the *object* is not. Measured directly from the datasets:

| env | stationarity of the goal-relevant object |
|---|---|
| `cube-single-play-v0` | cube stationary on **48%** of transitions; stationary runs have **median length 44 steps** |
| `cube-double-play-v0` | each cube stationary on **72%** of transitions |
| `scene-play-v0` | drawer and window stationary on **91%**; buttons on **99.6%** |

## 2.2 The mechanism

Take a stationary run of length `L` — the arm is doing something, the cube is
resting. The same object configuration is the observed successor `s⁺` on all `L`
transitions, so term **(b)** fires `L` times at that configuration. The single
transition that *moved* the cube there fires **once**.

> At cube-single's median run length of 44 steps, the resting configuration
> influences the observed-successor term roughly **forty times** more often than
> the transition that produced it.

That asymmetry is not itself wrong — the measure genuinely *should* put mass on
states the policy dwells in. The problem is how FB can express that mass.

## 2.3 The representational gap

`m_z(s,a,x) = F(s,a,z)ᵀB(x)` is an **unconstrained scalar**. But the true
measure has a **fixed total mass**, independent of `(s,a,z)`:

```
M^π(s,a, 𝒳) = Σ_{t≥0} γ^t = 1/(1−γ)
```

Nothing in the bilinear form enforces this. So the model has two ways to satisfy
repeated positive updates at a dwell state:

1. **Reallocate** a fixed budget of future occupancy toward that state — correct.
2. **Inflate the critic's scale** — cheap, unconstrained, and wrong.

Route 2 is always available and is the failure mode Density-FB targets.

---

# Part 3 — The method

## 3.1 Normalize the object

First, choose what to predict occupancy *over*. Define `x = φ(s)`, the
goal-relevant object coordinates (in our manipulation runs, `φ` slices the cube
xyz / drawer / window / button blocks out of the observation). `F` still sees the
full state; only `B`'s input is restricted:

```
F(s, a, g) ∈ ℝ^d          B(x) ∈ ℝ^d
```

This is the pushforward of the full-state measure and is sufficient for any
reward defined on those coordinates.

Now normalize. Since total mass is always `1/(1−γ)`, define

```
ν^π(·|s,a) = (1−γ) · M^π(s,a,·)          ⇒     ν^π(𝒳|s,a) = 1
```

`ν` is a **probability distribution**. The Bellman equation survives the rescale
intact:

```
M^π(·|s,a) = P_φ(·|s,a) + γ E_{s⁺,a⁺}[ M^π(·|s⁺,a⁺) ]
ν^π(·|s,a) = (1−γ) P_φ(·|s,a) + γ E_{s⁺,a⁺}[ ν^π(·|s⁺,a⁺) ]
```

The `(1−γ)` appears **only** on the immediate term; the recursive coefficient is
still `γ`. And since the immediate term cancels in a difference,

```
‖ Tν₁ − Tν₂ ‖_TV  ≤  γ ‖ ν₁ − ν₂ ‖_TV
```

so the normalized operator is a `γ`-contraction in total variation — the same
guarantee the unnormalized operator has. **Normalizing costs nothing
theoretically and removes route 2 by construction**: a distribution cannot
inflate its scale, it can only move mass around.

## 3.2 The estimator

Represent `ν` as a softmax over a finite query support `U`:

```
score      f(s,a,g,u) = F(s,a,g)ᵀ B(ũ) / τ
weights    w_ij       = softmax_j f(s_i, a_i, g_i, u_j)
```

`ũ` denotes ε-scaled coordinates (each hit-spec block divided by its own
radius). `τ` is a temperature.

**The query support is resampled every update** as the batch's own next-state
object coordinates. This is the single most important implementation detail:
`U` is a *Monte-Carlo sample*, not a fixed discretization. The model never
learns "bin indices"; it learns a continuous two-tower score that is evaluated
at whatever queries show up. (This is what distinguishes Density-FB from the
fixed-prototype "partition" variant — see §7.3, where the partition loses badly
on transfer.)

The immediate term is a kernel row over the same queries:

```
k_ij = softmax_j ( −½ Σ_b ‖ (x'_{i,b} − u_{j,b}) / ε_b ‖² )
```

TD target and loss:

```
t_ij = sg[ (1−γ) · k_ij  +  γ · w̄_ij ]           w̄ = softmax over TARGET towers
L    = − E_i [ Σ_j t_ij · log w_ij ]              (cross-entropy, rows on the simplex)
```

Both `t` and `w` are probability vectors. The loss is cross-entropy, not squared
error, because we are fitting a distribution to a distribution.

## 3.3 Soft vs hard immediate term

- **soft** (`measure_density_hard=False`): `k` is the ε-kernel softmax above —
  the successor smears mass over nearby queries.
- **hard** (`measure_density_hard=True`): `k` becomes a **one-hot** at the row's
  own successor.

This ablation separates "smoothing" from "simplex TD". Empirically **hard is the
better configuration** and its advantage is concentrated in transfer to held-out
reward support (§7.2).

## 3.4 The actor

The measure is only useful if a policy can be extracted. The actor maximizes the
**density readout of the goal reward**:

```
q_i = Σ_j w_j(s_i, π(s_i,g_i), g_i) · κ(g_i, u_j)
```

where `κ(g, ·)` is the same ε-kernel evaluated at the goal. In words: *the
probability, under the predicted future occupancy, of ending up near the goal.*
This slots into a standard flow-BC or DDPG+BC actor objective — the whole thing
stays inside ordinary offline-RL policy extraction.

## 3.5 The zero-shot reward readout

At test time we approximate `∫ r dν` with a pool `U_P` of 16,384 object
configurations drawn from the dataset:

```
PRIMARY:   Q̂_r(s₀,a) = ( r(x₀) + (w − κ(x₀|U_P))ᵀ r(U_P) ) / (1 − γ)
```

The `− κ(x₀|U_P)` term is a de-biasing partner to `r(x₀)`; the two cancel a
finite-pool artifact. **They must be kept together** — an early version of the
G4 calibration probe dropped one half of that pair and produced a dramatic false
null (Spearman .13–.18) that a control later corrected to .767–.790.

## 3.6 Where the estimator sits: it is TD InfoNCE's update (spec 30)

Honesty first: **the hard estimator is not ours.** TD InfoNCE ("Contrastive
Difference Predictive Coding", Zheng–Salakhutdinov–Eysenbach, ICLR 2024,
arXiv:2310.20141) uses the same categorical Bellman update — classify the
observed next state against negatives from the state marginal, bootstrap with
the stop-grad softmax of the next state-action's scores, bilinear two-tower
critic — and its appendix proves the classifier represents the normalized
discounted future-state distribution. With hard immediate labels and matched
query supports, our §3.2 target **reduces to** the categorical Bellman update
underlying their Eq. 10. This was verified against their paper, not assumed.
Rev-1 precision (do not overstate the identity): it is a reduction under
matched conventions, not an implementation identity. Their immediate term
labels the dynamics next state while their bootstrap term labels a *marginal*
draw — the two terms differ in the positive-slot support but share ONE
negative pool `s^(2:N)`; their towers are ℓ2-normalized; their actor is a
goal-contrastive cross-entropy (commanded goal vs marginal negatives, their
Eq. 11) plus an MSE goal-BC regularizer over relabeled goals.

What they do **not** have (verified; Rev-1 corrected): ⚠ TD InfoNCE **is**
offline-capable — it reports offline D4RL AntMaze with the GC-BC regularizer
(their App. D.3) — so "they are online, we are offline" is NOT the delta and
must never be written. The real delta: they learn a raw-goal-conditioned
policy and occupancy estimator, with **no reward-prompted latent policy
family** (`f(s,a,g,x)`, no `F(s,a,z)ᵀB`), **no general reward→policy-index
inference mechanism**, and **no validated zero-shot control for arbitrary
rewards** (their arbitrary-reward evaluation is fixed-policy, appendix-level).
Also absent: the outcome-space projection φ and reward-integrating policy
extraction.

Two consequences for how we talk:
- The estimator is **cite-and-extend**, never claimed as new. Novelty lives in
  the outcome space, the offline extraction, the gauge diagnosis (§6.6), and
  the corrected task interface (§6.7).
- ⚠ The **flagship cells are the soft variant** (`measure_density_hard: False`
  in every tab:main headline cell and the R-A screen). The hard mode — the
  direct TD InfoNCE correspondence — is the ablation, and it wins on reward
  *transfer* (§5.2), not on the headline goal-reaching cells. The soft
  variant's kernel row is normalized over the sampled pool, so its base
  measure is kernel × pool density — data-density-weighted smoothing, stated
  as such since spec 30.
- **The soft variant is a kernel-smoothed occupancy, not label smoothing**
  (Rev 1). With `K_ρ,ε(du|x) = k_ε(x,u)ρ_φ(du) / ∫k_ε(x,v)ρ_φ(dv)`, the
  population soft operator is
  `T_K ν = (1−γ)·E_{x′∼P_φ}[K_ρ,ε(·|x′)] + γ·E[ν(·|s′,a′)]`, whose fixed
  point ν_K is the kernel-smoothed discounted occupancy — valid, but reward
  integration under it obeys `∫r dν_K = ∫(K_ρ,ε r) dν`: the soft model
  evaluates (and its actor optimizes) the SMOOTHED reward `K_ρ,ε r`, not r.
  Sums stay linear; sharp boundaries, conjunctions, max-mixtures, and
  avoid-terms are altered ⇒ for those reward families the hard mode is the
  semantically exact interface (G29-3 consequence recorded in spec 30
  P-soft/P3).

---

# Part 4 — The code

All paths are `tools/wandb_mode_shim/gciql_flow.py` unless noted.

## 4.1 The measure loss

Query support, towers, and online logits:

```python
u_raw = self._raw_block_cols(nobs, blocks)        # [M, d_c] RAW block coords
u_sc  = self._density_queries(nobs, blocks)       # [M, d_c] eps-scaled
b_emb = self.network.select('density_b')(u_sc, params=grad_params)   # [M, d]

f_emb = self.network.select('density_f')(
    jnp.concatenate([obs, acts, g], -1), params=grad_params)         # [2, B, d]

logits = jnp.einsum('hbd,md->hbm', f_emb, b_emb) / tau               # [2, B, M]
w      = jax.nn.softmax(logits, -1)
logp   = jax.nn.log_softmax(logits, -1)
```

Note the leading `2`: this is a **twin-head** critic (the usual two-Q trick),
carried through every einsum.

Bootstrap leg on the frozen target towers, then the CE:

```python
tb = self.network.select('target_density_b')(u_sc)                   # frozen target
tf = self.network.select('target_density_f')(
    jnp.concatenate([nobs, a_plus, g], -1))                          # frozen target
tgt_logits = jnp.einsum('hbd,md->hbm', tf, tb) / tau
w_bar      = jax.nn.softmax(tgt_logits, -1)

if self.config.get('measure_density_hard', False):
    k = <one-hot at the row's own successor>
else:
    k = <eps-kernel softmax row>

t  = jax.lax.stop_gradient((1.0 - gamma) * k[None] + gamma * w_bar)
ce = -jnp.mean(jnp.sum(t * logp, axis=-1))
```

That is the whole estimator: two softmaxes, a convex combination, one
cross-entropy.

**A landmine worth knowing about.** From the docstring:

> No coef knob: this is the ONLY loss on the `measure_only` path when the knob is
> on, weight 1 (17b's `measure_coef` lesson — `total_loss` must not gate this a
> second time).

An earlier spec gated a sibling loss by a coefficient that defaulted to `0.0`,
which silently zeroed the objective. The wave would have trained a model with no
measure loss at all and logged a perfectly convincing null. Density-FB's CE is
deliberately ungated.

## 4.2 The actor readout

```python
kg     = jax.lax.stop_gradient(self._delta_phi(goals, u_raw18, blocks18))  # [B, M]
b_emb18 = self.network.select('density_b')(u_sc18)                        # frozen params
f_emb18 = self.network.select('density_f')(
    jnp.concatenate([obs, actions, goals], -1))                           # frozen params
w18 = jax.nn.softmax(jnp.einsum('hbd,md->hbm', f_emb18, b_emb18)
                     / self.config['measure_density_temp'], -1)
return (w18 * kg[None]).sum(-1)
```

Frozen towers, stop-gradded kernel: the actor reads the measure, it does not
train it.

## 4.3 The zero-shot readout

`scripts/rescuing_fb/eval_density_zeroshot.py`:

```
PRIMARY   Qhat_r = ( r(x0) + (w - kappa(x0|U_P))^T r(U_P) ) / (1 - gamma)
SECONDARY mu   = sum_j w_j B(u~tilde_j);  mu0 = kappa(x0|U_P)^T B(U~P)
          Qhat_r ~= ( r(x0) + (mu - mu0)^T w_r ) / (1 - gamma)
```

## 4.4 The R-A dual index (spec 29)

`_density_z_dual`, `_density_z_encode` (dual branch), and
`density_z_from_reward_samples`. One stored object, one solve, one renorm:

```python
# Sigma_hat_B = E[B(u)B(u)^T], the UNCENTERED second moment, EMA-written in
# update() and stored as an ORDINARY PARAM LEAF (module ZSigmaState) so it
# serializes; read here at frozen params — the EMA is its only writer.
sigma = self.network.select('density_zsigma')()                    # [d, d]
lam   = lam_mult * jnp.trace(sigma) / d          # multiples of the MEAN eigenvalue,
                                                 # same convention as spec28_encodings.py

# the primal moment of the row's reward samples, through the SLOW-EMA tower
# (a ridge solve through a fast-drifting B is the obvious non-stationarity risk).
# Hindsight rows: r = kg = _delta_phi(goals, u_raw) — the query-set-normalized
# kernel, matched exactly to the g path (the standing spec-28 rule).
b_u   = self.network.select(enc_mod)(u_scaled)                     # [M, d] frozen
m_hat = jnp.einsum('bm,md->bd', r, b_u) / r.shape[-1]              # (1/M) Σ_j B̄(u_j) r_j

# the primal→dual correction — the ONLY place it happens, so train and eval
# cannot drift apart in it; ‖ẑ‖ = √d is RE-IMPOSED (a ridge coefficient is not
# norm-preserving, and everything downstream assumes the standing invariant).
c    = jnp.linalg.solve(sigma + lam * jnp.eye(d), m_hat.T).T       # [B, d]
zhat = jax.lax.stop_gradient(_sphere(c, d))
```

The two arms differ only in what the actor integrates against `w(s,a,ẑ)`:

```python
q_kernel = (w * kg[None]).sum(-1)                    # R-A/kernel: actual reward samples
q_latent = (w * (b_emb @ zhat.T).T[None]).sum(-1) / jnp.sqrt(d)   # R-A/latent: B̃ᵀẑ
```

Three details that carry weight:

- **Eval without a pool.** `sample_actions` gets a goal and no query set, so the
  dual branch falls back to `m_hat = B̄(x̃_g)` — the **analytic point prompt**
  associated with a Dirac-style goal reward. ⚠ Rev-1 wording rule: never call
  this the "Dirac/M=1 limit of the reward-sample family". For the UNCENTERED
  encoder the one-sample limit happens to coincide directionally (the pinned
  unit test uses a degenerate queries=goals set), but the CENTERED encoder has
  **no one-sample limit at all**: `m̂ᶜ = (1/M)Σ(B(u_j)−μ)(r_j−r̄)` is
  identically zero at M=1 because `r₁−r̄=0`. Point prompting and reward-sample
  prompting are therefore SEPARATE APIs — point prompt
  `z = √d·norm(B̄(φ(g))−μ)` for native goals, reward samples (M>1, practically
  M≥256) for the general interface — and results for one are not evidence for
  the other. A harness wanting the pool moment calls
  `density_z_from_reward_samples` + `sample_actions_with_z`; general-encoder
  rows must never silently route through the point shortcut. Constant rewards
  give `m̂ᶜ = 0` exactly ⇒ `dual_centered` needs a `‖c‖ < ε_z → z_neutral`
  fallback, logging `‖m̂ᶜ‖`, `‖c‖`, `Var[r]`.
- **`density_z_from_reward_samples` is the general reward interface** — reward
  samples `{(u_j, r_j)}` → index, no goal anywhere. It is the identical solve
  the training rows go through; only the reward samples differ. This is the
  entrypoint gate G29-3 reads.
- **Everything is stop-gradded**: the dual correction must not become a path by
  which the actor's loss reshapes `B`, and the sphere branch is byte-for-byte
  the shipped code, so every pre-existing arm stays bit-identical.
- **`Σ̂` is deliberately UNCENTERED** (`test_sigma_is_ema_uncentered_and_matches_
  spec28_pool`). Spec 30 showed the uncentered statistic is dominated by the
  softmax-gauge mean (‖μ_B‖ = 33–103× the centered signal) and the resulting
  encoder is reward-shift-unstable (§6.6.1c). The planned correction is a
  `dual_centered` mode carrying EMA (μ, Cov) leaves and the centered moment
  `E[(B−μ)(r−r̄)]` — same serialization discipline, exact B-translation and
  reward-affine invariance. Rev-1 implementation pins: μ and Cov on the SAME
  EMA schedule against ONE pinned base measure (the batch next-state marginal,
  i.e. the query distribution — also the measure reward samples are assumed to
  come from; no importance correction for other proposals yet); symmetrize
  `C ← ½(C+Cᵀ)`; eigenvalue floor/jitter; log min-eig and the EMA-vs-probe sync
  errors `‖μ̂−E_probe[B̄]‖`, `‖Ĉ−Cov_probe(B̄)‖_F`. Scope of the invariance
  claim: **translation-gauge (additive) only** — the full linear gauge
  `B→AB, F→A⁻ᵀF` is not fixed (isotropic ridge + sphere norm are not
  A-invariant); the population encoder is exactly invariant, the EMA
  implementation approximates it during joint training.

---

# Part 5 — What Density-FB scores

All cells: OGBench state-based, 500k gradient steps, no reward labels during
training, native evaluation (5 tasks × 50 episodes), **3 seeds**, mean ± 95%
t-based CI. Behavior-cloning floors: **0.111** cube-single, **0.019**
cube-double.

## 5.1 Goal reaching (`tab:main`)

| Environment | Actor | Plain FB | **Density-FB (soft)** | **Density-FB (hard)** | GC-IQL (published) |
|---|---|---|---|---|---|
| cube-single | FQL + BC | 0.420 (10 seeds) | **0.929 ± 0.085** | 0.915 ± 0.087 | — |
| cube-single | DDPG + BC | 0.353 ± 0.38 | 0.879 ± 0.095 | 0.887 ± 0.066 | 0.68 ± 0.06 |
| cube-double | FQL + BC | 0.005 ± 0.016 | **0.736 ± 0.139** | 0.680 ± 0.070 | — |
| cube-double | DDPG + BC | 0.013 ± 0.029 | **0.652 ± 0.055** | 0.600 ± 0.147 | 0.40 ± 0.05 |
| scene | FQL + BC | 0.076 ± 0.057 | 0.628 ± 0.134 | **0.641 ± 0.109** | — |
| scene | DDPG + BC | 0.047 ± 0.057 | 0.563 ± 0.011 | 0.581 ± 0.322 | 0.51 ± 0.04 |
| antmaze-medium | FQL + BC | 0.593 ± 0.25 | 0.563 ± 0.167 | 0.669 ± 0.240 | — |
| antmaze-medium | DDPG + BC | *missing run* | 0.829 ± 0.058 | **0.877 ± 0.097** | 0.71 ± 0.04 |

Per-seed values, so nothing here is a single lucky run:

| cell | seeds |
|---|---|
| soft flow cube-single | .896 / .928 / .964 |
| soft flow cube-double | .680 / .736 / .792 |
| soft flow scene | .672 / .568 / .644 |
| hard flow cube-single | .876 / .944 / .924 |
| hard flow cube-double | .692 / .700 / .648 |
| hard flow scene | .692 / .620 / .612 |
| hard flow antmaze | .772 / .580 / .656 |
| soft ddpgbc cube-double | .676 / .648 / .632 |
| hard ddpgbc cube-double | .668 / .560 / .572 |
| hard ddpgbc scene | .664 / .648 / .432 |
| hard ddpgbc antmaze | .920 / .868 / .844 |

> The hard-flow cube-double wave finished 2026-08-16 (.692/.700/.648 at 500k
> from the runs' own `eval.csv`; an earlier .667 ± .072 in this cell had no
> recorded provenance). A fourth run in that wandb group, `hx_cd_hard_s0_re`,
> is a **restore-continuation of s0** (`restore_epoch=100k`, ran to 600k) and
> is excluded from the seed stats via the results-map's per-run exclusion —
> any `*_re` run sharing a group will otherwise contaminate its per-seed
> columns.

**Readings.**

1. **Cube-double is the load-bearing cell.** Plain FB at 0.005 is *at* the 0.019
   BC floor. Density-FB at 0.736 is not. This is not a tuning difference.
2. **It beats published GC-IQL at half the budget** (GC-IQL: 1M steps, 8 seeds).
   Cube-double 0.652 vs 0.40; scene 0.563/0.581 vs 0.51; cube-single
   0.879/0.887 vs 0.68.
3. **It survives the actor swap.** Going FQL+BC → DDPG+BC costs Density-FB 0.05
   on cube-single and 0.08 on cube-double. The gain is a property of the
   *representation*, not of the flow actor.
4. **AntMaze is a two-sided null.** 0.877 vs a matched GC-IQL control of 0.837,
   with soft at 0.829. *Both intervals contain the control.* No manipulated
   object, no advantage — which is the scoping evidence for the mechanism claim,
   not a failure of it.
5. **Scene hides a task split.** All seeds solve `open`, `unlock-and-lock`,
   `rearrange-medium` (0.88–1.0). `put-in-drawer` (0.02–0.50) and
   `rearrange-hard` (0.0–0.04) stay hard.

## 5.2 Zero-shot reward transfer (`tab:zeroshot`)

Rewards never seen in training. Pool of 16,384 configurations, 25 episodes per
task, 3 seeds. Plain FB is evaluated through **its own** interface
(`z_r ∝ E[rB]`, then rollout under `z_r`) with families, pool, episodes, and
success criterion held identical — only the readout differs.

| Environment | Reward family | Plain FB | **Density-FB (soft)** | **Density-FB (hard)** |
|---|---|---|---|---|
| cube-single | sparse | 0.413 ± 0.308 | 0.816 ± 0.269 | **0.888 ± 0.091** |
| cube-single | dense | 0.488 ± 0.397 | 0.725 ± 0.301 | **0.912 ± 0.087** |
| cube-single | held-out regions | 0.090 ± 0.114 | 0.293 ± 0.052 | **0.647 ± 0.301** |
| cube-double | sparse | 0.013 ± 0.023 | 0.549 ± 0.240 | **0.573 ± 0.126** |
| cube-double | dense | 0.037 ± 0.011 | 0.488 ± 0.163 | **0.520 ± 0.121** |
| cube-double | held-out regions | 0.000 ± 0.000 | **0.177 ± 0.160** | 0.167 ± 0.014 |
| scene | sparse | 0.472 ± 0.124 | 0.395 ± 0.092 | **0.640 ± 0.111** |
| scene | dense | 0.413 ± 0.030 | 0.347 ± 0.219 | **0.627 ± 0.094** |
| scene | held-out regions | 0.053 ± 0.080 | 0.220 ± 0.086 | **0.347 ± 0.038** |

**Readings.**

1. **The hard variant is the one to ship.** It wins nearly everywhere, and its
   margin is largest exactly where transfer is hardest: cube-single held-out
   regions 0.647 vs 0.293, scene 0.347 vs 0.220.
2. **Cube-double held-out regions: plain FB is at exactly 0.000** across all
   three seeds while Density-FB is at 0.17.
3. **Soft loses to plain FB on scene sparse/dense** (0.395/0.347 vs
   0.472/0.413). Only hard clears it there. Worth stating plainly.
4. **Held-out regions are the weak family everywhere.** This is the honest soft
   spot: transfer to genuinely novel reward *support* is much worse than
   transfer to novel reward *shape*.
5. **Cube-single intervals are wide** (±0.27–0.40). Those cells should not be
   read as separations at 3 seeds.

### The control that makes the plain-FB column trustworthy

The `sparse` family *is* the five native OGBench tasks with indicator rewards.
So FB's zero-shot sparse number should reproduce each checkpoint's own native
success:

| env | zero-shot sparse | native (own `z`) | agree? |
|---|---|---|---|
| cube-single | 0.413 | 0.413 | exact |
| cube-double | 0.013 | 0.012 | yes |
| scene | 0.472 | 0.115 | **no, 4× high** |

The scene gap is real and explained, not a harness bug: OGBench's native relabel
on scene's 5-way conjunction is `(k−5)+1` with mean **−2.17**, so the native `z`
is dominated by negative mass, while the dirac family is a clean 0/1 indicator.
Running FB's *own* inferred `z` through the identical rollout code reproduces
0.136 against wandb's 0.14. **Scene's plain-FB numbers are therefore FB's
best case** — which is the conservative direction for a baseline.

## 5.3 Ablation: resampled queries vs a fixed partition

Replacing the resampled query set with 256 fixed prototypes, identical
otherwise (cube-single):

| variant | sparse | dense | held-out regions |
|---|---|---|---|
| Density-FB (soft) | 0.816 ± 0.269 | 0.725 ± 0.301 | **0.293 ± 0.052** |
| Fixed partition (K=256) | 0.725 ± 0.121 | 0.723 ± 0.298 | 0.133 ± 0.201 |

Goal reaching is nearly unchanged; **transfer to held-out regions halves**. The
continuous two-tower score is what generalizes off the prototype set.

---

# Part 6 — The z-indexed interface, and why it fails

## 6.1 What was tried and why

Everything above indexes the policy family by **goals**. FB indexes by a
**latent** `z`, and that is what buys the general reward interface: hand it any
reward, get `z_r ∝ E_ρ[r(x)B(x)]`, run the policy. Goal-conditioning alone is
not an acceptable end state for this project.

Spec 24 swapped **only the index**, keeping the CE/softmax estimator
byte-identical:

```
z(x) = √d · B̄(x̃) / ‖B̄(x̃)‖                    B̄ = TARGET backward tower
z_i  = z(x_{g,i})    with prob 1 − z_rand_frac
     ∼ √d · Uniform(S^{d−1})   otherwise      (FB's standard mix, default 0.5)

scores    f_ij = F(s_i, a_i, z_i)ᵀ B(ũ_j) / τ
actor q   q_i  = Σ_j w_j(s_i, π(s_i,z_i), z_i) · ( B̃(u_j)ᵀ z_i / √d )
```

The actor line is the z-analog of the goal-mass objective: `r_z(u) = B(u)ᵀz` is
FB's reward model, and the actor maximizes its expectation under the predicted
occupancy.

```python
def _density_z_encode(self, goal_obs):
    d = float(self.config['measure_density_dim'])
    blocks = _parse_hit_spec(self.config.get('measure_hit_spec', '19:22:0.4'))
    x_sc = self._density_queries(goal_obs, blocks)                   # [B, d_c] eps-scaled
    b_emb = self.network.select(self._density_z_enc_module())(x_sc)  # [B, d], frozen tower
    z = _sphere(b_emb, d)
    return jax.lax.stop_gradient(z)
```

```python
def _density_z_masked(self, goal_obs, rng):
    z_g = self._density_z_encode(goal_obs)
    rng_sphere, rng_mix = jax.random.split(rng)
    z_rand = _sphere(jax.random.normal(rng_sphere, (B, d)), d)
    rand_frac = float(self.config.get('measure_density_z_rand_frac', 0.5))
    use_rand = jax.random.bernoulli(rng_mix, rand_frac, (B, 1))
    z = jnp.where(use_rand, z_rand, z_g)
    hindsight = (1.0 - use_rand[:, 0].astype(z.dtype))               # [B]
    return jax.lax.stop_gradient(z), jax.lax.stop_gradient(hindsight)
```

## 6.2 It collapses

| venue | z-indexed | g-indexed | BC floor |
|---|---|---|---|
| point-mass-maze | 0.200 ± 0.062 | 0.315 | 0.235 |
| **cube-double** | **0.017 ± 0.025** | **0.736** | **0.019** |

Cube-double is the load-bearing cell: from 0.736 to the BC floor.

> ⚠ On point-mass-maze do **not** say "below BC" — 0.235 sits inside the z arm's
> CI of [0.138, 0.262]. The licensed comparison there is against 0.315.

## 6.3 It is not a training failure

| instrument | z-indexed | g-indexed |
|---|---|---|
| cross-entropy | 5.894 | 5.856 |
| effective sample size | 299 | 296 |
| `k_self` (cube-double) | 0.515 | 0.515 |
| `z_cos_spread` | 0.34–0.35 | — |

The measure under `z` trains *normally*, and the index does not collapse
(`z_cos_spread` far from 1).

## 6.4 Four probes, four eliminations

### Probe 1 — is the index lossy? (`eval_z_identity.py`, actor-free)

Encode pool goals with the agent's own `_density_z_encode`, retrieve by cosine
nearest-neighbour, score by hit-spec ε-ball membership.

| checkpoint | top-1 | top-8 |
|---|---|---|
| z_s0 / s1 / s2 | 0.738 / 0.750 / 0.792 | 0.798 / 0.828 / 0.846 |
| g-indexed control | 0.782 | 0.818 |

**Read against two baselines, both computed rather than assumed:**
- **chance top-1 = 0.0005** (mean 7.8 of 16,383 pool points inside a ball)
- **ceiling top-1 = 0.818** — 18.2% of queries have *no other* pool point inside
  their own ε-ball and auto-fail

So z reaches **90–97% of achievable**, versus the control's 96%, at ~1500×
chance. **The index preserves goal identity.**

> ⚠ **This was originally written as "the index is not lossy." That conclusion
> was wrong and §6.6 overturns it.** The metric is purely *rank*-based, so it is
> satisfied by an arbitrarily tight cone as long as ordering survives. It does
> not measure whether two goals are far enough apart for a downstream actor to
> act on differently — and they are not.
>
> The visible symptom was in this probe's own output all along:
> `nearest_wrong_cosine ≈ 1.0` for all four checkpoints including the control.
> It was dismissed at the time as "saturated and uninformative." It was the
> finding.

### Probe 2 — is the measure broken? (G4 Spearman, actor-free)

Correlate the model's readout masses against a Monte-Carlo estimate of the same
masses. Gate: `spearman_tie ≥ 0.65`.

| checkpoint | pooled | degenerate_frac | mc>0 subset |
|---|---|---|---|
| z_s0 / s1 / s2 | 0.766 / 0.759 / 0.807 | 0.545 / 0.597 / 0.543 | 0.672 / 0.773 / 0.800 |
| g-indexed control | 0.739 | 0.635 | 0.666 |

All pass, and the z arms **beat** the control. I specifically checked the
obvious artifact — a frozen policy is trivially predictable — and it is ruled
out: the z arms have *lower* `degenerate_frac`, meaning they enter **more**
boxes than the working control.

> ⚠ Standing caveat, stated because it is easy to overclaim here: G4 rolls the
> agent's **own** policy, and its metric is Spearman rank agreement — call the
> result **occupancy-ranking fidelity**, not "calibration" (no NLL/Brier/
> reliability numbers exist; spec 30). It means the measure faithfully ranks the
> occupancy of the policy it trained alongside. Successor measures are
> policy-relative (`M^π` models `π`). This rules out a *broken* measure; it is
> **not** proof that a better policy is extractable from it.

### Probe 3 — does the model even use the index? (spec 25)

It did not. Candidate-score spread was **0.00086 max vs 0.00068 mean** — the
model had settled into an ignore-`z` fixed point. So spec 25 ported FB's three
index-coupling mechanisms, all default-off:

**C1 — structural logit tilt.** FB's value *is* `F(s,a,z)ᵀz`; the model cannot
ignore its index. Ours only saw `z` as an input feature the softmax could drop.
The port makes `w` an exponential-family tilt of the base scores, z-sensitive by
construction at every step including step 0:

```python
def _density_z_tilt(self, z, b_emb):
    beta = float(self.config.get('measure_density_z_tilt', 0.0))
    if beta == 0.0:
        return None          # trace-time branch: spec-24 arm stays bit-identical
    d = self.config['measure_density_dim']
    return beta * jnp.einsum('bd,md->bm', z, b_emb) / jnp.sqrt(d)
```

Applied to the **online** logits, the **target** logits, and the **actor's**
q-path — if the target is untilted the bootstrap fits a different family:

```python
logits = logits + tilt[None]                       # ONLINE leg
...
tgt_tilt = self._density_z_tilt(z, tb)             # TARGET leg, TARGET B tower
if tgt_tilt is not None:
    tgt_logits = tgt_logits + tgt_tilt[None]
```

**C2 — reward-decodability auxiliary.** FB gets a shaped `B` from
orthonormalization; ortho is convicted on manipulation, so instead: per batch,
draw a synthetic kernel reward, solve a closed-form ridge for `v*`
differentiably through `B`, and penalize `‖B_U v* − r_c(U)‖²`.

**C3 — index-consistent actor supervision.** Two knobs: mask the imitation term
to *hindsight* z rows (rows whose `z` came from real future coordinates rather
than a random sphere draw), and encode the index with a **slow EMA** copy of `B`
so the conditioning stops drifting with the critic.

**New instrument.** `density_w_ztv = E[TV(w(s,a,z), w(s,a,z'))]` — the direct
index-sensitivity meter the spec-24 wave lacked, logged unconditionally in z
mode. *This is what makes the next result sayable at all.*

**Stage-1 result** (cube-single, 500k, 1 seed each; BC floor 0.111, g-anchor
0.93 which already reaches 0.792 by 50k):

| arm | success | `density_w_ztv` | ztv gate | E1 ≥ 0.5 |
|---|---|---|---|---|
| C1 tilt | 0.096 | 0.054 | fail | **fail** |
| C2 rdecode | 0.108 | 0.009 | fail | **fail** |
| **C3 actor** | 0.108 | **0.602** | **pass** | **fail** |
| C1+C2+C3 | 0.108 | 0.588 | pass | **fail** |

**C3 restored index-sensitivity ~65× and bought exactly zero success.** The
`full` arm reproducing C3's value almost exactly says this is C3's doing, not an
interaction.

Two side findings worth keeping:
- **C1's structural guarantee did not hold at β=1.0.** The spec argued `F` can
  reshape around the tilt but not delete it; ztv decayed 0.093 → 0.054, so it
  largely deleted it. A dose sweep was anti-scoped, so this is an observation.
- **B-geometry is fine.** A ridge-decodability probe gives dense-family R² of
  0.853 (g) and 0.68–0.70 (z) against a dimension-matched RBF control of 0.466.
  Sparse and region are **uninformative, not negative**: the RBF control also
  reads ~0.02 there, so nothing linear decodes those families at this pool size.
  *The spec's own control was mis-specified* — it asked for a raw-coordinate
  baseline "known high" on a Gaussian bump, which is not linear in six
  coordinates and could never clear its gate.

### Probe 4 — with sensitivity restored, is the readout usable? (spec 27 Q2)

**First, what does *not* work as an instrument.** E2a/E2b score `success_final`,
a pipeline that ends in an actor rollout. On a checkpoint whose actor is at the
BC floor, that reads ~0 regardless of readout quality:

| arm | sparse | dense | region |
|---|---|---|---|
| E2a actor / full | .064 / .072 | .072 / .072 | .010 / .000 |
| E2b actor / full | .040 / .088 | .056 / .072 | .000 / .000 |
| cube-single anchors (soft / hard) | .816 / .888 | .725 / .912 | .293 / .647 |

Its control gate passed **bit-exactly** (`diff 0.000` vs archived, on both a
cube-double and a cube-single control) and the metric *still* could not answer
the question. **A passing control proves the harness is sound; it does not make
a pipeline metric able to isolate a component.**

**The actor-independent instrument settles it:**

| checkpoint | `spearman_tie` | degenerate_frac | mc>0 |
|---|---|---|---|
| cube-single control (g) | 0.778 | 0.606 | 0.801 |
| cube-double control (g) | 0.739 | 0.635 | 0.666 |
| **z25_cs_actor (C3)** | **0.728** | **0.463** | 0.718 |
| **z25_cs_full** | **0.713** | **0.503** | 0.704 |

Both pass, sit *between* the two g-indexed controls, and again have **lower**
`degenerate_frac`. **The readout has control-level occupancy-ranking fidelity.
Extraction is the blocker.** (Spec-30 terminology note: this is a Spearman
rank-agreement result, not probabilistic calibration — no NLL, Brier, or
reliability measurement exists; say "occupancy-ranking fidelity", never
"calibrated".)

**And the sharpest single result in the whole z program:** checking where E2b's
argmax actually points — the coordinate percentile of the selected candidate
against the true reward center — gives **0.35 (actor) and 0.47 (full) against a
0.50 null**, *despite exact matches existing in the candidate set*. **Reward →
index assignment is at chance.**

## 6.5 Where that left the diagnosis (as of spec 27)

> ⚠ **This section records the diagnosis *before* spec 28 measured B's geometry.
> Two of its rows and its proposed fork are superseded by §6.6 and §6.7. It is
> kept because the reasoning was the path to the fix, and because the row that
> turned out to be wrong was wrong in an instructive way.**

| stage | status *as then believed* | later verdict |
|---|---|---|
| goal → index encoding `z(g)` | "works" (retrieval 90–97% of ceiling) | **WRONG — §6.6.** Retrieval is rank-based; the index was collapsed to a single ray. |
| measure trained under `z` | works | holds |
| measure *uses* the index | fixable (C3 raised ztv 65×) | holds — and it changed nothing on its own |
| policy extraction from `z` | broken | holds — **cause identified in §6.6, fixed in §6.7** |
| reward → index assignment | broken | holds — **cause identified in §6.6, fixed in §6.7** |

The reading at the time was that the failure sat at the two **ends** rather than
the middle. That was right. What it could not see was that both ends fail for
the *same* reason, which is B's geometry.

**Eliminated as sole causes** (this part stands, and is why the geometry
explanation had to be found): lossy encoding, a broken measure, the model
ignoring `z`, BC dilution from `z_rand_frac=0.5`, and the non-stationary index.
C3 addressed the last two directly and success did not move.

The fork proposed here was "repair reward → index assignment, or bypass `z` with
a reward-embedding actor." The first branch is what §6.7 does.

## 6.6 Spec 28 — the diagnosis, solved

Everything above localizes the failure to two ends: the actor cannot use the
index, and reward→index assignment is at chance. Spec 28 found *why*, and the
cause is a single property of `B`.

### 6.6.1 The mechanism: `Σ_B` is dominated by a softmax-gauge mean

The cross-entropy loss pins only the **product** `FᵀB`. Nothing constrains `B`'s
Euclidean geometry — it is gauge-arbitrary. Measured on every checkpoint:

| quantity | value |
|---|---|
| effective rank of `Σ_B` | **1.0002** out of `d = 50` |
| condition number | **1e6 – 1e7** |
| eigenvalues (`z25_cs_actor`) | max 1.18e6, min 0.115 |

> ⚠ **Spec-30 correction to the reading of this table.** As first written, this
> section said "`B` collapses onto essentially one direction." That overstates.
> The softmax has an *exact translation gauge*: `B(x) → B(x)+b` shifts every
> logit in a row by the same `Fᵀb`, which the softmax cancels — the **mean of B
> is unidentifiable**, free to grow without changing any prediction. And it
> does: measured (results/spec30_geom/, control-gated against the archived
> spec-28 numbers),
>
> | ckpt | ‖μ_B‖ | E‖B−μ‖ | eff_rank Σ_B (uncentered) | **eff_rank Cov(B) (centered)** |
> |---|---|---|---|---|
> | z25_cs_actor | 1087.8 | 10.6 (103×) | 1.0002 | **20.8 / 50** |
> | d18_cs_dens_s0 (g) | 615.9 | 18.6 (33×) | 1.0017 | **6.3 / 50** |
> | z24_cd_z_s0 | 496.5 | 11.4 (44×) | 1.0010 | **19.1 / 50** |
>
> The centered geometry is **rich, not rank-1**. What the uncentered table
> measures is a gauge mean 33–103× the informative signal. Consequently
> **centering alone rescues the point index**: `normalize(B(g) − μ_B)` reads
> pairwise cos .001–.004 (was .9995+) and reconstruction AUC .990–1.000 (was
> .43–.86), verified on disjoint fit/eval pool halves. The ridge solve still
> earns its keep — participation ratio 28–30 vs 6.6–19.8 for centered pointing,
> and a small AUC edge — but the dominant effect is mean subtraction. All
> spec-28 *measurements* below stand; this reinterprets them.

Two consequences follow mechanically.

**(a) `z(g)` is collapsed.** Over 4096 dataset goals, mean pairwise cosine is
**0.998–1.000** against a sphere reference of **0.002**; median distance is
0.05–0.48 where `√(2d) = 10`. Every goal maps to nearly the same direction,
because everything is projected onto the dominant eigendirection before the
sphere normalization.

**(b) The primal moment is the wrong object.** FB embeds a reward as the
**primal** moment `z_r ∝ E[r·B]`. But once the softmax removes FB's linear
identity `Q_r = Fᵀ E[rB]`, the readout `B̃(u)ᵀz/√d` requires the **dual**
coefficient `c = (Σ_B + λI)⁻¹ m`. Under a whitened `B` these coincide — which is
exactly what FB's orthonormality loss enforces, and exactly what this project
dropped (correctly: ortho is a convicted toxin here, first-passage 0.967
no-ortho versus 0.225 with). Without whitening they do not coincide at all:

```
cos( primal moment , dual ridge )  =  0.010 – 0.025      ← nearly orthogonal
```

**We were using one object where the algebra requires the other.** This also
explains why dropping ortho was harmless on the g path: *the g path never reads
reward off `Bᵀz`.*

**(c) Spec-30 addition: the uncentered encoder is reward-shift-unstable.**
`r → r+β` leaves the optimal policy unchanged but moves the moment by
`β·E[B]` — almost exactly the gauge direction. Measured: the uncentered ridge
encoder (the one the trained R-A arm uses) maps `kg` and `kg+1` to
near-orthogonal indices, cos **.035 / −.000 / −.044** across the three
checkpoints. The centered encoder
`ẑᶜ ∝ (Cov(B)+λI)⁻¹ E[(B−μ_B)(r−r̄)]` reads **1.000000** on the same probe and
is exactly invariant to `B+b`, `r+β`, and (after √d-normalization) `αr` —
centering removes both translations identically: `(B+b)−(μ+b) = B−μ`,
`(r+β)−(r̄+β) = r−r̄`. This is the P1 correction of spec 30: the centered form
is the principled object, and the pending G29-3 must include shifted copies
of each reward family as the invariance check.

### 6.6.2 The consequences, measured (P0a)

The action-ranking probe computes both readouts under the **same** z-conditioned
weights, so conditioning and readout are separated without retraining.

| quantity | z cells | reference |
|---|---|---|
| actor sensitivity to different goal-derived `z` | **0.000** (4e-6…5e-4) | own noise floor 0.19 |
| actor sensitivity to a *random-sphere* `z′` | **1.9–2.0** | ceiling 1.83 |
| span of latent readout `B̃ᵀz/√d` across actions | **0.000–0.002** | kernel readout 0.13–1.15 |
| ρ(q_g, q_z) | 0.01–0.31 | null 0 |
| AUC of kernel readout `q_g` under z-conditioning | 0.65–0.71 | chance 0.50 |

Two readings matter. **The actor is `z`-capable** — a random `z` moves it to the
sensitivity ceiling — it is simply never handed two distinguishable goal-derived
indices. And **both effects appear on the g-indexed anchors too** (cos 0.998,
readout span 0.001), so this is a property of the encode map and of `B̃ᵀz`, not
of z-training.

> ⚠ **This overturns spec 24's "no index collapse."** That conclusion rested on
> `z_cos_spread ≈ 0.35`. But the instrument is computed on the **mixed** index
> (50% random spheres), so a *fully collapsed* hindsight branch dilutes to
> **0.326** and a fully spread one to **0.113**. Observed 0.346–0.350 matches
> full collapse almost exactly. The instrument could not see what it was built
> to detect.

> ⚠ Control gate: cube-single and cube-double **pass**; point-mass-maze **fails**
> (AUC_g 0.586 < 0.65) because its RND data contains no goal-directed actions, so
> the "good action" label is invalid there. Those cells are recorded unlicensed
> and are not read.

### 6.6.3 The repair works, at readout (P0b / P0c)

**Reward reconstruction from the pool** (mean over 14 cells, target tower):

| checkpoint | metric | point (current) | moment | **ridge (dual)** | oracle | random |
|---|---|---|---|---|---|---|
| g_cd | AUC | 0.871 | 0.867 | **0.999–1.000** | 1.000 | 0.501 |
| g_cs | AUC | 0.484 | 0.480 | **0.990–0.994** | 1.000 | 0.516 |
| z24_cd s0–s2 | AUC | 0.81–0.86 | 0.81–0.86 | **1.000** | 1.000 | 0.50–0.57 |
| z25_cs (4 arms) | AUC | **0.43–0.57** | 0.42–0.55 | **0.99–1.00** | 1.000 | 0.48–0.55 |
| z25_cs (4 arms) | Spearman | **−0.05–0.28** | same | **0.58–0.72** | 1.000 | −0.07–−0.01 |

On cube-single the encoding actually in use is **at or below chance**. The
un-normalized moment scores *identically* to the normalized one on every metric
(all are positive-scale-invariant, asserted by test), so **normalization is
provably not the variable** — it is primal versus dual.

**Reward→index assignment** (coordinate percentile; 0 best, 0.50 chance; the
convention pinned by reproducing spec 27's published 0.35 and 0.47 from archived
artifacts):

| checkpoint | current readout | moment | raw-B moment | **ridge** |
|---|---|---|---|---|
| z24_cd s0/s1/s2 | 0.506 / 0.431 / 0.489 | 0.000–0.001 | 0.18–0.39 | **0.000–0.001** |
| z25_cs_actor | 0.347 | 0.020 | 0.611 | **0.007** |
| z25_cs_full | 0.473 | 0.062 | 0.737 | **0.002** |
| g anchors cs / cd | 0.008 / 0.000 | 0.013 / 0.000 | 0.501 / 0.130 | 0.011 / 0.000 |

**The dual correction moves assignment from chance to the g-anchor level.** The
sharpest isolation is the **raw-B** column: in exactly the basis the actor's
reward term lives in, the primal moment is at or *below* chance while the dual
is near-perfect.

> Honest phrasing, because it is easy to overclaim: the sphere-normalized
> *moment* also beats chance (0.000–0.062), since normalizing candidates removes
> the dominant-magnitude confound. The supported claim is **"the dual correction
> fixes assignment and is strictly better than the moment everywhere, decisively
> so in the raw-B readout"** — not "only ridge works."

> ⚠ **Readout-only.** `‖z‖ = √d` wherever `z` enters `F` or the actor, and a
> ridge coefficient is not norm-preserving, so *conditioning* on it is
> out-of-distribution for these checkpoints. This licenses **building** the
> repair; it is not an end-to-end claim.

Control gates: the g-indexed rollout control reproduced its archived family
means at **diff 0.000**, and the assignment protocol reproduced the archived
`top1_idx` **bit-exactly on 70/70 cells**.

### 6.6.4 Status of the 2×2

The conditioning-versus-readout 2×2 (cells ii and iii) is **implemented, tested,
and deliberately not run**. P0a convicts both as written: cell ii conditions the
actor on the same collapsed index, so no reward readout can make it produce
goal-specific behaviour, and cell iii's readout is convicted on g-anchors that
never saw `z`. The launcher carries a DO-NOT-RUN gate header.

The licensed repair is **R-A**: a dual-corrected index, with the actor
integrating **actual reward samples** rather than `Bᵀz` — the latent becomes a
policy index only, and reward semantics stay external, which is precisely what
made the g path work. R-B (reward-embedding actor) is not indicated by this
evidence.

## 6.7 R-A — the repair, and what it scores

Spec 28 (§6.6) showed the index and the readout live in **dual spaces that had
drifted nearly orthogonal**. R-A changes exactly two things.

**1. The index becomes a dual coefficient rather than a primal feature vector.**

```
before:   z(g) = √d · normalize( B̄(g) )
after:    ẑ    = √d · normalize( (Σ̂_B + λI)⁻¹ m̂ ),    m̂ = (1/M) Σ_j B̄(u_j)·r_j
```

**2. The actor integrates actual reward samples rather than a latent
reconstruction.**

```
before:   q = Σ_j w_j(s,a,z) · B̃(u_j)ᵀz/√d
after:    q = Σ_j w_j(s,a,ẑ) · r(u_j)
```

`Σ̂_B` is EMA-tracked as an ordinary parameter leaf so it serializes and the eval
path reads the identical object — batch statistics would reintroduce the
train/eval mismatch that broke the first z build. The encode runs through the
slow-EMA backward tower, because a ridge solve through a fast-drifting `B` is the
obvious non-stationarity risk. `‖ẑ‖ = √d` is re-imposed after the solve so the
conditioning distribution keeps its standing invariant.

### Why change 1 is the unlock

Because `Σ_B` is rank-1 dominated, every `B(x)` is approximately one shared
direction plus a small `x`-specific part:

```
B(x)  ≈  (large) · v  +  (tiny) · δ_x
```

so `normalize(B̄(g))` returns ≈ `v` for **every** goal — the measured collapse.
Multiplying by `(Σ+λI)⁻¹` divides each eigendirection by its eigenvalue,
suppressing `v` (eigenvalue ~1e6) against the goal-specific directions
(~0.1). The dual is therefore not a correction to the primal; at
cos = .010–.025 it points somewhere else entirely.

Spec 30 sharpened what `v` *is*: the softmax-gauge **mean** of B (§6.6.1's
correction block — ‖μ_B‖ is 33–103× the centered signal, and the centered
covariance has effective rank 6–21, not 1). So the ridge solve works here
mostly because dividing by the top eigenvalue is a stand-in for subtracting
the mean; the centered encoder does the same job by construction and adds
exact shift invariance.

**Why the loss permitted this.** The cross-entropy pins only the *product*
`FᵀB`: for any invertible `A`, sending `F → FA⁻ᵀ` and `B → BA` leaves the loss
unchanged, so B's geometry is **gauge-arbitrary** and nothing pushes `Σ_B`
toward isotropy. FB's orthonormality loss is exactly what fixes that gauge, and
this project removed it — correctly, since it caps success at 0.225 here.
Removing it was harmless on the goal-indexed path, which never reads reward off
`Bᵀz`, and fatal on the latent path. **The gauge fixing was deleted while a
formula that is only valid in the fixed gauge was kept.**

### The gates, in order

| gate | question | result |
|---|---|---|
| G29-0 | are dual indices de-collapsed? | **PASS** — PR2 1.0 → 9–38; the primal-moment control stays at 1.0, isolating the *solve* as the cause |
| G29-1 | does the actor gain goal-sensitivity? | **PASS** — .132 vs .000 for every prior z arm, 63% of its .210 noise floor (the working g anchor is 47% of its own) |
| G29-2 | does it reach goals? | **see below** |
| G29-3 | does it transfer to unseen rewards? | **NOT YET RUN** |

At G29-1 the pathology also **inverted**: sphere-`z′` sensitivity fell to .065,
*below* goal-`z′` at .132. Previously these actors moved 1.9–2.0 for a random
off-manifold index and .000 for a real goal — they responded only to nonsense.
And ρ(q_g, q_z) rose to **.929**, against **−.026** on the g anchor: under the
dual index the closed-form readout and the exact kernel rank actions almost
identically, which is §6.6.3's prediction realized in a trained model.

The de-collapse also **survives training**, which G29-0 alone could not
guarantee: `z_cos_spread` ran .128 → .149 over the kernel arm's 500k steps
(references: .113 fully spread, .326 fully collapsed under the 50%-sphere mix),
rising to .157 at 185k and settling at .148–.154. The index never re-collapsed.
The mechanics of the encode — the EMA `Σ̂` as a serialized parameter leaf, the
slow-EMA tower, the analytic-point-prompt eval fallback — are in §4.4.

### G29-2 — the screen (cube-single, 500k, **seed 0 only**)

| arm | @500k | mean(≥350k) |
|---|---|---|
| **R-A/kernel** (dual index + actual reward samples) | **0.812** | 0.743 |
| R-A/latent (dual index + `B̃ᵀẑ` readout) | 0.368 | 0.291 |
| prior z arms (spec 24 / 25) | .017 – .108 | |
| behavior cloning floor | 0.111 | |
| goal-indexed anchor | 0.93 | |

**This is the first z-conditioned Density-FB that works** — roughly 7.5× every
previous z arm and about 87% of the goal-indexed anchor. Even the latent arm,
whose readout scored .108 before, now reaches .368.

The two arms share the dual index and differ only in the actor's reward, so the
split isolates change 2 as what converts a repaired index into control. The
100k probe predicted it before the results existed: kernel's actor sensitivity
lifted to .132 while latent's stayed at .006.

> ⚠ **Five limits, all load-bearing** (the last two added by spec 30).
> **One seed per arm** — the spec promotes discriminating cells to cube-double
> plus a second seed, and that has not run.
> **Eval noise is real**: the kernel arm dipped to **.564 at 400k** before
> reading .812 at 500k, so quote .812 by convention *and* .743 as the ≥350k
> mean; a single eval at exactly 500k can be a favourable draw.
> **G29-3 has not run.** The zero-shot reward-family eval through the dual
> interface *is* the general-reward-interface claim. Goal reaching through a
> repaired index does not license that sentence.
> **The .812 rollout used the analytic point prompt**: eval-time
> `sample_actions` has no pool, so ẑ ∝ (Σ̂+λI)⁻¹B̄(φ(g)) (§4.4). It validates
> the corrected *goal* code; `density_z_from_reward_samples` — trained-through
> and unit-tested — has never driven a rollout, and for the centered encoder
> no point shortcut to it exists (§4.4), so G29-3 must go through actual
> reward pools (M ≥ 256), which is exactly what it exercises.
> **The encoder in this run is shift-unstable** (§6.6.1c): cos(enc(r),
> enc(r+1)) ≈ 0. The centered form is the identified fix; G29-3 runs with it
> and includes shifted reward copies.

> This is a **new arm**: archived spec-24 and spec-25 numbers are not comparable
> to it in one table without saying so.

### What did not fix it

Worth recording, because five plausible repairs were tried first and every one
failed: masking imitation to reachable indices; a slow-EMA encoder to stop the
index drifting; making the measure index-sensitive (C3 raised sensitivity **65×**
and success did not move at all); the structural logit tilt (which deleted its
own tilt at β=1.0); and the reward-decodability auxiliary (inert).

Every one of those addressed **training dynamics**. None touched the geometry.
The failure was never dynamics.

---

# Part 7 — The mechanism story is not yet established

The empirical results are strong; the *causal* explanation is not. A synthetic
"fork-and-dwell" study was built to demonstrate the scale/capacity pathology
directly, and it has now failed its gate **twice**.

- **v1 failed by instrument.** Injecting duplicated `s'=s` rows changes the
  *empirical kernel*, so the exact successor measure of the corrupted kernel
  drifts identically and all five converged tabular arms sat on that curve. The
  conclusion was that the thesis is an **estimator** claim, not a fixed-point
  claim, so the intervention must hold the kernel fixed.
- **v2 applied multiplicity as a sampling weight** — keyed on the dwell *state*,
  not on `s'==s` rows, because a transition-keyed weight is algebraically
  identical to v1's duplication. Two **positive controls** prove both facts, and
  a byte-identity test pins that the empirical kernel is unchanged across weight
  factors.

**Gate P1-v2 fails both halves**, and the second half is the interesting part:

- **Half A** (separation should appear under rank constraint): the constrained
  arms are **dead, not flat** — margin ratio ≈ 0.00–0.06 at weight 1 already, so
  slopes are indistinguishable from zero. Their high `flat_ratio` (14.7, 20.8)
  is precisely the near-zero-over-near-zero artifact the `unnorm_slope_nonzero`
  guard exists to reject. The task hardening overshot.
- **Half B** (the converged tabular control should show *nothing*): instead
  `joint_unnorm` decays **1.036 → 0.705** with the unnormalized CI excluding
  zero.

**Mechanism:** the tabular fixed point is provably weight-invariant, so the
control's drift cannot be a fixed-point effect — it is **optimization rate**. At
weight 20 the decision state receives ~0.1% of draws; an unnormalized regression
learns that row only from its own draws, whereas softmax-over-pool arms couple a
starved row to well-sampled ones. **The channel is sample starvation, not
capacity** — which needs *headroom* to be visible, which the converged column has
and the rank-constrained columns do not. That is why both halves fail together.

> ⚠ **The limitation to carry into the paper.** The separation lives in the
> *joint unnormalized* critic (0.433 → 0.158) while **FB's own bilinear critic is
> flat** (0.437 → 0.417), with both normalized arms ≥ 0.95. So a claim of the
> form "FB's estimator is pathological under dwell" is **not supported** by this
> synthetic as built.

**Spec 30 added two more counterweights to the count-based dwell argument,
both verified.** First, the population LSIF objective has a *finite* scale
optimum (scaling m by c changes the loss by c²/2·E_ρ[m²] − c·E_{M^π}[m]), and
dwell states appear proportionally more often as ρ-negatives too — in our own
estimator this is built in, since the query support is literally the batch's
next states. Second, the "same successor repeated L times" story counts in
φ-space, but the plain-FB baseline's backward map receives the **full
observation** (agents/fb/model.py — the arm keeps moving while the object
rests), so the failing baseline never literally sees identical successors.
The paper wording adopted in consequence: we claim the empirical contrast and
name outcome imbalance plus the unconstrained scale direction as *candidate*
factors — no unique causal channel. The missing control is plain FB given the
same φ(s′), now a standing open item.

## 7.1 The cube bottleneck probe (2026-09-04) — the bottleneck story is refuted, not merely unestablished

Part 7 above says the mechanism is *not yet established*, on the strength of two null
synthetics. Two analysis-only tests on the real cube checkpoints now go further: on
cube-single the bottleneck account is **contradicted in the measured direction**, and the
remaining candidate factor named above (unconstrained scale) is **ruled out**.

Spec `docs/specs/2026-09-04-cube-bottleneck-mechanism-probe.md`; code
`[seam] scripts/probes/cube_bottleneck/`; results
`[seam] analysis/cube_bottleneck/cube-single-play-v0/RESULTS.md`. Deciding pair `fb_ref`
vs `density_fb_state` — both towers read the full observation, so the only difference is the
estimator. 500k, 3 seeds, 50 episodes per cell. Gates: G1 reproduces `table_privcoord`
within 0.024 on all 10 arms; G2 kept 64/64 bank rows; G4 passed 44/64.

**Test 1 — removing the bottleneck does not rescue FB (R-HELD2, 3/3 seeds).** Starting the
episode from a *recorded, in-distribution* held state (qpos/qvel restore, not the old S2
clamp) moves plain FB by Δ = +0.020 / +0.100 / +0.032, from 0.21 to 0.26, while Density-FB
sits at 0.99. FB's failures from that start are failures of retention: **49–56% end with the
cube back on the table**, and FB holds it for 70–75% of the first 50 steps against
Density-FB's 81–86%. Density-FB itself *gains* from the held start (0.90 → 0.99), so the
floor clause does not fire.

**Test 2 — at the bottleneck, plain FB is the BETTER discriminator.** Within-state AUC of
successor mass on held futures against a rollout ground truth (`p_secure`, 8 continuations
× 25 steps per action, 12 actions per state, goal-marginal):

| arm | P2 `auc_heldmass` | P2 `auc_Q` | cube-single success |
|---|---|---|---|
| `fb_ref` (plain FB) | 0.55 / 0.78 / 0.75 | **0.95 / 0.95 / 0.91** | 0.21 |
| `density_fb_state` | 0.44 / 0.55 / 0.64 | 0.55 / 0.50 / 0.50 | **0.90** |
| `density_fb` (obj. coords) | 0.78 / 0.82 / 0.73 | 0.55 / 0.85 / 0.70 | 0.94 |
| `fb` (`ortho_coef` 1.0) | **0.91** | 0.91 | **0.096** |

None of R-MEAS1–3 fires, because all three presuppose FB is the weaker discriminator. The
ordering is reversed, and the arm with the **best** bottleneck discrimination (`fb`, 0.91)
is the **worst** controller (0.096). Bottleneck discrimination and control performance are
anti-correlated across these four arms, so it cannot be the channel the win runs through.

**R-MEAS4 negative — the unconstrained-scale route is dead.** Part 7 kept "the unconstrained
scale direction" as a candidate factor. FB's mean predicted row mass is **87–105 against the
true 1/(1−γ) = 100**, varying less than ±8% across pre-contact / at-contact / held (ratio
0.94–0.98; the gate wanted > 3). FB's measure is well calibrated in total mass and
phase-invariant in it. That candidate should be dropped from the paper's wording.

**What survives.** The empirical contrast is untouched — Density-FB still wins by 4× at
state input. What does not survive is the *location*: the difference is not at acquisition.
Both the phase diagnostic and this probe put it after control is established, in retention
and transport. A proposed rewrite of `introduction.tex:10-27` on that basis is in
`[seam] analysis/cube_bottleneck/PROPOSED-INTRO-PARAGRAPH.md`, left unapplied.

**Two implementation facts recorded here because they will bite the next probe.**
(i) `[torch] scripts/value/counterfactual_value_probe.py:165` sets raw `ctrl = 0.0` to
"establish the grasp"; that is the gripper's **open** extreme. It passes a purely geometric
`held` test while contact collapses to 0.01 — a latent drop. 255 (action channel +1) closes.
(ii) `obs[18]` is `||cfrc_ext||/50`, which depends on `data.ctrl` — state no dataset records
— so a qpos/qvel restore reads contact 0.00 after `mj_forward`, 0.16 after
`mj_rnePostConstraint`, and only reaches the recorded 0.963 after two settle steps with the
gripper commanded closed. Any future restore-based probe must rebuild it or it will silently
evaluate held states that read as untouched.

---

---

# Part 8 — Honest summary

**What is solid.** On OGBench manipulation, Density-FB substantially outperforms
plain FB (cube-double 0.736 vs 0.005 against a 0.019 floor), beats published
GC-IQL at half the budget, survives the actor swap, and transfers zero-shot to
unseen rewards with a passing native control on the baseline column. The hard
immediate target is the better configuration and should be the shipped default.

**What is not ours (spec 30, Rev 1).** The hard estimator's update is TD
InfoNCE's categorical Bellman update, verified against their paper (§3.6) —
cite and extend, never claim; and TD InfoNCE is itself offline-capable, so the
delta is the reward-prompted latent interface, never "offline vs online".
Covariance-inverse task regression is also established prior art: Finer BFM's
§3.1 states the FB-vs-SF contrast `z = E[rB]` vs `(E[φφᵀ])⁻¹E[rφ]` verbatim
(an earlier fetch missed it — rebuttal RETRACTED), and OpTI-BFM (ICLR 2026
oral, arXiv 2510.20264) formalizes BFM task inference as exactly this reward
regression (uncentered normal equations, online reward feedback). What is ours
on that axis is strictly the **softmax-gauge × task-inference interaction**:
the density objective leaves B identifiable only up to density-invisible
transformations, so B is not automatically a valid task basis, the dominant
additive mode is measured, and the corrected regression must be the CENTERED
one. And the comparison against plain FB is not yet estimator-pure: Density-FB
receives the goal-relevant coordinate blocks and tolerances (privileged φ/ε)
that the baseline does not, and the TD InfoNCE / outcome-space LSIF controls
have not run. The scope of "any reward" is rewards measurable on the chosen
outcome coordinates and supported by the data — never arbitrary MDP rewards
(and under the soft variant, evaluated rewards are kernel-smoothed, §3.6).

**What is scoped.** AntMaze shows no advantage in either direction — consistent
with the mechanism only mattering when a goal-relevant object stays still while
actions continue.

**What now works that did not.** The FB-style latent interface, after the R-A
repair (§6.7). Both broken ends traced to one cause — B's uncentered geometry
is dominated by a softmax-gauge mean 33–103× the informative signal (the
centered covariance is rich, eff. rank 6–21 of 50), so the index and the
readout had drifted nearly orthogonal once the gauge-fixing orthonormality
term was removed. Replacing the primal index with the dual coefficient and
letting the actor integrate real reward samples takes cube-single from 0.108
to **0.812**, against a 0.111 floor and a 0.93 goal-indexed anchor. ⚠ One
seed; the rollout used the analytic point prompt, not the reward-sample
encoder — and the centered encoder has NO one-sample shortcut at all
(r₁−r̄=0), so the two prompting modes are separate APIs and the general-
interface eval must go through actual reward pools; the run's encoder is
reward-shift-unstable (centered form = identified fix); and the zero-shot
reward-family eval that would license the *general reward interface* claim
has not run.

**What is unproven.** The causal story. Two synthetic rounds have failed to show
the pathology in FB's own architecture, and the one real effect found lives in a
critic Density-FB does not use.

**Standing caveats for anyone quoting these numbers.**
- Published GC-IQL/TD-JEPA numbers are **quoted, not matched-seed reruns**. Only
  FB has been re-run in our codebase.
- 3 seeds against 8–10 in the published work; several of our intervals are wider
  than the differences they bracket.
- Never quote a mid-run eval. A cell in this program read 0.984 at 450k and
  0.876 at 500k.
- Never say "calibrated" for G4 — it is Spearman rank agreement
  (occupancy-ranking fidelity). Probabilistic calibration was never measured.
- Verify a run's identity from its recorded config, never its group or directory
  name. In the torch codebase `agent: fb` is the **Gaussian+BC** actor and
  `agent: fb_flowbc` is the **flow** actor — that exact confusion once put the
  DDPG+BC number into the FQL+BC row of a published table.
