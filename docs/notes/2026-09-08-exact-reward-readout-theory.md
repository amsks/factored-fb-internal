# Exact reward readout for a successor-measure critic: method and theory (2026-09-08)

Status: theory note for a Method + Theory section. Every claim about the implementation
was checked against the working tree on 2026-09-08 and carries a `file:line` citation.
Line numbers refer to `impls/critics/density_fb.py` unless another file is named. The
file was being edited in the working tree while this note was written (an uncommitted
`'kernel'` actor-reward mode, see section 0.4), so the line numbers are those of the
working tree, not of `HEAD` (`6cc2952`).

Vocabulary used throughout: critic, actor, value, reward, TD target, successor measure,
policy, rollout. Each symbol is defined once, at first use.

---

## 0. The method, as implemented (verified)

### 0.1 Critic: a successor measure over sampled candidates

Per update the critic receives a batch of `B` transitions `(s_i, a_i, s'_i)`. The
candidate set `U = {u_1, ..., u_M}` is the batch's own next states, so `M = B`
(lines 878-881: `nobs_q`, `u_sc` are built from `batch["next_observations"]`). Two
networks are trained: the forward tower `F_h(s, a, c)` (two heads `h = 1, 2`, line 517-523,
one MLP over the concatenation `[s, a, c]`) and the backward tower `B(u)` (line 525).
`c` is the conditioning input: the goal `g` on the goal-conditioned path, the task vector
`z` on the reward-sample path (line 872: `g = cond["_cvec_vgoal"] if self.z_on else
cond["value_goal"]`).

The critic's output is a distribution over the candidates,

    w_h(s, a, c)_m = softmax_m( F_h(s, a, c) . B(u_m) / tau ),          (line 885-886)

with temperature `tau = 1.0` (`measure_density_temp`, `impls/configs/critic/density_fb.yaml`)
and feature width `d = 50` (`measure_density_dim`).

The measure loss is a cross-entropy against a TD target on the simplex,

    t_i = (1 - gamma) * k_i + gamma * wbar(s'_i, a'_i, c),                (line 926)
    L   = - mean_i sum_m t_im log w_im,                                   (line 927)

where `gamma = 0.99`, `wbar` is the same softmax computed with the target (Polyak) copies
of both towers (line 917) at the actor's action `a'_i` drawn at the value goal (lines
896-911, `boot["next_action_vgoal"]`), and the immediate term `k_i` is one-hot at the row's
own next state, `k = I_B` (line 922, `measure_density_hard: true` is the default,
`density_fb.yaml`). With the hard target off, `k_i` is instead the kernel
`_delta_phi(s'_i, U)` (line 925); that variant is not used in any shipped cell.

### 0.2 Task vector from reward samples

`z_from_reward_samples(net, u_sc, r)` (lines 717-721) takes `M` candidate rows and their
reward values `r_1, ..., r_M` and returns `z` through `_z_index` (709-715), which is three
steps:

1. centered moment (lines 660-679):
   `m_c = (1/M) sum_j (B_xi(u_j) - mu_hat) (r_j - rbar)`,  `rbar = (1/M) sum_j r_j`,
   where `B_xi` is a slow EMA copy of the backward tower (`density_b_zenc`, rate
   `measure_density_z_enc_tau = 0.0005`, lines 780-783) and `mu_hat` its stored EMA feature
   mean (lines 791-796);
2. ridge solve (lines 681-697): `c = (C_hat + lambda I)^{-1} m_c`, with `C_hat` the stored
   EMA covariance of `B_xi` over the candidate marginal (line 794-796), eigenvalues floored
   at `1e-6 tr(C)/d`, `lambda = 0.01 tr(C)/d`;
3. normalization (lines 699-707): `z = sqrt(d) c / ||c||`, or the all-ones vector when
   `||c|| < 1e-6` (a constant reward carries no task).

So the shipped `z` is not the raw reward-weighted mean `E[r B]`. It is, up to scale, the
ridge-regularized least-squares coefficient of the reward on the centered features: the
minimizer of `E_rho[(r - rbar - c . (B_xi(u) - mu_hat))^2] + lambda ||c||^2`. This matters for
Proposition 1: `z` is (a scaled) coefficient of the L2 projection of `r` onto the span of
the centered features. FB's own moment `z = E_rho[r B]` coincides with this coefficient only
when `E_rho[B B^T] = I`, which is what FB's orthonormality loss enforces; this critic has no
such loss (`density_fb.yaml` header: "NO ortho"), so the whitening is done in the solve.

The base distribution for the samples is the next-state marginal of the dataset
(`INFER_COND_MARGINAL = "next_observations"`, line 479). At test time the harness calls
`infer_cond(net, obs, rewards)` (lines 1127-1156), which maps reward samples to `z` and
nothing else; the actor then rolls on `z` (`cond_vec` returns `cond["z"]`, lines 1031-1037).
A goal supplied by the environment is converted to `z` through the same reward-sample
API over a frozen pool of `M = 256` candidates drawn once from the first update's next states
(`map_eval_cond`, lines 816-862, pool written at line 804).

### 0.3 Actor: maximize the reward integrated over the candidates

The actor is FlowBC (`impls/actors/flowbc.py`). Its objective is
`-q_coeff * mean Q(s, pi(s, c)) / sg(mean |Q_s|)` plus behaviour-cloning terms
(`flowbc.py:74-75, 86`), where `Q` comes from the critic's `q_fn` (lines 986-1029):

    Q_h(s, a, c) = sum_m w_h(s, a, c)_m * kg_m,       Q = min(Q_1, Q_2)   (lines 1015-1021)

with the towers frozen (`stop_gradient`, lines 996-997), so the actor's gradient reaches
only the action.

What `kg_m` is depends on the path:

* Goal-conditioned path (`measure_density_z: false`; every shipped `density_fb` cell,
  including the running tdjtab campaign, `scripts/tdjtab_manifest.sh:87-105`):
  `kg = _delta_phi(g, U)` (line 1012), that is

      kg_m = softmax_m( -1/2 sum_b ||(u_m,b - g_b) / eps_b||^2 ),      (lines 354-373)

  a product of per-block Gaussian kernels in the object coordinates, normalized over the
  `M` candidates. Here `b` indexes the hit-spec blocks (cube-single: obs dims 19:22;
  cube-double: 19:22 and 28:31), `eps_b = 0.4` in observation units. OGBench multiplies
  block positions by 10 (`ogbench/manipspace/envs/cube_env.py:787`, `xyz_scaler = 10.0`),
  so `eps = 0.4` is 0.04 m, the environment's success radius
  (`cube_env.py:715`, `np.linalg.norm(obj_pos - tar_pos) <= 0.04`). The normalization
  over candidates is a positive constant per `(g, U)` that does not depend on the action, so
  the greedy policy is unchanged by it; it does rescale each row's gradient, which FlowBC's
  own `sg(mean |Q_s|)` normalizer then absorbs.

* Reward-sample path, `measure_density_z_actor_reward` in `{latent, latent_centered}`
  (committed, lines 956-974): `kg_m = B~(u_m) . z / sqrt(d)`, the linear proxy for the
  reward. `latent` uses the target backward tower, uncentered; `latent_centered` uses
  the slow index tower centered by `mu_hat`. This is FB's readout on the candidate set.

* Reward-sample path, `measure_density_z_actor_reward: kernel` (UNCOMMITTED working-tree
  edit dated 2026-09-08, lines 975-984, accepted by `build` at line 1174, default in
  `impls/configs/critic/density_fb_z.yaml`): `kg = _delta_phi(g, U)` on the rows whose `z`
  was encoded from their own actor goal (`cond["_z_hindsight"] == 1`, a fraction
  `1 - measure_density_z_rand_frac = 0.5` of the batch, lines 723-733), and the
  `latent_centered` proxy on the remaining rows, whose `z` is a fresh draw from the
  `sqrt(d)` sphere and has no goal. `F` and the actor still condition on `z` only.

### 0.4 What the method statement in the brief does and does not match

The brief describes: critic = softmax successor measure (matches 0.1); `z` from reward
samples (matches 0.2, with the correction that the solve is whitened); actor maximizes the
exact reward on the candidates rather than the linear proxy; training tasks = goal-reaching
kernels at the success radius.

* "Actor maximizes the exact reward" is true of the goal-conditioned path, where `F` and
  the actor condition on the raw goal, and of the uncommitted `kernel` mode on its hindsight
  rows. It is NOT true of the committed reward-sample cells (`density_fb_z`,
  `density_fb_z_state`): those train the actor on the linear proxy. No completed run exists
  in which the actor conditions on `z` and is trained on the exact reward; that is the
  pending cube-double run.
* "z equals FB's training-time family, z = B(future state)": FB's hindsight index is
  `z = project(B(g))` for a future state `g` (`impls/critics/fb.py:89-96`). Ours is the
  centered ridge coefficient of the kernel reward `r_g` on the candidates (0.2). They are
  the same direction only when `B` is whitened and `r_g` is near a Dirac at `g` on the
  support; with `eps` at the success radius and `M = 1024` candidates that is approximately
  but not exactly the case. Section 4 gives the exact statement.

---

## 1. Setup

An MDP has state space `S`, action space `A`, transition kernel `P(s' | s, a)`, discount
`gamma in (0, 1)`. A policy `pi(a | s)` is evaluated under a reward `r : S -> R` counted at
the next state (the convention of this code base, `density_fb.py:9`). The offline dataset
`D` induces a next-state marginal `rho` on `S`; every expectation `E_rho` and inner product
`<f, h>_rho = E_rho[f(u) h(u)]` below is with respect to it.

The successor measure of `pi` is

    m^pi(U | s, a) = sum_{t >= 0} gamma^t P(s_{t+1} in U | s_0 = s, a_0 = a, pi),

the expected discounted number of visits to the set `U`. It is a positive measure of total
mass `1 / (1 - gamma)`; its normalized form `p^pi = (1 - gamma) m^pi` is a probability
distribution and satisfies the Bellman equation

    p^pi(. | s, a) = (1 - gamma) P(. | s, a) + gamma E_{s' ~ P, a' ~ pi} p^pi(. | s', a'),

which is the TD target of line 926 with the one-hot immediate term standing in for
`P(. | s, a)` (one sample of it, the row's own `s'_i`). The value identity is

    Q^pi_r(s, a) = integral m^pi(du | s, a) r(u) = (1 / (1 - gamma)) integral p^pi(du | s, a) r(u).   (1)

(1) is the starting point of Dayan's successor representation (`dayan1993improving`), of
successor states and goal-dependent values (`blier2021learning`), and of FB
(`touati2021learning`, `touati2023does`). It says: once the critic knows the policy's
successor measure, the value under any reward is an integral, not a new learning problem.

When `p^pi(. | s, a)` has a density with respect to `rho`, write the density ratio
`q(u | s, a) = p^pi(u | s, a) / rho(u)`. FB models the ratio for the unnormalized
`m^pi / rho` as a rank-`d` bilinear form `F(s, a, z) . B(u)`; this critic models
`log q(u | s, a)` on a sampled support as `F(s, a, c) . B(u) / tau` up to an additive
constant per `(s, a)` (0.1). The additive constant is what the softmax removes.

---

## 2. Proposition 1: the linear readout reads the projected reward

Let `B : S -> R^d` be the backward features and let `Pi` denote the orthogonal projection
in `L2(rho)` onto `span(B) = { u -> b . B(u) : b in R^d }`. Write `G = E_rho[B B^T]` for the
feature Gram matrix, assumed invertible. FB's task vector is `z_r = E_rho[r(u) B(u)]`
(`touati2021learning`, Section 3; implemented as `r @ B(obs)` then projection to the
sphere, `impls/critics/fb.py:200-205`; RLDP and PSM use the same moment divided by `M`,
`impls/critics/rldp.py:459-462`, `impls/critics/psm.py:298-302`).

**Proposition 1 (FB reads the projected reward).** Fix `(s, a)` and a policy `pi`, and
suppose the learned bilinear form is the best rank-`d` fit of the ratio in `L2(rho)`, that
is `F(s, a) . B(.) = Pi (m^pi(. | s, a) / rho)`. Then

    F(s, a) . G^{-1} z_r = Q^pi_{Pi r}(s, a),                                            (2)

the value of `pi` under the projected reward `Pi r = B^T G^{-1} z_r`, and

    Q^pi_r(s, a) - F(s, a) . G^{-1} z_r = < (I - Pi)(m^pi / rho), (I - Pi) r >_rho.       (3)

In particular the linear readout equals the true value for every `(s, a)` if
`r in span(B)`; when `G = I` (FB's orthonormality constraint) the readout is FB's own
`F . z_r`.

*Proof.* Two lines. (2): `F . G^{-1} z_r = E_rho[(F . B)(u) (G^{-1} z_r) . B(u)]
= < Pi(m/rho), Pi r >_rho = < m/rho, Pi r >_rho = integral m (Pi r)`, using that `Pi` is
an orthogonal projection (`<Pi f, Pi h> = <f, Pi h>`) and `Pi r = B^T G^{-1} E_rho[r B]`.
(3): `Q^pi_r - integral m (Pi r) = < m/rho, (I - Pi) r >_rho = < (I - Pi)(m/rho), (I - Pi) r >_rho`,
again by orthogonality. If `r in span(B)` then `(I - Pi) r = 0`. QED.

Remarks.

* The condition `r in span(B)` is the successor-feature condition `r = phi . w` of
  `barreto2017successor` with `phi = B`. FB's own statement is Theorem 2 of
  `touati2021learning`: if `m^pi_z(du | s, a) = F(s, a, z) . B(u) rho(du)` holds exactly for
  all `z`, then `pi_{z_r}` is optimal for every `r`; with finite `d` the model is exact
  only on a `d`-dimensional family of rewards, and `touati2023does` (Section 3, the SF/FB
  comparison) states that SF-style methods are exact for rewards in the linear span of the
  features and that FB's job is to learn features whose span covers the rewards of
  interest. The section and theorem numbers are from memory; the PDFs are not on this
  machine, so they are UNVERIFIED and must be checked against the papers before use.
* The residual (3) is a product of two projection errors. It is small either when the
  reward is nearly linear in the features or when the ratio is; it is not small when the
  policy's successor measure concentrates exactly where the reward's nonlinearity lives,
  which is the case for a conjunctive reward at the goal configuration (section 4).
* Nothing in Proposition 1 involves the actor. It is a statement about what number the
  critic hands the actor for a given reward.

---

## 3. Proposition 2: the self-normalized readout reads any bounded reward

Let `u_1, ..., u_M` be i.i.d. from `rho`, and let `ell_m = log q(u_m | s, a) + kappa(s, a)`
for an arbitrary constant `kappa(s, a)` (the logits identify the log-ratio only up to a
per-`(s, a)` constant, and the training loss of 0.1 cannot pin it down either). Define

    w_m = softmax_m(ell)  =  q(u_m | s, a) / sum_{m'} q(u_{m'} | s, a),
    Qhat_M(s, a) = (1 / (1 - gamma)) sum_m w_m r(u_m).                                   (4)

**Proposition 2 (self-normalized importance-sampling readout).** Assume `r` is bounded,
`p^pi(. | s, a)` is absolutely continuous with respect to `rho` (`rho(u) > 0` wherever
`p^pi(u | s, a) > 0`), and `E_rho[q(u | s, a)^2] < infinity`. Then for every bounded `r`,
without any condition on `span(B)`:

(i) `Qhat_M(s, a) -> Q^pi_r(s, a)` almost surely as `M -> infinity` (consistency);

(ii) `E[Qhat_M] - Q^pi_r = O(1/M)` and `Var[Qhat_M] = O(1/M)`, with the asymptotic
variance `(1/M) E_rho[q^2 (r - (1 - gamma) Q^pi_r)^2] / (1 - gamma)^2`;

(iii) `Qhat_M` is invariant to `kappa(s, a)`.

*Proof.* (4) is the self-normalized importance-sampling estimator of `integral p^pi r`
with proposal `rho` and unnormalized weight `q`: numerator `(1/M) sum_m q(u_m) r(u_m) ->
E_rho[q r] = integral p^pi r`, denominator `(1/M) sum_m q(u_m) -> E_rho[q] = 1`, both by
the strong law (finite means follow from boundedness of `r` and `E q^2 < infinity`); the
ratio converges by the continuous-mapping theorem, which gives (i). (ii) is the standard
delta-method expansion of a ratio estimator; see Owen, *Monte Carlo theory, methods and
examples*, chapter 9 (self-normalized importance sampling), `owen2013mc`, and Kong 1992
for the effective-sample-size diagnostic `1 / sum_m w_m^2`, `kong1992ess`; the chapter
number is from memory and UNVERIFIED. (iii): adding `kappa` to every logit multiplies every
`q(u_m)` by `e^kappa`, which cancels in the ratio. QED.

What the proposition does and does not license.

* It licenses exact policy evaluation under any bounded reward, given the true ratio
  `q` up to a row constant. In the implementation `q` is learned (0.1), the candidates are
  the batch's own next states (`M = 1024`), and the same candidates serve both the measure
  fit and the readout. Consistency in the proposition is therefore conditional on the
  critic's fixed point being the true log-ratio up to the row constant; the code logs the
  effective sample size of `w` (`density_ess`, `density_ess_median`, lines 929-941) as the
  diagnostic (ii) calls for.
* (iii) is why a softmax critic can be trained on a restricted support, where only the
  normalized ratio is identifiable, and still evaluate rewards exactly. FB's linear readout
  `F . z` has no such invariance: it needs the unnormalized ratio, and its ortho loss is
  partly there to fix the scale. This is also why the reward-sample encoder of 0.2 is
  centered: `r -> r + beta` changes nothing in (4), so it should change nothing in `z`
  either.
* Same estimator, different reward: C-learning (`eysenbach-iclr21a`) and contrastive RL
  (`eysenbach-neurips22a`, `eysenbach2022contrastive`) train the same kind of classifier
  or InfoNCE softmax over sampled future states and read the goal-conditioned value as the
  log-ratio evaluated at the goal, that is (4) with `r` a Dirac at `g` and `M = 1` term
  retained. TD-InfoNCE / contrastive difference predictive coding (`zheng2024tdinfonce`)
  gives the temporal-difference form of that objective, which is the categorical Bellman
  target of line 926 with the reward still a Dirac. The present readout keeps their critic
  and replaces the Dirac by an arbitrary bounded `r` summed over the candidates; the
  implementation's `crl.py:72-75` reads the logit at the goal directly, which is the special
  case. Stabilizing contrastive RL (Zheng, Eysenbach, Walke, Yin, Fang, Salakhutdinov,
  Levine, ICLR 2024) is UNVERIFIED locally (no bib entry).
* Relation to universal successor features and GPI. Universal successor feature
  approximators (Borsa, Barreto, Quan, Mankowitz, Munos, van Hasselt, Silver, Schaul,
  ICLR 2019; UNVERIFIED locally, no bib entry) condition a policy on a task vector `w` and
  evaluate it under another task `w'` linearly, `psi(s, a, w) . w'`, then pick the best
  policy by generalized policy improvement (`barreto2017successor`; Barreto, Borsa, Quan,
  Schaul, Silver, Hessel, Mankowitz, Zidek, Munos, ICML 2018, UNVERIFIED locally). Ours is
  the same structure with two changes: the conditioning vector is the projection
  coefficient `z` of 0.2, and the number the actor ascends is the exact reward (4), not the
  linear form. No GPI over several `z` is implemented; each rollout uses one `z`.

---

## 4. Corollary: the two readouts agree exactly on `span(B)`, and the gap is the residual

**Corollary.** Under the assumptions of Propositions 1 and 2, with the same `F, B, rho`
and `q = F . B / tau` up to the row constant, and in the limit `M -> infinity`:
`Qhat(s, a) - F(s, a) . G^{-1} z_r = < (I - Pi)(m^pi/rho), (I - Pi) r >_rho`. If
`r in span(B)` the two readouts coincide.

This is (3) with the exact readout substituted for the true value, which (i) of
Proposition 2 permits. The corollary says that switching the actor's training signal from
the linear proxy to the exact readout changes nothing for rewards the features already
represent linearly, and changes exactly the projection residual otherwise.

Where the residual is measured to be large: the cube-double conjunction. The probe of
2026-09-08 (`/tmp/claude-1001/-home-adityamohan-git-Factored-FB-seam/678b32e6-6347-4028-b719-b8c7b248a0ea/scratchpad/report_probe.md`,
frozen 1M checkpoints, seed 0, `cube-double-play-v0`, `d = 50`, index tower
`density_b_zenc` centered by the stored `mu_hat`, 5-fold held-out `R2`) reports, on the
hit-spec `B`:

* 10k candidate pool, tasks 3/4/5: a linear fit in `B` of the both-cubes-placed indicator
  (the conjunction `r0 * r1`) has held-out `R2` .05 / .02 / .03, while the linear fits of
  the two single-cube marginals have `R2` .33/.43, .39/.46, .33/.02. The product of the two
  marginal linear fits, with no refit, has `R2` .24 / .18 / .04; a rank-2 bilinear refit
  `(a.B + a0)(b.B + b0) + c` has `R2` .40 / .20 / .24. On the raw six cube coordinates every
  linear or bilinear fit has `R2` .00 to .01.
* 100k pool, tasks 1..5: linear-in-`B` conjunction `R2` .02 .02 .03 .03 .07; product of
  marginals .10 .15 .21 .23 .07; rank-2 bilinear refit .16 .17 .26 .29 .42. The group
  contrast `one_frac` (mean prediction on exactly-one-cube-placed rows relative to the
  none-to-both range; 0.5 for a sum, 0 for a conjunction) is .40 .42 .41 .43 .34 for the
  linear index, .04 .05 .05 .05 .25 for the product of marginals, .06 for all five tasks
  for the bilinear refit; ranking AUC of both-placed against one-placed rows is .99 to 1.0
  for all three.

Read through the corollary: the per-cube rewards are close to `span(B)` (the marginals are
linearly readable), the conjunction is not (`R2 <= .07` in the linear family), and a
rank-2 product of two vectors in `span(B)` recovers the conjunction's shape. So the
residual `(I - Pi) r` is essentially the whole conjunctive reward, and Proposition 1 says
the linear readout hands the actor the sum of the two marginals, which is what the shipped
index does: the probe measures `cos(z_sum, z0 + z1) = .99 to 1.00` on four of five tasks
(.85 on task 5) for the campaign's own cube-double reward. One further fact from the same
report must be carried into any table: the campaign's Route-B relabel reward on
cube-double is `(number of cubes placed) - 2` shifted by 1, that is `r0 + r1 - 1`
(`env/ogbench.py:82-107`, `cube_relabel_reward`), which is a sum, not the conjunction; the
encoder is being handed the projected reward already.

---

## 5. Test-time interface, and what is privileged

Interface at test time (reward-sample route): a set of `M` states drawn from the dataset's
next-state marginal and their reward values go in (`infer_cond`, lines 1127-1156, which
calls `z_from_reward_samples`, line 1154); `z` comes out; the policy `pi(s, z)` is rolled
(`cond_vec`, lines 1031-1037). No goal, no object coordinate and no reward function is read
during the rollout; the exact readout (4) is used only inside the actor's training loss
(`flowbc.py:74-75`). This is the FB interface (`touati2021learning`, "zero-shot" task
specification from reward samples): the two methods differ only in how `z` is solved (0.2)
and in what the actor was trained to maximize (0.3). One harness caveat recorded at
`map_eval_cond`'s docstring (lines 816-841): a goal passed to `sample_actions` overrides a
previously inferred `z`, so the reward-sample route must be rolled with `goals=None`.

What is privileged at training time. The training-task family is goal reaching in the
object-coordinate blocks of the hit spec: the reward on the candidates, `_delta_phi(g, U)`,
is computed on `u_raw`, the raw object-coordinate columns (line 1043, `_kernel_rows`, lines
554-587, which refuses any array that is not an object-coordinate block). The design note
(`docs/density-fb-design.md:180`) and the pixels spec call this channel privileged. Two
knobs move the towers but not the kernel: `backward_input: state` feeds `B` the full 29-dim
observation instead of the block slice (lines 629-641), and `norm_obs` standardizes the
network inputs, explicitly not the kernel (`density_fb.yaml`, "THE HIT KERNEL IS NOT
STANDARDIZED"). So every shipped cell uses object coordinates in the training reward.

Two privilege-free replacements for the training family, neither implemented in
`density_fb.py`:

1. Goal reaching in dataset-normalized full-state coordinates:
   `r_g(u) = exp(-1/2 ||(u - g) / sigma||^2)` with `u, g` the full observation standardized
   by the running mean and variance the `norm_obs` path already maintains (lines 600-627,
   `_ObsNormState` at 249), and a single `sigma` in standardized units. This keeps the kernel
   form of 0.3 and removes the block spec; the price is that all 29 dimensions count,
   including the arm, which the object-coordinate kernel deliberately marginalizes out.
2. Goal reaching in the learned backward space: `r_g(u) = exp(-1/2 ||B(u) - B(g)||^2 / sigma^2)`,
   or the log-ratio at `g` (the contrastive-RL Dirac). HILP (`park-iclr24a`) already trains
   its policies on a latent distance of this kind: its value is `-||phi(s) - phi(g)||`
   (`impls/critics/hilp.py:75-80`), with `phi` learned by a goal-conditioned value loss, and
   the reward-sample task is solved by least squares on `phi(s')` (`hilp.py:408-418`). The
   same construction on `B` would make the training family a function of the features the
   index is solved in, which is exactly the family Proposition 1 covers.

What is not yet shown. (a) That an actor conditioned on a single linear `z` can act on a
conjunctive training signal: the `kernel` mode of 0.3 trains the actor on the exact
two-cube kernel on hindsight rows while `F` and the actor still condition on one `z`; the
probe's own reading (`report_probe.md`, "Is the critic linear in z?") is that a product
reward has no `z in R^50` that represents it through the linear encoder, so the policy
family may or may not contain a policy that solves the conjunction at the sum-of-marginals
index. That is the pending cube-double run; no number exists. (b) That Proposition 2's
consistency transfers to the learned, support-restricted critic at `M = 1024`. (c) That the
reward-sample `z` is a sufficient statistic for the policy when `r` is outside `span(B)`:
two rewards with the same projection get the same policy by construction, so the exact
readout improves the training signal for each `z` but cannot separate rewards the encoder
does not separate. (d) Any claim on inputs other than object coordinates: the two
privilege-free families above are proposals.

---

## 6. Related work: what each method reads out, and how

* FB (`touati2021learning`, `touati2023does`): rank-`d` bilinear model of the unnormalized
  successor-measure ratio, `m^pi_z / rho = F(s, a, z) . B(u)`, trained by a least-squares
  TD loss with an orthonormality term on `B`; reads `Q = F(s, a, z) . z` with
  `z = E_rho[r B]`, that is the projected reward of Proposition 1
  (`impls/critics/fb.py:171-172, 200-205`).
* HILP (`park-iclr24a`): learns `phi` so that `-||phi(s) - phi(g)||` is a goal-conditioned
  value, then successor features `psi` of `phi` under a `z`-conditioned policy; reads
  `Q = psi(s, a, z) . z` with `z` from a least-squares solve of `r` on `phi(s')`
  (`impls/critics/hilp.py:349-355, 408-418`); linear readout in the span of `phi`.
* PSM, proto successor measure (`agarwal-arxiv25a`): learns a basis of the affine space of
  all successor measures the environment permits, then a policy per basis coefficient;
  reads `Q = psi(s, a, z) . z` with `z = mean_j r_j phi(s'_j)` (`impls/critics/psm.py:235-242,
  298-302`); linear readout.
* RLDP, regularized latent dynamics prediction (`jajoo-iclr26a`): FB with a latent-dynamics
  regularizer on the representation; same `F . z` readout and the same reward moment
  divided by `M` (`impls/critics/rldp.py:427-428, 459-462`); linear readout.
* TD-JEPA (`bagatella-iclr26a`): a latent-predictive encoder trained by temporal-difference
  prediction in latent space, with a successor-feature head and the linear `z` readout of
  the SF family; the published OGBench table it reports is the comparison target of the
  tdjtab campaign (`docs/notes/2026-09-07-tdjepa-table-needed-runs.md`).
* Contrastive RL and C-learning (`eysenbach-iclr21a`, `eysenbach-neurips22a`,
  `zheng2024tdinfonce`): the same softmax or classifier density-ratio critic over sampled
  futures as 0.1, trained by Monte Carlo or TD contrastive objectives; reads the value of a
  goal as the ratio evaluated at the goal state, that is (4) with a Dirac reward
  (`impls/critics/crl.py:72-75`); no task vector, no integral over a reward.
* This method: the contrastive-RL critic (0.1), FB's reward-sample interface (0.2), and the
  self-normalized readout (4) of an arbitrary bounded reward as the actor's training
  signal (0.3). Proposition 1 locates the difference from the FB/HILP/PSM/RLDP/TD-JEPA
  family in the projection residual; Proposition 2 says the readout itself needs no span
  condition; section 5 says what still has to be shown for the policy family.

---

## Citation keys

Present in `PAPER/ICLR/notes/references.bib`: `blier2021learning`, `touati2021learning`,
`touati2023does`, `barreto2017successor`, `eysenbach2022contrastive`, `dayan1993improving`,
`park2024ogbench`, `park2023hiql`, `sutton2018reinforcement`.

Present in the shared bib directory `PAPER/ICLR/bib -> ../Multistep-FB/bib` (the symlink is
dangling in this worktree; the files live at `/home/adityamohan/git/Factored-FB/PAPER/Multistep-FB/bib/`):
`touati-neurips21a`, `touati-iclr23a`, `park-iclr24a` (HILP), `eysenbach-iclr21a`,
`eysenbach-neurips22a`, `zheng2024tdinfonce`, `agarwal-arxiv25a` (PSM, ICML 2025),
`jajoo-iclr26a` (RLDP), `bagatella-iclr26a` (TD-JEPA), `owen2013mc`, `kong1992ess`,
`barreto-neurips17a`, `mohan-rlbrew26a`.

Not found in any local bib (UNVERIFIED; full references given from memory, to be checked):

* Barreto, Borsa, Quan, Schaul, Silver, Hessel, Mankowitz, Zidek, Munos. Transfer in Deep
  Reinforcement Learning Using Successor Features and Generalised Policy Improvement.
  ICML 2018.
* Borsa, Barreto, Quan, Mankowitz, Munos, van Hasselt, Silver, Schaul. Universal Successor
  Features Approximators. ICLR 2019.
* Zheng, Eysenbach, Walke, Yin, Fang, Salakhutdinov, Levine. Stabilizing Contrastive RL:
  Techniques for Robotic Goal Reaching from Offline Data. ICLR 2024.

The keys `agarwal-icml25a`, `zheng-icml24a`, `zheng-iclr24a`, `sikchi-arxiv25a`,
`cetin-rlj24a`, `pirotta-iclr24a`, `li-iclr26a` are cited in `PAPER/ICLR/content/*.tex` but
have no entry in any local bib (`PAPER/ICLR/main.blg` reports them missing); do not reuse
them without adding entries.
