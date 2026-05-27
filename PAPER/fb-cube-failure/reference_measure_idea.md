# Coverage-balanced FB for OGBench cube placement

## Main claim

Forward-backward representations fail on `cube-single-play` placement because the learned backward embedding \(B\) is optimized under the behavior-data reference measure \(\rho\). In the offline play dataset, the terminal placement region is present but low-mass. As a result, the finite-rank FB representation allocates too little resolution to the cube configurations that matter most at evaluation.

The proposed fix is to train FB under a goal-space, coverage-balanced reference measure \(\rho'_G\), so that the backward embedding is accurate over the cube-placement manifold rather than merely where the behavior policy spent most of its time.

This is the FB analogue of hindsight relabeling:

\[
\text{HER relabels goals.}
\]

\[
\text{FB should relabel or reweight the goal-side reference measure.}
\]

The goal is not to post-process \(B(g)\) after training. The goal is to change the measure under which \(B\), the FB factorization, and reward inference are learned.

---

## 1. FB setup

For a latent-indexed policy family \(\{\pi_z\}\), FB approximates the policy-conditioned successor measure as a density with respect to a reference measure \(\rho\):

\[
M^{\pi_z}(ds^+ \mid s,a)
\approx
F(s,a,z)^\top B(s^+)\,\rho(ds^+).
\]

Equivalently,

\[
F(s,a,z)^\top B(s^+)
\approx
\frac{dM^{\pi_z}}{d\rho}(s^+ \mid s,a).
\]

For a reward \(r\), the task vector is

\[
z_R
=
\mathbb{E}_{s^+ \sim \rho}
\left[
r(s^+)B(s^+)
\right],
\]

and the induced value is

\[
Q_z(s,a)
=
F(s,a,z)^\top z.
\]

For a point-goal reward at \(g\), this reduces, up to scaling, to

\[
z_g \approx B(g).
\]

Thus, in zero-shot goal reaching, the goal enters FB primarily through the backward embedding \(B(g)\). If \(B\) does not resolve the relevant local geometry near the goal, then the induced policy cannot reliably distinguish “placed” from “almost placed,” even if the forward representation contains enough information.

---

## 2. Goal-space FB for manipulation

For cube manipulation, the relevant reward does not depend on the full simulator state. It depends mostly on cube configuration relative to the target.

So we should write FB over a goal feature space

\[
g = \phi(s),
\]

for example

\[
g =
(x_{\text{cube}}, y_{\text{cube}}, z_{\text{cube}}, \text{orientation / task-relevant pose features}).
\]

Then the successor measure is over future cube configurations:

\[
M^{\pi_z}(dg \mid s,a)
\approx
F(s,a,z)^\top B(g)\,\rho_G(dg),
\]

where \(\rho_G\) is the empirical marginal over cube configurations induced by the offline play dataset.

This is the right object for placement: \(F\) can still condition on the full state-action pair, but \(B\) only needs to represent the future variables on which the reward depends.

---

## 3. Empirical diagnosis

### Observation 1: the failure is terminal, not reach/grasp

The trajectories are not failing because the cube is unreachable or because the data are out of support. The failures concentrate near the final placement phase.

Evidence:

| Phase | FB | GCIQL | Interpretation |
|---|---:|---:|---|
| Approach | 98% | 99% | Essentially solved |
| Transport | 83% | 94% | Gap appears after grasp/lift |
| Near-goal transport | 80% | 94% | Gap is local to terminal placement |

All trajectories grasp and lift. The issue is not reaching or grasping; it is fine terminal control.

---

### Observation 2: \(B\) under-resolves terminal cube geometry

The backward embedding does not linearly encode distance-to-goal well enough:

\[
R^2(B \to d_{\text{cube}\to\text{goal}})
=
0.20
\quad \text{on state}.
\]

The pixel version is much better:

\[
R^2(B \to d_{\text{cube}\to\text{goal}})
=
0.76.
\]

Using an MLP readout does not substantially improve over a linear readout, suggesting that the relevant geometry is not merely nonlinearly hidden in \(B\). It is mostly absent from \(B\).

By contrast, the forward featurizer retains more of the same information:

\[
R^2(F\text{-features} \to d_{\text{cube}\to\text{goal}})
=
0.53.
\]

So the failure is not simply that the state representation discards cube geometry. The bottleneck is specifically the goal-side representation \(B(g)\).

---

### Observation 3: goal-conditioned value methods retain the geometry

On the same data, GCIQL/GCIVL value features retain local goal geometry:

\[
R^2(\phi(s,g) \to d_{\text{cube}\to\text{goal}})
\approx
0.93
\quad \text{on state},
\]

and

\[
0.95\text{--}0.98
\quad \text{on pixels}.
\]

This is an important positive control. The information is present in the dataset and can be learned by a goal-conditioned value network. FB loses it because of how the goal is compressed into \(B(g)\).

---

### Observation 4: the bottleneck is \(z=B(g)\)

The privileged-\(z\) experiment isolates the failure.

Freeze \(F\), then compare:

1. the standard goal vector

\[
z = B(g),
\]

against

2. a free optimized vector

\[
z^\star = \arg\max_z \text{alignment}(Q_z, -d_{\text{cube}\to g}).
\]

Near the goal:

\[
\rho_S(V, -d)
=
0.04
\quad \text{with } z=B(g),
\]

but

\[
\rho_S(V, -d)
=
0.20
\quad \text{with } z=z^\star.
\]

That is roughly a \(5\times\) improvement from changing only the goal vector. The forward path is not perfect, but it has more usable structure than \(B(g)\) exposes.

This points to \(B\) as the critical bottleneck.

---

### Observation 5: this is not a data absence problem

Near-goal states exist in the offline dataset:

\[
\sim 35{,}000
\]

states are within 5 cm of the goal.

But they represent only about

\[
3.5\%
\]

of the reference measure \(\rho\).

So the problem is not that the terminal region is absent. The problem is that it is low-mass under the measure used to train the finite-rank representation.

---

## 4. Mechanistic interpretation

FB learns a finite-rank approximation to successor densities under an inner product induced by \(\rho\). Therefore, representation quality is not uniformly distributed over the state or goal space. It is allocated according to the measure used in the loss.

With the default offline choice,

\[
\rho = \text{behavior-data marginal},
\]

the representation is most accurate where the play policy spends most of its time.

For cube placement, that is misaligned with the deployment objective. Evaluation requires precision near arbitrary target cube placements, especially terminal configurations. But these configurations are low-mass under the play distribution.

Therefore, \(B\) becomes accurate enough for coarse reach/grasp/transport structure but too coarse for terminal placement. The result is a collapse of near-goal configurations:

\[
\text{placed cube}
\approx_B
\text{near-miss cube}.
\]

Since FB uses

\[
z_g = B(g)
\]

as the goal vector, this collapse is fatal. The policy receives a goal vector that does not sharply specify the terminal condition.

The key insight is:

> FB is not failing because the task is unreachable. It is failing because the finite-rank goal representation is optimized under the wrong reference measure.

---

## 5. FB-HER: relabel the measure, not just the goal

HER reuses a failed trajectory by pretending that an achieved future state was the intended goal.

For FB, the analogous move is not simply to relabel the scalar reward. The object that controls goal resolution is the reference measure on the backward side.

So the FB-HER analogue is:

\[
\rho_G
\longrightarrow
\rho'_G,
\]

where \(\rho'_G\) places more mass on goal-relevant cube configurations.

Then FB is trained as

\[
M^{\pi_z}(dg \mid s,a)
\approx
F(s,a,z)^\top B(g)\,\rho'_G(dg).
\]

Equivalently,

\[
F(s,a,z)^\top B(g)
\approx
\frac{dM^{\pi_z}}{d\rho'_G}(g \mid s,a).
\]

This changes the finite-rank projection problem from

\[
L^2(\rho_G)
\]

to

\[
L^2(\rho'_G).
\]

That is the central intervention.

---

## 6. Constructing a coverage-balanced reference measure

We want \(\rho'_G\) to improve coverage of the goal manifold without leaking evaluation goals.

Let \(\hat{\rho}_G(g)\) be an offline density estimate over cube configurations, fit only on the play dataset. Then define

\[
\rho'_G(dg)
=
\frac{w(g)\rho_G(dg)}
{\mathbb{E}_{g\sim\rho_G}[w(g)]},
\]

with

\[
w(g)
=
\operatorname{clip}
\left(
(\hat{\rho}_G(g)+\epsilon)^{-\alpha},
w_{\min},
w_{\max}
\right).
\]

Here

\[
\alpha \in [0,1]
\]

controls the strength of balancing.

Special cases:

| \(\alpha\) | Meaning |
|---:|---|
| \(0\) | Baseline FB; no reweighting |
| \(0.25\) to \(0.5\) | Mild coverage balancing |
| \(0.75\) to \(1.0\) | Strong inverse-density balancing |

The weights should be mean-normalized:

\[
\mathbb{E}_{g\sim\rho_G}[w(g)] = 1.
\]

The density model can initially be simple:

- kNN density in cube-pose space
- discretized cube-position bins
- VAE or flow density over cube configurations
- stratified sampling over cube pose bins

The point is not perfect density estimation. The point is to stop the representation from spending almost all of its resolution on the behavior-dense bulk.

---

## 7. Consistency requirement

To preserve the FB interpretation, all goal-side \(\rho\)-dependent terms must use the same reference measure \(\rho'_G\).

That means:

### Measure-matching target samples

When the FB loss samples future/target goals \(g^+\), sample them from

\[
g^+ \sim \rho'_G
\]

or equivalently sample from \(\rho_G\) with importance weights \(w(g^+)\).

### Backward orthonormality

Replace

\[
\mathbb{E}_{g\sim\rho_G}[B(g)B(g)^\top] = I
\]

with

\[
\mathbb{E}_{g\sim\rho'_G}[B(g)B(g)^\top] = I.
\]

Otherwise, the whitening term will fight the reweighted FB objective.

### Reward inference

Use

\[
z_R
=
\mathbb{E}_{g\sim\rho'_G}
\left[
r(g)B(g)
\right].
\]

For point goals, this still reduces to using \(B(g)\), up to scaling. For dense rewards, this change matters.

### Practical note on the diagonal term

In the FB Bellman loss, the diagonal or positive term evaluates the immediate successor contribution at the current achieved goal. If transition samples remain drawn from the original replay distribution, this term should be handled consistently with that sampling distribution.

The safe implementation rule is:

> Use \(\rho'_G\) everywhere the loss draws or integrates over the goal-side variable \(g\). If you also reweight the initial transition distribution, then reweight the diagonal term consistently as well. Do not reweight only one piece of the objective.

The failure mode to avoid is a mixed objective where the quadratic target samples, the \(B\)-whitening, and reward inference correspond to different measures. That no longer represents a clean FB factorization with respect to a single \(\rho'_G\).

### Implementation

The realization of this consistency rule is specified in `docs/superpowers/specs/2026-05-25-coverage-balanced-fb.md`. With per-sample weights \(w_j=w(g_j)\), mean-normalized so \(\frac1n\sum_j w_j=1\), each goal-side term is reweighted on the goal axis:

- **Successor loss.** Weight the off-diagonal (negative) entry \((i,j)\) by the goal column \(w_j\); weight the diagonal (positive) entry \(i\) by \(w_i\). Because \(\overline{w}=1\), the normalization constants are unchanged — only the distribution over goals shifts.
- **Orthonormality.** Realize \(\mathbb{E}_{g\sim\rho'_G}[B(g)B(g)^\top]=I\) by whitening the *scaled* embedding \(\tilde B_i=\sqrt{w_i}\,B_i\): the Gram matrix \(\tilde B\tilde B^\top\) gives \(\tfrac1n\sum_i \tilde B_i\tilde B_i^\top=\tfrac1n\sum_i w_i B_iB_i^\top\), whose minimum of the metamotivo-style loss is \(\mathbb{E}_{\rho'_G}[BB^\top]=I\).
- **Reward-inference covariance.** Use \(\operatorname{cov}=\tfrac1n\sum_i w_i B_iB_i^\top\) in the implicit-reward solve (only active when \(q_{\text{loss}}>0\); a no-op for point-goal eval).

> **This is exactly the lever the diagnosis demands.** The whitening \(\mathbb{E}_\rho[BB^\top]=I\) is computed under the bulk-dominated \(\rho_G\), and the frozen proxy shows it isotropizes \(B\) over the dense region (effective rank 47.6/50). A reweight of *only* the successor diagonal (the first, discarded attempt) leaves the whitening untouched and is precisely the mixed objective warned against above; the orthonormality term must be reweighted too.

---

## 8. Hindsight sampling as an additional mechanism

Coverage balancing changes the marginal goal measure. Hindsight sampling can be added on top.

For a trajectory segment, sample achieved future cube configurations

\[
g_{t+k} = \phi(s_{t+k}),
\qquad k \ge 0.
\]

There are two ways to use these.

### 8.1 Use hindsight goals to build \(\rho'_G\)

This is the clean version.

Pool achieved future cube configurations from the offline trajectories, estimate their density, and construct \(\rho'_G\) from the resulting coverage-balanced marginal.

This preserves the interpretation

\[
F^\top B \approx \frac{dM}{d\rho'_G}.
\]

### 8.2 Use hindsight goals for \(z\)-sampling

During FB training, sample

\[
z = B(g_{t+k})
\]

for achieved future goals \(g_{t+k}\).

This trains the \(z\)-conditioned policy/value family on goal vectors that correspond to reachable achieved configurations. It is closer to ordinary HER.

However, this is secondary. The diagnostics suggest that the main bottleneck is not merely which \(z\)'s are sampled. It is that the map

\[
g \mapsto B(g)
\]

does not resolve terminal geometry. So hindsight-\(z\) sampling should be treated as an ablation, not the main fix.

### 8.3 Avoid conditional target-goal sampling unless corrected

If we sample target goals from

\[
q(dg \mid s_t,a_t)
\]

instead of a global marginal \(\rho'_G(dg)\), then the loss no longer corresponds to a single reference measure unless we importance-correct it.

This may still be useful as a heuristic, but it should be separated from the principled \(\rho'_G\) experiment.

---

## 9. Proposed experiments

Run the following ablations.

This document carries **two distinct levers** that must be ablated separately:

1. **Coverage-balanced \(\rho'_G\)** (§§5–7) — changes the *measure* the finite-rank projection is taken under, leaving \(B\)'s input space (full state) unchanged. This is what the spec implements and what is being run first.
2. **Goal-space \(B\)** (§2) — changes \(B\)'s *input/output space* to the cube-config features \(g=\phi(s)\). This is an architectural change, **not** the current \(B\) (today \(B\) ingests the full observation; the measured \(R^2(B\to d)=0.20\) is on that full-state \(B\)). It needs its own row and must not be conflated with the reweight.

| Variant | Change | What it tests |
|---|---|---|
| Baseline FB + FlowBC | Current setup (full-state \(B\), \(\rho_G\)) | Reference point |
| Hindsight-\(z\) only | Sample \(z=B(g_{t+k})\), keep \(\rho_G\) unchanged | Whether \(z\)-sampling coverage is the main issue |
| **Coverage-balanced \(\rho'_G\) (consistent)** | Reweight successor + \(B\)-orthonormality + reward-cov under \(\rho'_G\) (§7 Implementation) — **not** diagonal-only | Whether \(B\) improves when goal-side resolution is reallocated |
| Goal-space \(B\) only | \(B\) over \(g=\phi(s)\) (cube config), keep \(\rho_G\) | Whether restricting \(B\)'s input to reward-relevant variables alone helps — isolates lever 2 from lever 1 |
| Coverage-balanced \(\rho'_G\) + goal-space \(B\) | Both levers | Combined structural + measure change |
| Coverage-balanced \(\rho'_G\) + hindsight-\(z\) | Combine measure reweight and \(z\)-sampling | Likely best version of the measure-side fixes |
| Oracle / privileged \(z^\star\) | Freeze \(F\), optimize \(z\) directly | Upper bound on how much better goal vectors can be (currently \(z^\star=0.20\ll\) forward featurizer \(0.53\): a forward-path ceiling the \(B\)-side levers may not clear) |

Suggested sweep:

\[
\alpha \in \{0, 0.25, 0.5, 0.75, 1.0\}.
\]

Suggested clipping:

\[
w_{\min} \in \{0.1, 0.2\},
\qquad
w_{\max} \in \{5, 10, 20\}.
\]

Start with mild balancing. Strong inverse-density weighting can make the covariance ill-conditioned and may sacrifice too much performance in the behavior-dense bulk.

---

## 10. Expected effects

If the diagnosis is correct, coverage-balanced FB should produce the following changes.

### Prediction 1: \(B\) should encode terminal distance better

\[
R^2(B \to d_{\text{cube}\to\text{goal}})
\]

should increase, especially near the placement region.

This should be measured globally and also conditionally near the goal.

---

### Prediction 2: nearest neighbors in \(B\)-space should become pose-consistent

For a target cube pose \(g\), nearest neighbors under

\[
\|B(g_i)-B(g)\|
\]

should correspond to nearby cube configurations in physical pose space.

Currently, near-miss and placed states appear too collapsed.

---

### Prediction 3: the privileged-\(z\) gap should shrink

The gap between

\[
z=B(g)
\]

and

\[
z=z^\star
\]

should shrink.

In particular, near-goal value-distance alignment should improve:

\[
\rho_S(V_{B(g)}, -d)
\uparrow.
\]

---

### Prediction 4: value should rise along hindsight trajectories

Along coherent hindsight trajectories ending near the goal, FB values should become more monotone:

\[
Q_{B(g)}(s_t,a_t)
\]

should increase as the cube approaches the goal.

The current failure mode is that FB values remain too flat near the terminal region.

---

### Prediction 5: placement improves more than approach/grasp

The main success improvement should occur in the transport-near-goal phase.

Approach and grasp are already close to saturated, so they should not change much. The target metric is terminal placement success.

---

### Falsification / pre-registered kill criterion

If the consistent reweight raises \(R^2(B\to d)\) and closes the \(z=B(g)\to z^\star\) gap (Predictions 1, 3) **but terminal placement success does not move** (Prediction 5 fails), the headline claim is falsified: the binding constraint is not the goal measure but the forward-path / bilinear readout (§11, "Remaining risk"). In that case, also check whether \(z^\star\) *itself* rose after retraining \(F\) under \(\rho'_G\) — if \(F\) improved but \(B(g)\) still cannot exploit it, the next move is rank/architecture (§11 next steps), not further measure tuning. Reporting this negative outcome is part of the contribution.

---

## 11. What this does and does not fix

### What it fixes

Coverage-balanced FB can fix a finite-rank allocation problem:

\[
\text{too much resolution in behavior-dense regions}
\]

and

\[
\text{too little resolution near rare but important goal configurations}.
\]

It should sharpen \(B(g)\), making the goal vector more informative for placement.

### What it does not fix

It cannot solve genuine support mismatch.

The new measure must satisfy

\[
\rho'_G \ll \rho_G.
\]

That is, we can upweight cube configurations that exist in the offline data, but we cannot invent unsupported placements.

If evaluation goals lie outside the offline support, this method cannot make the policy reliable there.

### Remaining risk

Even with a sharper \(B\), the bilinear value form

\[
Q_z(s,a)=F(s,a,z)^\top z
\]

may remain limiting. The privileged-\(z\) result suggests that a better \(z\) helps, but does not fully reach the forward-feature ceiling. If coverage-balanced \(B\) improves but still plateaus, the next issue may be the single-vector bilinear readout itself.

Possible next steps would then be:

- richer goal readout \(z=h(B(g))\)
- multi-vector or nonlinear reward embeddings
- additional terminal contrastive losses on \(B(g)\)
- fine-tuning \(Q_z\) near terminal states

But those should come after testing the cleaner measure-level intervention.

---

## 12. One-sentence version

FB rations representation accuracy by its reference measure; in OGBench cube placement, the behavior-data measure underweights terminal cube configurations, so \(B(g)\) becomes too coarse to specify placement. The fix is to train FB in a coverage-balanced goal-space measure \(\rho'_G\), optionally enriched with hindsight achieved goals, so the finite-rank representation allocates resolution where evaluation goals actually live.