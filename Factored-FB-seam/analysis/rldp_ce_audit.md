# RLDP categorical adaptation audit, 2026-09-11

Scope: local source, current paper math, resolved cube-single/cube-double configs,
and CPU value/gradient checks. No training, sweep, or checkpoint evaluation.
The current comparison is joint RLDP versus joint Density-RLDP.

## Core mathematical behavior

The original estimator fits a linear successor density ratio
\(m_h=F_h^\top B\). It uses the observed next state as the positive pair and
other minibatch next states in its squared Bellman residual. The bootstrap uses
the same task and its actor's next action. See impls/critics/rldp.py:369–404.

The adaptation instead normalizes logits over the same candidate pool and fits
\[
p_{hi}=\operatorname{softmax}_j(F_{hi}^\top\operatorname{sg}B_j/\tau),\qquad
t_{hi}=(1-d_i)e_i+d_i\operatorname{sg}\bar p_{hi}.
\]
Targets are headwise; CE averages heads and rows. The positive pair is included
in softmax and CE. The off-diagonal mask serves only orthogonality. Uniformly
sampled candidate rows define the empirical reference measure; there is no
missing importance factor at this interface.

Backward features are deliberately detached in both successor losses.
Self-predictive rollout and orthogonality train them instead
(rldp.py:306–353; rldp_density.py:125–129). CE therefore changes the forward
estimator and left encoder, and cannot directly choose the useful backward
logit features from the deterministic-path construction. This is an intentional
estimator retrofit, not end-to-end Density-FB geometry.

The actor integrates \(B(u)^\top z\) under each predicted distribution.
Default reduction is the mean, following RLDP; optional pessimism acts after
scalar integration. F supplies logits, so \(F^\top z\) would be an invalid
categorical value readout.

## Confirmed defects, repaired

Four real-network regression tests failed before the patch for the intended
assertions. They cover three ignored inherited options:

| Defect | Prepatch evidence | Repair |
|---|---|---|
| Requested online backward target ignored. | Parent loss changed 17.684826 → 13.896115; density remained 14.022649. | Honor online/target backward selection. |
| Diagonal-only orthogonality ignored. | Representation-phase backward-gradient error norm 38.666274; predictor error zero. | Use inherited orthogonality helper. |
| Actor pessimism ignored. | Q unchanged despite head disagreement 0.989600. | Apply mean minus uncertainty after reward integration. |

The centered second return uses the selected policy value, matching current
Density-FB at impls/critics/density_fb.py:1127–1134. For two heads and coefficient
one-half, the test checks the minimum integrated Q and its centered value.

Missing phase-specific early returns were not called optimizer leakage:
inherited stage_groups already freezes the appropriate parameter groups.
No phase cleanup, backward-gradient change, task change, or terminal change was
made.

These options are inactive in the current resolved joint runs. The four
exp/cube-{single,double}-play-v0_{rldp,rldp_density}_flowbc_seed_0/config.json
files match batch 256, discount .99, dimension 50, two heads, spherical B,
orthogonality coefficient .01, and equal random/hindsight task mixing. The
official density preset enables the defective options, but the shared audit
found no local resolved density run using them. These repairs therefore do not
establish a cause for the reported current mixed results.

## Verification

- Four regression tests failed before editing production.
- All 59 tests across the new contract file, existing Density-RLDP tests, and
  parent RLDP tests passed. One existing dependency deprecation warning.
- Independent NumPy and actual CE both equal 2.1076750921530683; maximum target
  row-sum error is \(2.22\times10^{-16}\). Terminal rows are one-hot.
- All target-parameter gradients are zero. CE gradient norms: forward 1.527833,
  left encoder 1.243469, backward zero, predictor zero.
- Default joint backward/predictor gradients exactly match the parent.
- Fixed-array postpatch checks show default losses, CE/representation gradient
  checks, and actor-normalization values exactly unchanged.

Artifacts: analysis/rldp_ce_audit.py, its prepatch and after-fix JSON files,
analysis/rldp_ce_audit_contracts_red.txt and _green.txt, and
tests/impls/test_rldp_density_contracts.py.

Prepatch density SHA256:
8d3306a57e1523eabab5238b7bd343af4c74c984131e1c9542db6e9af5f374ac.
Functional patch SHA256:
6b4c9811830fecb50cdd5e943903f6413398327779efc765b1ec7bd3c80e38ed.
Source after a documentation-only covariance qualification:
79d2723da09f79b8494c27592ded78dda37e676e1bc02cc921f2ee3fd9644e3a.

## Additional reward-inference defect and repair

**The old reward projection needed a whitening assumption.** Parent inference sets
\(z=\mathbb E_\rho[B r]\) (rldp.py:459–462), while the categorical actor rewards
candidates by \(B^\top z\). With \(C=\mathbb E_\rho[BB^\top]\), this equals the
linear reward projection only when C is identity, or for special task directions.
For fixed B, population measure-regression stationarity gives
\(C(F-\gamma\mathbb E F')=\mathbb E B(s')\), so ordinary F implicitly contains
\(C^{-1}\). Direct categorical integration loses that compensation.
The audit includes spherical B with \(C=\operatorname{diag}(1.6,.4)\): a
representable true reward gap is +1.4142, while the inherited density proxy gap is
−1.1314. This proves the projection mismatch, not causation in a trained run.

After verifying the density actor's reward convention, a separate authorized
fix now overrides Density-RLDP reward inference: for latent rewards it solves
least squares against B with JAX's default rank cutoff, then applies the existing
task-vector normalization. This aligns requested rewards with the existing
linear reward family. Parent RLDP and all training losses, task sampling, and
goal-conditioned mappings remain unchanged. Goal-kernel mode delegates exactly
to parent moment encoding because its trained goal embedding is not a latent
linear-reward coefficient.

Tests in tests/impls/test_rldp_density_inference.py first produced five expected
failures and ten passes. They exercise the actual categorical action readout
with nonwhite spherical features, exact white-feature equivalence, minimum-norm
rank-deficient solutions, zero rewards and zero features, exact goal-kernel
delegation, and a real backward network's representable reward. Logs are
analysis/rldp_ce_audit_inference_red.txt and _green.txt.
All 37 tests across the inference file, inherited-option contracts, and existing
Density-RLDP suite passed in 59.38 seconds, with one existing dependency warning.
Independent source/test review passed. The parent RLDP suite had already passed
in the earlier 59-test run and was not changed by this inference override.

This fix intentionally changes inferred conditions for nonwhite latent features.
Historical reward-inference results require reevaluation. It also does not make
the same numerical training index mean the same reward in ordinary and density
RLDP when C differs from identity.

Final source SHA256:
ec2214d925545e42e6134d1e08bccf3f1d93ea817f72e39ca362d88715d5b7ab.

## Remaining mathematical issues

**Actor weighting changes too.** FlowBC divides ordinary RLDP by mean absolute
head values, but Density-RLDP by the spread of its selected scalar values
(flowbc.py:74–87). This changes the balance with behavior cloning. Omitting
\(1/(1-\gamma)\) in normalized Q preserves rankings and cancels homogeneous
FlowBC scaling; it changes step scale for unnormalized pure DDPG. RLDP does not
use a feature-difference reward, so no telescoping subtraction belongs here.

**Terminal algebra is not an episodic-equivalence proof.** Full one-hot terminal
targets treat the terminal outcome as absorbing. They are not a universally
rescaled finite-horizon terminal reward. Current WindowedDataset emits zero
terminals (datasets.py:592), so this boundary is inactive in these runs.

**Paper descriptions need backbone-specific qualifications.** Method.tex:176
says minimum head reduction, whereas default Density-RLDP uses the mean.
Method.tex:158–165 describes same-trajectory future kernel rewards encoded into
tasks, whereas RLDP uses embeddings of permuted minibatch next states
(rldp.py:449–456). Its optional goal-kernel variant changes both tasks and reward.

No current joint CE sign, axis, stop-gradient, or simplex defect was found.
The strongest remaining hypotheses concern separately learned backward geometry
and actor weighting; reward-inference reevaluation is needed after the projection
repair.
