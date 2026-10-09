# HILP categorical-objective audit, 2026-09-11

No training or environment rollouts were launched. Small CPU forward/gradient
calculations found three implementation defects, now fixed in the assigned
files: temperature names, actor head reduction, and a zero actor-normalization
denominator. None is established as the cause of the recorded mixed results.

## What HILP and its categorical adaptation learn

The original feature learner fits the distance value
`V(s,g)=-sqrt(max(||phi(s)-phi(g)||²,1e-6))` to a discounted reaching target.
Reward is zero at the goal and minus one elsewhere; continuation stops at the
goal. There are two feature heads. The advantage uses the minimum next value
and mean target baseline, while each head has its own regression target.
These arithmetic choices match the local official source
`../baselines/HILP/hilp_zsrl/url_benchmark/agent/sf.py:154–175`.
The configured expectile is .5, making its weight constant. See
[hilp.py](../impls/critics/hilp.py:94).

The ordinary successor critic predicts discounted sums of the **centered next
state feature**, `phi(next)-running_mean`, for its task-conditioned policy.
Its actor uses `psi(s,a,z)·z`. Current builds accept state features and reject
feature differences/concatenation; there is no omitted telescoping subtraction
in this configuration. Reward inference solves least squares against centered
features before normalizing the task vector, so it already includes the
second-moment inverse. See [successor target](../impls/critics/hilp.py:291),
[readout](../impls/critics/hilp.py:349),
[reward inference](../impls/critics/hilp.py:434), and
[feature-type check](../impls/critics/hilp.py:489).

Density-HILP leaves that feature loss unchanged and detaches the features from
the categorical loss. Its candidate set is the batch's next states. For each
forward head, its logits are `psi(s,a,z)·phi(candidate)/temperature` and its
target is `(1-d)*one_hot(own_next_state) + d*target_softmax`, with the target
forward network evaluated at the policy's next action for the **same z**.
Softmax and cross-entropy both reduce over the candidate axis. It retains
separate target distributions for the two heads. The inherited factor .5
multiplies the head-averaged CE. See
[categorical loss](../impls/critics/hilp_density.py:93).

The new actor readout is correctly changed to the distribution-weighted reward
`sum(q(candidate)*phi(candidate)·z)`, rather than the old dot product. For a
fixed feature map and unrestricted forward vector, the CE gradient is
`(E_q[phi]-E_target[phi])/temperature`: moment matching is compatible with this
linear reward even when the full distribution cannot be represented exactly.
This does not establish an advantage for a shared neural forward network.
Unlike Density-FB, this adaptation cannot reshape its backward features using
CE. See [reward integration](../impls/critics/hilp_density.py:164).

Current HILP batches come through `HILPFutureDataset(FBDataset(...))`, which
sets terminal flags to zero. Thus terminal masking and the fallback to batch
goal-success masks do not explain the current campaign results. With constant
discount, the density value equals `(1-gamma)` times the ordinary discounted
value. This factor cancels in FlowBC's normalized value objective, above the
numerical floor. With unnormalized pure policy gradients it scales the raw
gradient; this alone is not a prediction of Adam's resulting parameter steps.

## Verified defects and bounded fixes

1. **Temperature overrides were ignored.** The current campaign YAML uses
   `measure_density_temp` and `measure_density_actor_temp`; the build method
   read only `temp` and `actor_temp`. An actual small build requesting .25/.5
   returned 1/None. The build now accepts both naming schemes and rejects
   conflicting double specifications. Existing 1/None configurations retain
   their numerical settings.
2. **Actor head selection was ignored.** Requesting the minimum returned the
   head mean. A controlled pair of integrated values `(+.9640,-.9640)` returned
   zero instead of `-.9640`. The actor now applies the inherited mean/min
   choice after reward integration. Its optional magnitude normalization
   centers the selected scalar value, consistent with Density-FB. Examined
   campaign configurations request mean, so this fix does not change those
   configurations' head choice.
3. **A constant centered value produced an undefined FlowBC update.** The
   production actor loss divided by zero, giving negative-infinite loss and
   gradient in the small reproduction. The detached denominator is now
   floored at `1e-6`; values above that threshold and the pure behavior-cloning
   branch retain their previous behavior. This guards the numerical case; it
   does not establish that a recorded run encountered it.

The deliberately different actor normalization was **not** removed. With the
same values `(10.1,12.1)` and the same cloning penalty, the production combined
actor gradient is `+.50991` using the original uncentered magnitude and `-.4`
using the centered magnitude. This is a substantive method difference beyond
critic loss, not a proof that either choice caused the empirical results.

## Other findings and limits

- In the prepatch object-coordinate variants, the feature network only saw
  object coordinates while the goal-equality predicate compared full states.
  Equal-object/different-arm examples had forced value near zero, reward minus
  one, loss `.99998`, and zero feature gradient. This affects both HILP variants;
  the separate audit repaired that predicate in `hilp.py` and verified it with
  six dedicated regressions. The direct pure-function probe in the archived
  diagnostic preserves the prepatch comparison.
- The optional exact-goal-reward variant trains a whitened absolute-goal task
  vector but inherits an evaluation hook using the relative goal-minus-current
  feature direction. At the goal, the latter becomes zero even though the
  training vector is nonzero. The inspected latest exact-reward runs use reward
  inference and bypass this hook, so this mismatch does not explain their
  reward-inference results. A correct goal-evaluation implementation needs a
  calibration pool/covariance; current HILP checkpoint statistics contain only
  mean/std. Merely replacing the relative feature with an unwhitened goal
  feature would remain inconsistent. This route was not changed.
- Online feature gradients from CE and target-network gradients were exactly
  zero, while the forward gradient was nonzero. Candidate-axis reductions,
  terminal one-hot targets, and target row sums passed independent checks.
  The target running mean's zero update rate is harmless to its softmax:
  changing a common feature mean only shifts every logit in a row equally.
- The original official implementation updates its feature target before the
  feature optimizer, while this stack updates targets after optimizers. The
  current campaign follows the latter convention, as does the local TD-JEPA
  source. Do not call all older HILP configurations an exact copy of one
  reference. Current YAML also differs from some historical resolved configs;
  the cross-cutting audit owns that provenance comparison.

## Reproduction and validation

[hilp_ce_audit_prepatch.json](hilp_ce_audit_prepatch.json) records the numerical
findings and prepatch source hashes. The two edited source files are archived
under [hilp_ce_prepatch_sources](hilp_ce_prepatch_sources/).
[hilp_ce_audit.py](hilp_ce_audit.py) is the prepatch diagnostic: its assertions
describing defects are expected to fail after the fixes. The maintained
regression tests below assert corrected behavior.

Run each file in a separate process with the existing cached dependencies:

```bash
JAX_PLATFORMS=cpu ../Factored-FB/.venv/bin/python /tmp/carry-drop-python.py analysis/hilp_ce_pytest.py tests/impls/test_hilp_density_contracts.py -q
JAX_PLATFORMS=cpu ../Factored-FB/.venv/bin/python /tmp/carry-drop-python.py analysis/hilp_ce_pytest.py tests/impls/test_flowbc_q_normalizer.py -q
JAX_PLATFORMS=cpu ../Factored-FB/.venv/bin/python /tmp/carry-drop-python.py analysis/hilp_ce_pytest.py tests/impls/test_hilp_density.py -q
JAX_PLATFORMS=cpu ../Factored-FB/.venv/bin/python /tmp/carry-drop-python.py analysis/hilp_ce_pytest.py tests/impls/test_flowbc_actor_static.py -q
```

The new tests failed on the intended branches before editing production:
five initial HILP failures, one additional selected-value normalization
failure, and two FlowBC failures. The corrected code passes **21 tests**:
10 new HILP contracts, 6 new FlowBC checks, 4 existing Density-HILP checks,
and the existing numerical FlowBC fixture. Red/green logs are
`hilp_ce_*_red.log`, `hilp_ce_*_green.log`, and `hilp_ce_existing_*test*.log`.
Only an existing third-party JAX deprecation warning was reported.

Independent review added a regression with action-dependent value spread:
`Q=(a,2a)` at `a=.1`. It checks loss `-2.97` and detached actor gradient
`-29.4`; differentiating through the scale would instead give gradient `.6`.
The six FlowBC checks pass with that additional case, without further production
changes or training runs.
