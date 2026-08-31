# The ortho arms clear the 500k gate: the dead regularizer was the whole story

**Gate as agreed:** compare the two live-regularizer arms against run 2 early, and
continue only if they look different from what we have run so far. They do, by an
order of magnitude, on the RLDP paper's own Figure 1 metric.

## The measurement

`scripts/rldp_feature_similarity.py`, walker, seed 1, 1024 sampled next-states.
`mean |cos|` between pairs of `B(s)`; `eff.rank` = participation ratio of B's
covariance spectrum, out of `z_dim = 512`.

| run | regularizer | step | mean\|cos\| | eff.rank |
|---|---|---|---|---|
| init (pre-port nets)   | --     | 0    | 0.3475 | 9.8   |
| init (ported nets)     | --     | 0    | 0.2956 | 6.6   |
| run 2 `rldp_official`  | inert  | 250k | 0.5557 | 15.2  |
| run 2 `rldp_official`  | inert  | 500k | 0.5927 | 19.2  |
| run 2 `rldp_official`  | inert  | 1M   | 0.5175 | 22.0  |
| run 2 `rldp_official`  | inert  | 2M   | 0.4270 | 25.8  |
| run 2 `rldp_official`  | inert  | 5M   | 0.4229 | 25.0  |
| `rldp_orthoa` (coef 1) | live   | 500k | 0.0482 | 181.0 |
| `rldp_orthob` (1e-4)   | live   | 500k | 0.0501 | 144.4 |

## What it says

Run 2 is the paper's own failure mode, quoted from Section 3: *"while the solutions
do not collapse, there is an increase in feature similarity over the course of
learning, which we refer to as a mild form of collapse."* Run 2's similarity **rises**
from 0.35 at init to 0.59 by 500k and never returns below its starting value. It ends
5M steps of training spanning 25 of 512 available directions.

That is the diagnosis for the action-blind critic. With B confined to a ~25-dim cone
whose pairwise cosines are all positive, `F(s,a,z)ᵀB(s')` cannot separate successor
distributions, so `M(s,a,s')` came out the same for real, random, and opposite actions
(589.1 / 591.9 / 586.0). The critic was not ignoring the action; it had no representation
left to express the difference in.

Both live arms invert the trend instead of merely slowing it: 0.30 -> 0.048, and 6.6 ->
145-181 effective directions.

## The network port is not the cause

The arms carry the network port as well as the live regularizer, so the two were
confounded. Measuring at initialization separates them: the ported nets start at
**lower** effective rank than the old ones (6.6 vs 9.8) and comparable similarity. The
port cannot explain a 20x increase in rank that runs opposite to its own initial effect.
What changed is that something is now pushing features apart during training.

## Standing conclusion

The official RLDP repository's regularizer contributes exactly zero gradient
(`orth_loss_diag` is constant `d` under `b.norm=True`, and the off-diagonal term enters
multiplied by zero). Runs 1 and 2 replicated that faithfully -- which is precisely why
they reproduced a collapse the paper claims to prevent. td_jepa's formulation, which the
seam already had before I introduced the "official" knobs, is the one that runs.

Arm A buys ~25% more effective rank than arm B (181 vs 144) at 3.4x the prediction loss
(sp_loss 1.78 vs 0.52). Which trade lands better is a returns question, not a
representation question, and phase 1 trains no actor -- so it is not answerable before
2M. Both arms continue.

## Caveat carried forward

Seed 1 only, one domain, one checkpoint per arm. The effect is large enough that seed
noise is not a plausible explanation for a 10x gap, but the arm A vs arm B ordering
is well inside the range where it could be.

## Operational addendum: the whole matrix is now on the mitigation

Two more cells (orthoa-s2 @ 500k, orthob-s0 @ 250k) hit
`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` in the same place -- the device-to-host copy
behind the logging line. Both were launched before the command-buffer mitigation went
into the launcher. The two cells restarted WITH it were still running. That is three
waves in which only unmitigated cells have died, which is enough to stop treating it as
coincidence.

All 8 cells now run with `XLA_FLAGS=--xla_gpu_enable_command_buffer=` and all resumed
from checkpoint rather than from zero.

Two launcher defects surfaced doing it:

- **`RESUME=1` is opt-in and the earlier restart did not set it.** Harmless at the time
  (those cells crashed at 65k and 120k, before the first 250k checkpoint existed, so
  there was nothing to resume) but it would have silently discarded 500k steps here.
- **`GPU_START` did not exist.** The device counter always starts at 0, so relaunching
  2 of 8 cells would have placed them on GPUs 0 and 1, on top of two healthy runs,
  instead of on the devices the dead cells had just freed.

The four healthy-but-unmitigated cells were rolled onto the fix at their 1M checkpoint
rather than immediately. Waiting is weakly dominant: they were at ~790k against a 750k
checkpoint, so rolling at once would have discarded 30-50k steps each, whereas a death
while waiting resumes from that same 750k. Nothing died; the roll cost ~10k steps per
cell instead of ~40k.

## End of phase 1 (2M): B is final, and the arm ordering resolves

Phase 2 freezes B, so these are the numbers that characterise each arm's representation.

| arm | seed | mean\|cos\| | mean cos | eff.rank |
|---|---|---|---|---|
| `rldp_orthoa` | 1 | 0.0481 | +0.0093 | 181.6 |
| `rldp_orthoa` | 3 | 0.0480 | +0.0092 | 180.5 |
| `rldp_orthob` | 1 | 0.0482 | +0.0155 | 151.4 |
| `rldp_orthob` | 2 | 0.0487 | +0.0166 | 151.9 |
| run 2 (inert)  | 1 | 0.4270 | +0.4270 | 25.8  |

**Correction to the 500k entry above.** I wrote there that the arm A vs arm B ordering
was "well inside the range where seed noise could explain it." With two seeds per arm
that is wrong and the ordering is resolved: within-arm spread is ~1 rank unit (181.6 vs
180.5; 151.4 vs 151.9) against a ~29 unit gap between arms. Arm A really does hold ~20%
more effective dimensions than arm B.

The representation also converges early -- 500k to 2M moves effective rank by less than
7 units in either arm -- so the 500k gate reading was not a transient.

What is still open is whether rank buys returns. Arm A pays 3.4x arm B's prediction loss
(sp_loss 1.78 vs 0.52) for its extra dimensions, and nothing measured so far says which
side of that trade lands better on task performance. Phase 2 trains the actor from 2M to
5M; that is the first point where the question is answerable, and mid-run evals are not
a substitute (see the standing rule about comparing evals across steps).

## Phase 2 at 2.5M: de-collapsing is necessary but NOT sufficient

`scripts/rldp_action_sensitivity.py`, walker, 1024 states, one fixed random z, matched
step 2.5M (500k of phase-2 training in every run). Sign tests over the same 1024 states;
chance is 0.500 and the standard error is about 0.016.

| run | eff.rank | ratio | P[Q_data > Q_rand] | P[Q_data > Q_neg] |
|---|---|---|---|---|
| run 2 (inert)   | 25.8  | 0.016 | 0.415 | 0.493 |
| `orthoa` s1     | 181.6 | 0.033 | 0.529 | **0.614** |
| `orthoa` s3     | 180.5 | 0.049 | 0.547 | **0.646** |
| `orthob` s1     | 151.4 | 0.033 | 0.427 | 0.469 |
| `orthob` s2     | 151.9 | 0.037 | 0.445 | 0.490 |

**Correction to what I claimed at the gate.** I wrote that the collapse "is the diagnosis
for the action-blind critic." That is not supported. Arm B de-collapses the representation
just as arm A does -- 152 effective dimensions against run 2's 26 -- and its critic still
ranks actions at chance, indistinguishable from the collapsed run. Removing the collapse
does not by itself produce a critic that can order actions. The collapse and the
action-blindness are both real, and the first does not explain the second.

**And the raw ratio was the wrong statistic.** In any control task Q varies far more
across states than across actions at a fixed state, because state value dominates the
advantage. A ratio of 0.03 is not evidence of anything on its own, so the original
"action/state sensitivity 0.061" reading proved less than I took it to prove. The sign
tests are scale-free and are what separates the arms.

**What does separate them.** Arm A is the only configuration whose critic beats chance,
and it is not marginal on the negated-action test: 0.614 and 0.646 against 0.500, roughly
7 and 9 standard errors, consistent across both seeds. Arm B sits at or below chance on
both tests in both seeds, exactly where run 2 sits.

So the ordering is arm A > arm B, on the one measurement that speaks to control, and it
agrees with the effective-rank ordering (181 vs 152) rather than with the prediction-loss
ordering (sp_loss 1.78 vs 0.52). The cheaper prediction loss buys nothing here.

This is the opposite of what the gradient-balance argument predicted. Arm A is the
configuration with the 1.09e4 ortho/SP gradient imbalance -- the one I had characterised
as pathological, with arm B at a balanced 1.09 chosen to resemble td_jepa's working 0.56.
The imbalanced arm is the one that works. Whatever the regularizer is doing for action
discrimination, gradient balance is the wrong lens for predicting it.

**Standing caveats.** Two seeds per arm, walker only, one fixed z, and 500k steps into a
3M-step phase 2. Above-chance ranking is a low bar -- it says the critic is not broken,
not that it is good. Returns at 5M remain the test that matters, and nothing here
forecloses arm B recovering with more actor training.

### 3M: neither arm moves, so "arm B may recover" looks wrong

Same measurement 500k steps later.

| run | 2.5M P[>rand] / P[>neg] | 3M P[>rand] / P[>neg] |
|---|---|---|
| `orthoa` s1 | 0.529 / 0.614 | 0.530 / 0.619 |
| `orthoa` s3 | 0.547 / 0.646 | 0.553 / 0.618 |
| `orthob` s2 | 0.445 / 0.490 | 0.419 / 0.465 |

I listed "arm B recovering with more actor training" as an open possibility. Half a
million further steps move neither arm in either direction, so on this metric both have
converged and that possibility is weakly refuted rather than still open. Arm A holds
above chance, arm B holds below it.

Returns at 5M remain the deciding measurement -- a critic that ranks actions above chance
is a precondition for control, not a demonstration of it.

## The run-2 baseline, scored at n=4 (walker, 5M, 50 episodes, 50k inference samples)

| task | run 2 (ours) | RLDP Table 1 | ours / published |
|---|---|---|---|
| stand | 365.4 +- 153.9 | 877.69 +- 45.03 | 0.42 |
| walk  | 106.9 +- 87.7  | 790.94 +- 67.55 | 0.14 |
| run   |  66.5 +- 32.7  | 324.85 +- 54.57 | 0.20 |
| flip  | 124.9 +- 90.3  | 491.64 +- 37.30 | 0.25 |

The spec's gate -- per-task mean within 1 sigma of the official cell on at least 3 of 4
tasks -- fails on all four. Run 2 reaches 14-42% of published.

This is the number the campaign has been chasing, and it now has a mechanism attached
rather than being an unexplained shortfall: run 2 replicates the official repository
including its inert regularizer, and consequently trains a backward map that ends at 26
of 512 effective dimensions with a critic that ranks actions at chance.

Note also the seed spread. Our stds (33-154) are two to four times the published ones
(37-68) on three of four tasks, so run 2 is not merely lower, it is much less stable
across seeds -- consistent with a critic providing little usable signal to the actor.

The ortho arms finish 5M shortly and get the identical treatment. That comparison is the
campaign's actual result: whether a live regularizer moves these four numbers.
