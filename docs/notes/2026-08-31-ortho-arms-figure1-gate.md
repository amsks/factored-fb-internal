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
