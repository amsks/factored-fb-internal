# ExoRL official-parity campaign, run 1 — FAILED the replication gate

Spec: `docs/specs/2026-08-28-exorl-official-parity-validation.md`.
Arms: `rldp_official` x {dmc-walker, dmc-cheetah} x seeds 0-3, 2M phase-1 + 3M phase-2.
Verdict: **vanilla RLDP gate FAILS**, root cause identified and measured, fix is one flag.

## 1. The result

Walker seed 0, all four tasks, paper protocol (z inferred from 50,000 physics-relabelled
transitions, projected to sqrt(d), goal path disabled), 10 episodes:

| task  | ours  | official RLDP (Table 1) | ratio |
|-------|-------|-------------------------|-------|
| stand | 207.5 | 877.69 +- 45.03         | 0.24  |
| walk  |  17.8 | 790.94 +- 67.55         | 0.02  |
| run   |  16.0 | 324.85 +- 54.57         | 0.05  |
| flip  |  88.4 | 491.64 +- 37.30         | 0.18  |

Gate (spec sec.4) needs >= 3 of 4 within 1 sigma and no task below 50% of an official mean
> 100. All four fail both conditions. Not a near miss.

FULL TABLE, all 8 cells x 4 tasks x 4 seeds, 10 episodes (`scripts/exorl_paper_eval.py`,
32/32 cells evaluated, 0 failures):

| domain  | task          | ours (n=4)      | official RLDP    | ratio |
|---------|---------------|-----------------|------------------|-------|
| walker  | stand         | 226.0 +- 149.4  | 877.69 +- 45.03  | 0.26  |
| walker  | walk          |  99.2 +- 105.6  | 790.94 +- 67.55  | 0.13  |
| walker  | run           |  38.5 +-  42.9  | 324.85 +- 54.57  | 0.12  |
| walker  | flip          |  90.2 +-  42.2  | 491.64 +- 37.30  | 0.18  |
| cheetah | run           |  16.6 +-  19.0  | 236.31 +- 20.75  | 0.07  |
| cheetah | run_backward  |  20.3 +-  20.7  | 322.08 +- 39.28  | 0.06  |
| cheetah | walk          | 104.0 +- 124.6  | 895.31 +- 49.84  | 0.12  |
| cheetah | walk_backward |  84.7 +- 112.0  | 984.76 +-  0.85  | 0.09  |

0 of 8 cells pass. Note the SPREAD as a second piece of evidence: std ~= mean on five of the
eight cells. That is the signature of policies driven by a representation that learned no
structure -- return dominated by initialization -- not of a method that is merely mistuned,
which would show a tighter spread at a lower mean. It is consistent with sec.2's measured
gradient imbalance and inconsistent with a hyperparameter being slightly off.

### Against the RANDOM-POLICY FLOOR (`scripts/exorl_random_floors.py`, 10 eps)

Raw returns have no natural zero, so the ratio to official understates how bad this is.
Measured floors for a uniform-random policy under the same eval harness:

| task                  | random floor    | run 1           | verdict                |
|-----------------------|-----------------|-----------------|------------------------|
| walker stand          | 126.16 +- 18.95 | 226.0 +- 149.4  | ~0.7 sigma above       |
| walker walk           |  33.92 +-  5.35 |  99.2 +- 105.6  | within noise           |
| walker run            |  22.65 +-  3.21 |  38.5 +-  42.9  | within noise           |
| walker flip           |  50.07 +-  5.94 |  90.2 +-  42.2  | ~1 sigma above         |
| cheetah run           |   3.68 +-  1.25 |  16.6 +-  19.0  | within noise           |
| cheetah run_backward  |  20.15 +-  2.02 |  20.3 +-  20.7  | EXACTLY AT FLOOR       |
| cheetah walk          |  18.39 +-  6.24 | 104.0 +- 124.6  | within noise           |
| cheetah walk_backward | 100.75 +- 10.08 |  84.7 +- 112.0  | BELOW FLOOR            |

**Run 1's policies are statistically indistinguishable from uniform random on 6 of 8
tasks, and below random on one.** That is a stronger and more useful statement than
"0.06-0.26x of official": it says the learned representation contributed nothing, which is
exactly what the 6.1e4 gradient imbalance predicts, and it rules out "trained but
under-tuned" as an explanation.

It also shows why the floors were worth measuring at all (spec sec.6 item 3): cheetah
walk_backward has a floor of 100.75, so reading its 84.7 as a weak-but-real result would
have had the sign backwards. Any future table cell should be quoted against its floor.

Episode count is 10, not the paper's 50. For an arm failing by 4-16x that is ample to
document the magnitude, and spec sec.6 records `eval-10` as a legitimate setting. Any arm
that PASSES must be re-scored at 50.

## 2. Root cause: phase 1 optimized whitening, not dynamics

`scripts/rldp_grad_balance.py`, real config, real data, batch 1024:

    ||grad SP||    =      0.588
    ||grad ortho|| = 35806.6
    ratio                6.1e4

The self-predictive term -- the only thing that teaches B the dynamics -- contributed about
one part in 60,000. B never learned to predict, so phase 2 regressed F onto a
dynamics-blind frozen basis.

The loss values did NOT show this. Phase 1 logged `sp ~ 1.66` against `ortho_off ~ 510`,
`ortho_diag = -512`, summing to `ortho_loss ~ -1.5` -- a near-cancellation that looks
harmless. Losses are not gradients; the two terms cancel in VALUE while ortho dominates in
GRADIENT. Any future phase-1 configuration should be checked with the script, not by
reading the loss column.

## 3. It was the config, not the transcription

Two knobs moved in the same direction at once, relative to the audited `rldp` arm:

| knob                            | effect on SP:ortho |
|---------------------------------|--------------------|
| `rldp_sp_official_scale: true`   | SP / z_dim  -> 102x weaker |
| `ortho_coef: 1.0` (seam: 0.01)   | ortho      -> 100x stronger |

~1e4 combined, which accounts for the measured 6e4 to within single-batch noise. The seam's
own `rldp` arm sits near 6:1 because it pairs td_jepa's SP scale with `ortho_coef: 0.01`.

And official has no such term at all: its ortho is diag-only
(`loss = ortho_coef * orth_loss_diag`, agent.py:270 -- the off-diagonal is computed for
logging and never enters the objective), and under the sqrt(d)-normalized B that diagonal
is the constant d. Measured: `ortho_diag = -512.0` exactly, every logged step of phase 1.
So OFFICIAL PHASE 1 IS SP-ONLY. Following the paper's Eq-7 form (`rldp_ortho_inert: false`,
per spec sec.1) introduced an objective the reference does not optimize.

## 4. Fix

`rldp_ortho_inert: true` in `rldp_official.yaml` and `rldp_density_official.yaml`. That
zeroes the ortho gradient and makes phase 1 SP-only = code-exact. It also makes
`ortho_coef` and `rldp_sp_official_scale` moot for B's learning, since SP becomes the only
term carrying gradient. This is the contingency spec sec.1 wrote the knob for. Requires a
full re-run: phase 1 changes, so ~12 h x 8 GPUs.

## 5. Two real bugs found and fixed en route (both two-phase-only)

Neither is visible in the joint td_jepa arm the audit validated, which is why the audit
passed them.

1. **Actor was DDPG+BC, official is pure TD3.** `actor_loss = -Q.mean()`, no BC term, action
   sampled at `actor_std=0.2` with `stddev_clip=0.3`, and no q-magnitude normalizer. The
   seam shipped `alpha=0.3`, noiseless modes, and `-q.mean()/sg(|q|.mean())`. New
   `actor.q_normalize` knob (default true) + launcher overrides. Commit `5e7d82e`.

2. **FB target read the TARGET backward tower; official reads the ONLINE one.** Official's
   `_target_backward_map` is phase-1-only (the SP target in `update_mr`). With B frozen in
   phase 2 the seam compared two different frozen bases forever, and the linear
   `-mean(diag(diff))` term grew F unboundedly. A/B from ONE shared 2M checkpoint, this flag
   the only difference, 50k steps:

       target tower   fb_loss   256 -> 30,650   (doubling every ~5k steps)
       online tower   fb_loss   227 ->    437   (flat)

   New `rldp_fb_target_online_b` (default false). Commit `683afd6`.

## 6. Also worth knowing

- **In-training evals for this campaign are meaningless.** The launcher did not pass
  `--infer_cond_episodes`, so eval ran on `eval_cond0` = `{"z": zeros}`, i.e. Q = F.0 = 0 and
  a goal-blind policy. No warning fired because the warning lives inside the
  `infer_cond_episodes > 0` block. Only the final `scripts/exorl_paper_eval.py` numbers mean
  anything. `launch_exorl_paper.sh` passes these flags; `launch_exorl_official.sh` should too.
- **`bc_log_prob ~ -2e9`** in phase 2. Multiplied by `alpha=0` so it is inert, but it says the
  actor drifts astronomically far from the behavior data -- inherent to official's
  unanchored TD3 objective. First place to look if sparse cheetah tasks come in low.
- **The launcher cannot restart a single cell**: its session-exists guard exits 1, so a
  scraped dry-run returns empty. Add a WINDOW_ONLY path.
- **The existing compendium RLDP numbers may be mildly whitening-skewed too** (the seam's
  6:1 balance is far better than this campaign's 6e4, but it is not 0). Worth a check before
  those numbers are quoted again.
