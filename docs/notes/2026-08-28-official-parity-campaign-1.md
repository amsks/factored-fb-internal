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
