# Spec 12 — L2 zero-shot reward test on the λ=0 phase-2 checkpoints

**Date:** 2026-07-28. **Status:** pre-registered before results.
**Depends on:** ledger §9.9 (corrected L1) — the λ=0 phase-2 object + N̂ head is
~set-level calibrated (1.1–1.2× at hw 0.8/1.6, Spearman 0.72–0.79), licensing
this test. **Approach 1** (minimal pipeline) per the 2026-07-28 brainstorm:
ONE readout — FB's standard `reward_inference` — no ridge, no N̂ correction,
no GPI. Eval-only; checkpoints `exp_spec10/torchfb/p2_s4{2,3}` untouched.

## Question

Does the trained object deliver the *capability* a successor measure promises:
given a reward it never trained on, (A) predict the return it will obtain, and
(B) act to earn it — with no retraining?

## Reward families (all functions of cube xyz, obs units; trained kernel ε=0.4)

8 instances each, centers = random dataset cube positions (seed 0):

| family | r(c) | why |
|---|---|---|
| `ball0.4` | 1[‖c−c*‖≤0.4] | trained radius — reference/sanity |
| `ball0.8`, `ball1.6` | same, ε∈{0.8,1.6} | off the trained kernel width |
| `smooth0.4`, `smooth0.8` | exp(−‖c−c*‖²/2σ²) | no threshold anywhere — strongest break |
| `twozone` | 1[ball A]∨1[ball B], ε=0.4, ‖A−B‖>1.6 | additivity — only a measure composes |

## Readout (the one under test)

`z_r = model.reward_inference(next_obs=10k dataset rows, reward=r(cube(rows)))`
— exactly the evaluator's path. Actor and F consume the **projected** z_r.
Return prediction is recorded twice:
- `pred_proj` = ⟨F(s0,z_r,a0), z_r⟩ — what the critic thinks with the actor's z
  (projection destroys reward scale ⇒ correlation meaningful, scale not);
- `pred_raw` = ⟨F(s0,z_r,a0), z_raw⟩ with z_raw = (1/n)Σ r_i B_i unprojected —
  the ∫r dM̃ estimate this readout implies. **Known, pre-registered bias:** no
  1/N̂ density correction (Approach 1 excludes it), so `pred_raw` estimates
  ∫r·N·dM, overweighting data-dense regions; and no-ortho training voids the
  Cov(B)=I assumption behind E[r·B] inference. If A fails on scale/correlation,
  the readout formula — not the measure — is the first suspect (the L1fix
  evidence already localizes calibration to the M̃/N̂ object).

## Stage A — prediction (per reward × 12 teleport starts)

MC truth: 2 rollouts × H=200 of π(·,z_r), discounted r summed from the state
after a0 (γ⁰ first, training convention), **env.reset() before every rollout**
(audit §9.1). Metrics per family: Pearson+Spearman of pred vs MC across
(instance×start) pairs; scale = mean(pred_raw)/mean(MC).

**Gate (per family): Spearman ≥ 0.8 AND scale within 2×.** Miss on scale only
⇒ readout-scale defect (fixable, measure intact). Miss on correlation ⇒ the
representation does not support zero-shot transfer at that family.

## Stage B — control (per reward × 10 episodes, standard env resets)

Drives, same checkpoint: **z_r** (the test) vs **z_goal** = project_z(B(row
nearest c*)) (ceiling; for twozone: better single center) vs **z_rand** =
sample_z (floor). Score: discounted return under r (computed from obs cube pos
each step) + undiscounted time-in-region. Read per family:
`(ret(z_r) − ret(z_rand)) / (ret(z_goal) − ret(z_rand))` — fraction of the
floor→ceiling gap recovered. **Pre-registered read: ≥0.7 = zero-shot transfer
works for that family; ≤0.3 = readout fails to steer; between = partial.**
For `smooth*`/`twozone` the "ceiling" is itself a heuristic (goal-z at center
ignores reward shape/second zone) — if z_r *beats* z_goal there, that is
evidence the measure integrates reward shape, and is the headline outcome to
look for.

## Execution

Driver `scripts/rescuing_fb/torchfb_zeroshot_l2.py` (`--stage A|B`), results
`results/spec10_torchfb/L2_{stage}_{run}.json`, logs `/dev/shm/spec10_logs/`
(backup to `_logs_backup/` after), tmux `l2zs`, 4 jobs (2 seeds × 2 stages) on
idle GPUs. Ledger §9 gets the outcome; `L1fix` conventions reused (CUBE_OBS
19:22, seed 0, dataset from cfg.data_path).
