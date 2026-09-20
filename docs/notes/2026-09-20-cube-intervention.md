# Intervention: does correcting FB's action values prevent its manipulation failures? (cube-single, 2026-09-20)

Figures: `PAPER/ICLR/figures/gap_preservation/fig_intervention_by_phase.{pdf,png}` (FB, success
after the intervention) and `fig_intervention_held_by_phase.{pdf,png}` (FB, cube still held
after 20 steps, in-hand and carrying states); the same two for Density-FB with suffix
`_density`. 3.2 x 2.3 in, saved at the exact figure size, phase colours and fonts of
`fig_alignment_main.py`. Tables: `intervention_by_phase.csv` (pooled over seeds),
`intervention_by_phase_per_seed.csv`, `intervention_paired.csv` / `_per_seed.csv` (within-state
differences between conditions), `intervention_per_state.csv` (one row per state, condition),
`intervention_selection.csv` (which candidate each rule picked, its selection-time values),
`intervention_picks.json`, `intervention_qcheck.json`. Raw rollouts:
`analysis/failure_figs/cube-single-play-v0/intervention/rollouts_seed{0,1,2}_{approach,inhand,carrying}.csv`
(one row per rollout) and `qcheck_seed*_*.csv`. Script:
`scripts/probes/cube_bottleneck/intervention.py` (stages `select`, `roll`, `aggregate`;
`--smoke` = 2 carrying states x 2 rollouts, both critics, 82 s on one GPU), launcher
`scripts/probes/cube_bottleneck/launch_intervention.sh`. Tests:
`tests/probes/test_cube_bottleneck_intervention.py` (7 tests: selection rule and tie handling,
selection table, per-rollout metrics, per-phase aggregation pooled and per seed, paired table).

## Why

The decision-by-phase figure (2026-09-19) is an association: at carrying states FB's critic
orders the true best above the true worst candidate at chance, and FB's rollouts fail there.
This probe intervenes on the one thing the critic supplies to the actor, the values used to
choose the next action, and holds everything else fixed: the same restored state, the same 12
candidate actions, the same frozen policy for every later step. If the failures come from the
values, replacing them with correct ones at a single step should already move the outcome.

## Protocol (exact)

- Checkpoints: tdjtab 1M, `fb_tdj` (exp/) and `density_fb_z` (exp/tdjtab/), seeds 0/1/2. Task
  vector z: the reward-inference route (10,000 relabelled transitions per task, fixed NumPy
  seed per task, shift 1.0), the same z as the action probe and the separation probe.
- States: the three 64-state banks per seed of the failure figures (approach, in-hand,
  carrying), restored with `failure_figs.restore_full` (env reset to the recorded task, cube
  goal mocap restored, qpos/qvel set, `mj_forward` + `mj_rnePostConstraint`, two settle steps
  through the env controller with the gripper commanded closed for held states, open
  otherwise). 192 states per phase per critic pooled over seeds.
- Candidates: the 12 of the action probe, regenerated with its keys (each policy's proposal at
  `PRNGKey(4242 + 7 i + seed)`, secure = gripper +1, release = gripper -1, four Gaussian
  perturbations of each proposal, sigma 0.3, clipped, NumPy rng `[20260912, seed, i, bank]`).
  Identity check: the critic's readout of the regenerated candidates against the cached
  `q_pred`, relative to the state's value range: FB max 0.004, Density-FB max 0.034 (mean
  0.003; its readout is a softmax over a 1024-row pool, more sensitive to device numerics);
  no ordering-relevant deviation (`intervention_qcheck.json`, per state in `qcheck_*.csv`).
- Selection rules for the FIRST action (all later steps: the critic's own frozen flow policy,
  temperature 0, the same inferred z):
  (a) actor: the actor's own proposal (`fb_policy` for FB, `density_policy` for Density-FB);
  (b) critic: argmax over the 12 candidates of the critic's own actor readout q(s, a, z)
      (FB: F(s,a,z)^T z, min over heads; Density-FB: the measure-weighted task reward);
  (c) rollout: argmax of the cached Monte-Carlo return of the action probe: candidate executed
      once, then K = 10 continuations of the SAME policy, 100 steps,
      return = sum_t 0.99^t 1[cube within 0.04 m of its goal] -- the reward and discount the
      critic is trained on. Nothing was re-estimated for the selection; the values are read
      from `actions_long_seed*_1000000_{app,car}.csv`. Ties at the maximum keep the actor's
      own proposal if it is among the tied (MC ties, all 12 candidates at return 0: FB 10,
      7, 6 states at approach / in hand / carrying; Density-FB 4 / 0 / 0).
- Evaluation: fresh rollouts, keys `fold_in(PRNGKey(31337 + 13 i + seed), candidate index)`,
  disjoint from the selection rollouts (`PRNGKey(99 + 13 i + seed)`) and the separation
  probe (`777 + ...`); K_eval = 10 per state and chosen candidate; horizon H = 100 env steps
  (the forced action is step 0, then 99 policy steps -- the probes' horizon; the restore
  resets the episode clock, so no truncation occurs). Conditions that pick the same candidate
  share the same 10 rollouts (same action, same policy, same distribution). The flow actor
  draws its noise from the rng at every step regardless of the temperature argument, so the
  10 rollouts are distinct samples.
- Metrics per rollout: success at any step (cube within 0.04 m of its goal), success at the
  final step, discounted return, held over the first 20 steps and over all 100 (held =
  `phases.transport_mask`: cube more than 3 cm above the table, gripper opening > 0.5, the
  vendored definition). Aggregation: mean within a state over its rollouts, then over states;
  s.e. over states; pooled over three seeds (n = 192 states, 1920 rollouts per cell).
- Compute: 9 jobs (seed x bank) on GPUs 0-2, 19-24 min each (MuJoCo-bound), 25 min wall.

## Headline: success after the intervention (any step within 100; mean +- s.e. over states)

| critic | phase | (a) actor | (b) critic | (c) rollout | (c) - (b), same-state | (c) - (a) |
|---|---|---|---|---|---|---|
| FB | approach | 0.355 +- 0.024 | 0.355 +- 0.024 | 0.383 +- 0.023 | +0.028 +- 0.014 | +0.028 +- 0.012 |
| FB | in hand (maintain) | 0.376 +- 0.024 | 0.385 +- 0.025 | 0.442 +- 0.025 | +0.056 +- 0.015 | +0.066 +- 0.015 |
| FB | carrying (transport) | 0.587 +- 0.029 | 0.521 +- 0.028 | 0.716 +- 0.026 | **+0.195 +- 0.022** | +0.129 +- 0.019 |
| Density-FB | approach | 0.852 +- 0.022 | 0.793 +- 0.025 | 0.926 +- 0.015 | +0.133 +- 0.020 | +0.074 +- 0.017 |
| Density-FB | in hand | 1.000 +- 0.000 | 0.955 +- 0.011 | 1.000 +- 0.000 | +0.045 +- 0.011 | 0.000 |
| Density-FB | carrying | 1.000 +- 0.000 | 0.994 +- 0.003 | 0.999 +- 0.001 | +0.006 +- 0.003 | -0.001 +- 0.001 |

Final-step success equals any-step success in every cell (a rollout that reaches the goal
stays there). Per seed (0/1/2), FB carrying: (a) 0.50 / 0.58 / 0.68, (b) 0.41 / 0.53 / 0.63,
(c) 0.67 / 0.69 / 0.79; FB in hand: (a) 0.24 / 0.41 / 0.47, (b) 0.27 / 0.41 / 0.47,
(c) 0.33 / 0.45 / 0.55. The ordering (c) > (a) >= (b) at carrying holds in every seed.

Discounted return (mean +- s.e.): FB carrying (a) 25.7 +- 1.7, (b) 22.0 +- 1.5, (c) 37.3 +- 1.7;
FB in hand 9.1 / 11.3 / 14.5; FB approach 6.3 / 6.3 / 6.7. Density-FB carrying 52.5 / 51.9 /
54.9; in hand 40.5 / 36.5 / 44.1; approach 20.6 / 18.7 / 23.0.

## Held rate (cube held at every one of the first 20 steps; whole horizon in brackets)

| critic | phase | (a) actor | (b) critic | (c) rollout |
|---|---|---|---|---|
| FB | in hand | 0.199 +- 0.022 [0.019] | 0.506 +- 0.031 [0.041] | 0.515 +- 0.030 [0.059] |
| FB | carrying | 0.184 +- 0.020 [0.114] | 0.195 +- 0.021 [0.093] | 0.131 +- 0.017 [0.053] |
| Density-FB | in hand | 0.722 +- 0.023 [0.433] | 0.534 +- 0.030 [0.348] | 0.754 +- 0.022 [0.486] |
| Density-FB | carrying | 0.545 +- 0.018 [0.511] | 0.477 +- 0.021 [0.457] | 0.320 +- 0.023 [0.306] |

Caveat: "not held" conflates dropping the cube with placing it at the goal; at carrying the
rollout rule picks `release` at 65 of 192 FB states and 64 of 192 Density-FB states (the cube
is already over the goal and the best first action is to let go), which is why (c) has the
LOWEST held rate at carrying while having the highest success. At in-hand states the release
pick is rare (FB 11, Density-FB 4 of 192) and the held rate reads as intended: FB's own action
keeps the cube for 20 steps in only 20 % of in-hand states; both the critic-selected and the
rollout-selected first action raise that to 51 %.

## Agreement between the rules and the value of what they pick

Fraction of states where the rule picks the actor's own proposal ((b) = (a); (c) = (a)) and
where (b) = (c):

| critic | phase | (b) = (a) | (c) = (a) | (b) = (c) |
|---|---|---|---|---|
| FB | approach | 0.000 | 0.141 | 0.062 |
| FB | in hand | 0.005 | 0.089 | 0.167 |
| FB | carrying | 0.026 | 0.057 | 0.120 |
| Density-FB | approach | 0.000 | 0.057 | 0.068 |
| Density-FB | in hand | 0.047 | 0.042 | 0.203 |
| Density-FB | carrying | 0.031 | 0.031 | 0.271 |

Mean selection-time MC return of the chosen candidate (fresh evaluation return in brackets):
FB carrying (a) 25.5 [25.7], (b) 21.9 [22.0], (c) 38.4 [37.3]; FB in hand (a) 9.0 [9.1],
(b) 11.4 [11.3], (c) 18.5 [14.5]; FB approach (a) 6.9 [6.3], (b) 5.9 [6.3], (c) 10.5 [6.7].
Density-FB carrying (a) 52.4 [52.5], (b) 51.9 [51.9], (c) 55.0 [54.9]; in hand 40.7 [40.5],
36.6 [36.5], 44.4 [44.1]; approach 20.6 [20.6], 18.7 [18.7], 24.0 [23.0]. The fresh return of
the rollout-selected candidate is below its selection-time estimate by 1-4 return units
(the optimism of taking the maximum of 12 noisy estimates); the success rates above are from
the fresh rollouts only.

What the rules pick (`intervention_picks.json`): FB's critic picks `secure` (close the gripper)
at 121 of 192 approach states, where the gripper is nowhere near the cube; at in-hand and
carrying it picks a perturbation of one of the two policies' proposals almost everywhere
(secure 16 / 11, release 1 / 2). The rollout rule picks a perturbation of a proposal at
most in-hand states and `release` at a third of the carrying states.

## Reading

The key comparison is (b) against (c): identical candidate set, identical continuation policy,
identical restored state, only the numbers used to rank the 12 candidates differ. For FB at
carrying, ranking by the critic's values gives 0.52 success, ranking by Monte-Carlo returns
under the same policy gives 0.72: a within-state difference of +0.20 +- 0.02 (+0.22 on the 169
states where the two rules chose different candidates), consistent in every seed. At in-hand
the difference is +0.06 +- 0.02, at approach +0.03 +- 0.01. So a single correctly-valued
action at a carrying state raises FB's success by a fifth even though the next 99 actions come
from the same policy; the failures FB shows after grasping are, to that extent, failures of
the values it ranks actions with, not of the policy's ability to execute. The critic's own
ranking is no better than the actor's proposal at in-hand (0.385 vs 0.376) and worse at
carrying (0.52 vs 0.59, -0.07 +- 0.02): FB's argmax over the candidates is a worse first
action than what its actor already proposes, which is the decision-by-phase chance-level
ordering turned into an outcome.

Density-FB is at the ceiling at in-hand and carrying under every rule (0.99-1.00), so no
intervention can show anything there; at approach its critic-selected action is worse than
its actor's (0.79 vs 0.85) and the rollout-selected better (0.93), so its critic's ranking
of the first action is also imperfect where the decision matters least.

Caveats. (i) The rollout rule is an oracle over 12 candidates, not a learnable critic; it
bounds what correcting one action's values can buy, it does not say a critic could reach it.
(ii) The selection estimate (10 rollouts) and the evaluation (10 fresh rollouts) use the same
estimator, so (c) carries the optimism of a maximum; it is 1-4 return units and does not
touch the (b) vs (c) success comparison, which uses fresh rollouts for both. (iii) The
candidate set is dominated by perturbations of the two policies' proposals; the gain from
(c) is what is available within that set. (iv) Held rates at carrying are not a drop rate
(release at the goal counts as not held). (v) Each seed's states are its own bank; pooling
treats 192 states as the unit.

## Reproduce

    # selection table from the cached action probe (no simulator)
    MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python scripts/probes/cube_bottleneck/intervention.py select
    # fresh evaluation rollouts, 9 jobs on GPUs 0-2 (25 min wall)
    scripts/probes/cube_bottleneck/launch_intervention.sh
    # tables + figures
    MPLCONFIGDIR=/tmp/mpl JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python scripts/probes/cube_bottleneck/intervention.py aggregate
    # smoke (2 carrying states x 2 rollouts, both critics, then aggregate into smoke_int/)
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python scripts/probes/cube_bottleneck/intervention.py --smoke
    # tests
    /dev/shm/.venv-jax/bin/python -m pytest -q tests/probes/test_cube_bottleneck_intervention.py
