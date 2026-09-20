# Does the critic pick the right action where the choice matters? (cube-single, 2026-09-19)

Paper figure (the only one): `PAPER/ICLR/figures/gap_preservation/fig_decision_by_phase.{pdf,png}`.
Tables behind it: `fig_decision_by_phase.csv` (per phase, pooled over seeds),
`fig_decision_by_phase_per_seed.csv`, `gap_preservation_per_state.csv` (one row per restored
state and model). Script: `scripts/probes/cube_bottleneck/gap_preservation.py` (`--smoke` =
5 states x 4 actions x 1 rollout end to end, 63 s on CPU; default = aggregate the cached action
probe). Test: `tests/probes/test_cube_bottleneck_gap_preservation.py` (7 tests: gaps, ratio,
binning, s.e., Wilson interval, per-phase aggregation). `fig_gap_preservation.{pdf,png}` in the
same directory is an internal diagnostic (calibrated ratio / share / scatter) and is not
referenced anywhere.

## Protocol

The per-candidate readings are the cached action probe of
`scripts/probes/cube_bottleneck/failure_figs.py` (stage `actionprobe`, written 2026-09-12; the
same rows Fig. 5's sign metric reduces in `sign_agreement.py`). Nothing was re-rolled.

- Checkpoints: tdjtab 1M, cube-single-play-v0, `fb_tdj` (exp/) and `density_fb_z`
  (exp/tdjtab/), seeds 0, 1, 2.
- Task vector z: the reward-inference route of `impls/main.py` eval_relabel, one z per task
  from 10,000 relabelled training transitions (reward 1 if the next state's cube is within
  0.04 m of the task goal, shift 1.0), fixed NumPy seed per task so both models read the same
  rows; the env goal is never shown to the policy.
- States: the three 64-state banks of the failure figures per seed (approach, in-hand,
  carrying; 576 per model), drawn from the pooled rollouts of both models and restored with the
  verified restore (`restore.py`: qpos/qvel set, `mj_forward` + `mj_rnePostConstraint`, two
  settle steps through the env's own controller, gripper closed for held states, fidelity gate
  on contact, lift and drift < 4 cm).
- Candidate actions (12, identical for both models): each policy's own action, secure
  (gripper +1), release (gripper -1), 4 Gaussian perturbations (sigma 0.3, clipped) of each
  policy action.
- True value of a candidate: restore, execute the candidate once, then K = 10 continuations of
  THAT model's own frozen policy (temperature 0, inferred z) for 100 steps; return =
  sum_t gamma^t 1[cube within 0.04 m of its goal], gamma = 0.99, averaged over continuations.
  a+ / a- = highest / lowest return; true action gap Delta_MC(s) = Q_MC(s, a+) - Q_MC(s, a-).
  This is the state importance of Torrey & Taylor (2013), max minus min action value,
  estimated by Monte Carlo over the candidate set.
- Critic decision: correct at s iff q(s, a+, z) > q(s, a-, z), q the critic's own actor
  readout (FB: F(s,a,z)^T z; Density-FB: the measure-weighted task reward the actor ascends).
  No scale enters: the test is the order of two values. States whose 12 candidates all return
  0 (no decision to get right) are excluded: FB 23 of 576, Density-FB 4 of 576.
- Rollouts are per model (own policy), so the true gaps in the top panel are per model.
  Error bars: s.e. over states (top), 95 % Wilson interval of the binomial fraction (bottom);
  faint markers = per-seed fractions. Pooled over 3 seeds.

## Per-phase table

| model | phase | n | mean true gap +- s.e. (return) | fraction correct | 95 % Wilson |
|---|---|---|---|---|---|
| FB | approach | 182 | 8.1 +- 0.3 | 0.489 (89/182) | 0.417 - 0.561 |
| FB | in hand | 185 | 16.2 +- 0.9 | 0.654 (121/185) | 0.583 - 0.719 |
| FB | carrying | 186 | 32.6 +- 1.4 | 0.500 (93/186) | 0.429 - 0.571 |
| Density-FB | approach | 188 | 10.7 +- 0.6 | 0.468 (88/188) | 0.398 - 0.539 |
| Density-FB | in hand | 192 | 27.2 +- 0.7 | 0.641 (123/192) | 0.571 - 0.705 |
| Density-FB | carrying | 192 | 23.3 +- 1.0 | 0.771 (148/192) | 0.706 - 0.825 |

Per seed (0/1/2), fraction correct at carrying: FB 0.46 / 0.51 / 0.53, Density-FB 0.86 / 0.77 / 0.69.

## Caption (plain RL terms)

Later in the manipulation the action choice matters more: the gap between the best and the
worst candidate action's Monte-Carlo return, under each model's own policy, grows from about
8-11 return units at approach to 16-33 once the cube is held (top; FB's largest gap is at
carrying, Density-FB's at in-hand). At approach neither critic orders the decisive pair better
than chance, and both reach about 0.65 once the cube is in hand. At carrying, where FB's action
gap is largest, FB's critic orders the true best above the true worst at chance level (0.50,
Wilson 0.43-0.57), while Density-FB keeps ordering it correctly about seven times in ten
(0.77, 0.71-0.83) (bottom). Every state is scored against rollouts of the critic's own
policy, so the top-panel gaps are per model.
