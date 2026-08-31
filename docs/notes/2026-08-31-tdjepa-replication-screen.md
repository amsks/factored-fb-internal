# td_jepa replication screen — walker, 500k, 3 seeds

## Why this exists

Our ExoRL campaign has been aiming at RLDP Table 1, and that target has problems we
cannot fix from outside: their orthogonality regularizer contributes zero gradient
(2026-08-31-ortho-arms-figure1-gate.md), no launch command in their repo shows which
CLI produced the table, and the paper text contradicts the shipped code on both the step
budget (5M vs 6M) and the eval episode count (50 vs 10).

td_jepa (arXiv 2510.00739, `~/git/baselines/td_jepa` @ 840a745) is a better anchor. It
publishes RLDP *and* HILP on the same ExoRL RND data under one protocol, its regularizer
runs, and its configuration is fully specified. It also reproduces RLDP's own numbers at
1.07x while getting HILP 2.5x higher than RLDP's baseline column — so its RLDP is
credible and its HILP is the one to trust.

This screen asks one question: **do our jax implementations reproduce td_jepa's, run
side by side under the same configuration?** Not "do we match Table 1".

## What is running

12 cells, walker, 500k steps, 3 seeds each.

| side | code | arms | GPUs |
|---|---|---|---|
| theirs | `td_jepa/run_one.py` (imports their BASE_CFG untouched) | `rldp`, `hilp` | 0-5 |
| ours | `impls/main.py` | `rldp_tdj`, `hilp_tdj` | 6-7 |

`run_one.py` exists because their launcher fans a sweep out sequentially in-process with
no way to pin one configuration or choose a GPU. Nothing in `metamotivo/` or their
`scripts/` is modified.

## Configuration

Read from `scripts/train/proprio/launch_rldp_dmc.py` BASE_CFG and the
`FBAgentTrainConfig` defaults it inherits (`metamotivo/agents/fb/agent.py:22`).

**Matched:** batch 1024 · z_dim 50 · L_dim 256 · discount 0.98 · f/b_target_tau 0.001 ·
lr_f/lr_b/lr_actor 1e-4 · ortho_coef 0.01 · train_goal_ratio 0.5 · norm_z · horizon 5 ·
5000 RND episodes · forward 1024x1 · backward 256x2 norm · left_encoder 256x0 norm ·
predictor 1024x1 · single joint objective, no phase split · bc_coeff 0, pessimism 0,
q_loss_coef 0.

ortho_coef deserves a note: BASE_CFG says 1, but `sweep_walker` overrides it to
{0.001, 0.01, 0.1}, so 1 is never what their walker column reports. We pin the middle
value on both sides.

**NOT matched — do not let these go unstated in any writeup:**

1. Our predictor carries an `embedding_layers: 2` stage; theirs has no such knob.
   Removing it means changing a shared critic other arms depend on, so it is disclosed
   rather than "fixed".
2. Actor structure unverified: same width (1024x1), theirs is `"simple"`, ours is
   `motivo_archi`.
3. jax vs torch — different init and RNG. This is why the read is on seed distributions,
   not single numbers, at n=3.
4. **`hilp_tdj` is only a PARTIAL match.** Our HILP is transcribed from OFFICIAL HILP;
   td_jepa's is an SF-family agent wearing their FB towers. Optimiser knobs are matched
   (both taus 0.001, lr_features 1e-4, expectile 0.5, prob_random_goal 0.375, future
   0.99); the towers are a different family. A HILP gap is not by itself evidence of a
   bug. `rldp_tdj` carries the clean test.

## Tasks

td_jepa's walker column is (stand, walk, run, **spin**); RLDP Table 1's is (stand, walk,
run, **flip**). Verified different rewards: spin is `tolerance(angmomentum, bounds=(10,
inf))` returned alone; flip is `tolerance(..., bounds=(5, inf))` multiplied into
standing. `spin` is now ported into `env/exorl_dmc_tasks/walker.py` and selected with
`--task-set tdjepa`, so all four cells line up and the RLDP-paper comparison still
scores flip.

Their walker.xml differs from ours only in three `<include>` lines hardcoding
`/private/home/denisy/.conda/...` for skybox/visual/materials. Cosmetic; dynamics
identical.

## How to score

Theirs writes one row per eval EPISODE per task per eval point into `reward.csv`, at
`eval_every_steps = 250_000` (so 0 / 250k / 500k). `scripts/tdjepa_report.py` averages
episodes WITHIN a seed before taking mean/std ACROSS seeds.

Ours will be scored at **their** protocol, not the RLDP-paper one: 10 episodes, 10k
inference samples, `--task-set tdjepa`.

Untrained floor, from their own t=0 evals (3 seeds): stand 164.9, walk 29.7, run 27.4,
spin 27.0. Any trained cell must clear these or the eval path is broken.

## Reading the result

500k is a screen, not a converged run — their published numbers are 3M and best-of-sweep
over lr_b x ortho_coef x 5 seeds. So the question is whether the two implementations sit
in the same place at the same step under the same config, NOT whether either hits 723.9.

A large RLDP gap means our jax RLDP diverges from theirs and the divergence is findable.
A small gap licenses using our jax codebase — which has PSM and all three density
variants, none of which exist in td_jepa — for the density comparison, anchored to an
external reference.
