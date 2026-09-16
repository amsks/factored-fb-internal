# 2026-09-15 — Why our FB underperforms on ExoRL: audit against the reference

Question: our FB reaches 193 ± 83 on ExoRL walker (1M, ortho 1) and 88 ± 82 at ortho 1000,
against published FB anchors of 760 ± 19 (Touati), a 595 four-task mean (RLDP Table 1) and
a 389 IQM (Jeen et al. on full RND). What is the reference doing that we are not?

## The loss is not the problem

`impls/critics/fb.py::_loss` is a faithful transcription of
`~/git/baselines/td_jepa/metamotivo/agents/fb/agent.py::update_fb`: same off-diagonal
squared term normalised by `off_diag_sum`, same negative-diagonal term scaled by the number
of heads, same B-covariance orthonormality loss (off-diagonal squared + negative diagonal),
same optional auxiliary Q loss with the implicit reward from the B covariance solve (we use
pinv where they use solve, documented in-line). Target computation, the head-ensemble
pessimism reduction and the z mix (`train_goal_ratio` over B(next_obs)) all match.

## The configuration is the problem

Reference: `~/git/baselines/td_jepa/scripts/train/proprio/launch_fb_dmc.py` (BASE_CFG, the
DMC/walker launcher). Ours: `impls/configs/critic/fb.yaml` + the ExoRL launch overrides,
audited from a running `config.json`.

| knob | reference | ours | ratio |
|---|---|---|---|
| train steps | **3,000,000** | 500k–1M | 3–6× |
| batch size | **1024** | 256 | 4× |
| gradient samples seen | 3.07 B | 0.26 B | **12×** |
| L_dim (left-encoder output) | **256** | 50 | 5× |
| forward map | **1024 × 1** | 512 × 2 | — |
| backward map | **256 × 2**, norm | 512 × 4, norm | — |
| left encoder | **256 × 0**, norm | 512 × 4, norm | — |
| f_target_tau | **0.001** | 0.005 | 5× faster tracking |
| b_target_tau | **0.001** | 0.005 | 5× |
| ortho_coef | 1 (swept 0.1 / 1 / 10) | 1.0 (`fb`), 1000 (`fb_tdj`) | 1000× on fb_tdj |
| z_dim | 50 | 50 | ✓ |
| discount | 0.98 | 0.98 (override) | ✓ |
| dataset | 5,000 RND episodes | 5,000 (`exorl_max_episodes`) | ✓ |
| lr_f / lr_b | 1e-4 (lr_b swept 1e-4 / 1e-5) | 1e-4 | ✓ |
| lr_actor | not swept on walker | **3e-4** (yaml says 1e-4; the run carries 3e-4) | check |
| actor | TD3 ("simple", 1024 × 1), z refreshed every 150 steps | ddpgbc with alpha 0 + motivo archi | — |

Three of these are first-order and none of them is the objective:

1. **Sample budget is 12× smaller.** 3M × 1024 against 1M × 256. The compendium already
   listed "1M updates rather than their ~3M" as a stated deviation.
2. **The left encoder is 5× wider and 4 layers deep where the reference has ZERO hidden
   layers**, and its output is 50-dimensional where the reference uses 256. Our FB is
   compressing the observation to a 50-dim bottleneck before F ever sees it.
3. **Target networks track 5× faster** (tau 0.005 vs 0.001), which is the usual cause of
   the seed-to-seed instability we see (walker seeds 148 / 483 / 297 at arch-shape).

`fb_tdj`, which every cube table uses, additionally runs ortho_coef 1000 — three orders
above the reference's swept range {0.1, 1, 10}. On walker that is worth 193 → 88.

## Other references disagree about the budget, so quote the one you compare against

- td_jepa / metamotivo DMC launcher: **3M** steps, batch 1024.
- `enjeeneer/zero-shot-rl` (NeurIPS 2024): **1M** learning steps, batch 512, z_dim 50,
  lr 1e-4, discount 0.98, 10k z-inference samples, eval every 20k over 10 rollouts —
  i.e. our budget, with 2× our batch.
- RLDP (the table we quote): **2M representation + 3M policy**, d = 512, 4 seeds,
  50 eval episodes.

So "FB needs 3M" is true of the metamotivo launcher and false of the NeurIPS-24 reference.
The number we compare against must come with its protocol.

## Running now

`scripts/launch_walker_fbref.sh` — FB at the reference configuration (batch 1024, L_dim 256,
F 1024×1, B 256×2, left encoder 256×0, taus 0.001, ortho 1), walker, 3 seeds, 1M steps with
`--auto_resume` so it can be extended to 3M in place. Save dir `exp/walker_fbref`, tmux
`fbref`, launched 11:33Z. This isolates architecture + batch + tau at OUR budget; if it
lands near the published anchors, the budget question is moot and our default fb.yaml is
simply the wrong configuration for ExoRL.

Also in flight: `exp/walker_arch` (Density-FB and FB at the arch-shape recipe, 1M, 3 seeds)
and `exp/exorl_arch` (Density-FB on cheetah, same recipe, 1M).


## Density-FB audited the same way (2026-09-15 17:30Z)

With FB fixed, Density-FB became the outlier on walker (592 against FB's 703 on the three
tasks scored). The same audit, applied to `impls/configs/critic/density_fb.yaml` + the
arch-shape overrides, leaves exactly TWO deltas against the reference FB configuration:

| knob | reference FB | Density-FB as run | note |
|---|---|---|---|
| forward tower | 1024 wide, **1** trunk layer | `value_hidden_dims=[512,512,512]` = 512 wide, **3** trunk layers | under `measure_density_archi=motivo` the list is read as (width, n_layers) |
| target polyak tau | **0.001** | 0.005 | same 5x mismatch that destabilised FB |
| batch size | 1024 | 1024 | ✓ |
| lr | 1e-4 | 1e-4 | ✓ |
| discount | 0.98 | 0.98 | ✓ |
| z / measure dim | 50 | 50 | ✓ |
| backward map | 256 x 2, norm | `[256,256]` + `b_norm=true` | ✓ |
| architecture | Motivo forward/backward | `archi=motivo` | ✓ |

`scripts/launch_walker_dfbref.sh` runs Density-FB with those two corrected, walker, 3 seeds,
1M steps, same actor and eval protocol as `exp/walker_fbref` -> `exp/walker_dfbref`, tmux
`dfbref`, launched 17:32Z.

This is the arm that decides between the two readings of the walker gap:
* configuration -- Density-FB closes on FB once it has the same shapes and target rate;
* structural -- it does not, and the categorical estimator genuinely trades dense-reward
  resolution for rare-bottleneck resolution. The supporting evidence for the structural
  reading is that the SAME estimator loses 15-21% inside three different hosts on walker
  (FB 703 -> 592, RLDP 584 -> 495, HILP 753 -> 694/594), and that its predicted
  distribution on walker spreads over 28-48 candidates against 7-14 on cube-single.
