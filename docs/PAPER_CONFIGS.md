# Paper configurations: exactly which variant of each method the ICLR paper reports

Updated 2026-09-13. One row per number that appears in the paper's tables; the config that
produced it; whether the backward / feature network receives privileged object coordinates
("object coords" = the hit-spec slice of the observation, `backward_input=hit_spec`) or the
full state. Every OGBench run: 1M gradient steps, flow actor (`--actor flowbc`), reward-inference
evaluation with `eval_relabel_size=10000 eval_relabel_shift=1.0 eval_relabel_episodes=10`,
evaluated every 100k steps, score = mean of the last three evaluation passes (800k/900k/1M),
3 seeds unless stated. Launch lines: `scripts/tdjtab_manifest.sh` (campaign),
`scripts/launch_ortho1000.sh` / `launch_ortho1.sh` (ortho sweep), `scripts/launch_levers.sh`
(Density-RLDP levers). Hub receipts: HF `amsks/factored-fb-checkpoints` `status/`.

## Baselines (all full state; TD-JEPA's published OGBench hyperparameter cell)

| paper row | critic (yaml in `impls/configs/critic/`) | key hyperparameters | B / feature input | where |
|---|---|---|---|---|
| FB | `fb_tdj` | ortho 1000, lr_b 1e-4, batch 256, TD-JEPA architecture | full state | hub `*_fb_tdj_*` |
| HILP | `hilp_tdjc` | phi lr 1e-4, p_randomgoal 0.375, expectile 0.5, batch 256 | full state | hub `*_hilp_tdjc_*` |
| RLDP (Table 1) | `rldp` | ortho **0.01**, lr_b 1e-4, batch 256, SP horizon 5 | full state | hub `*_rldp_*` (cube-single = eval-only rescore, 1 value) |
| RLDP at its own default coefficient | `rldp` + `--override ortho_coef=1` | ortho **1.0** (RLDP's published default; TD-JEPA's grid used 0.01/0.1) | full state | `exp/ortho1/*_rldp_*` (cube-single 27.7 vs 25.0 at 0.01) |

Ortho sweep on the manipulation envs (plain RLDP and Density-RLDP): 0.01 (campaign), 0.1
(`exp/rldp_ortho01`), 1 (`exp/ortho1`), 1000 (`exp/ortho1000`). Higher than 1 is worse everywhere;
1 is the best on cube-single, level elsewhere.

## Density-FB (the method)

| paper row | critic | task interface | B input | where |
|---|---|---|---|---|
| Density-FB, main tables (`z^R`, `z^G`) | `density_fb_z` | reward-sample index (ridge regression on centered B), `measure_density_z_actor_reward=latent_centered`, `z_rand_frac=0.5`, hard one-hot target, batch 1024, d=50, temp 1 | **object coords** (`measure_hit_spec`, per env in the manifest) | hub `*_density_fb_z_*` |
| Density-FB without privileged input | `density_fb_z_state` | same | **full state** (`backward_input=state`) | hub `cube-single/cube-double *_density_fb_z_state_*` (cube-single 90.9 B); scene/puzzle NOT run |
| Density-FB goal-conditioned critic (`goal^G`, diagnostic only) | `density_fb` / `density_fb_state` | goal kernel, exact reward | object coords / full state | hub; scene/puzzle/antmaze `density_fb_state` = luis arm A (pending receipts) |

## Density objective inside the baselines (retrofits)

| paper row | critic | task construction | B / feature input | ortho | where | cube-single B |
|---|---|---|---|---|---|---|
| Density-RLDP, campaign | `rldp_density` | feature index (z = B(random state)), linear actor reward | full state | 0.01 | hub / `exp/ce_audit_retrain_20260911` | 23.8 |
| Density-RLDP, ortho 1 | `rldp_density` + `ortho_coef=1` | feature index | full state | 1 | `exp/ortho1` | 36.7 |
| + CE trains B | `rldp_density_trainb` | feature index | full state | 1 | `exp/levers` | 32.3 |
| + regression-index hindsight tasks (**no privileged input**) | `rldp_density_reg` | hindsight goals, same ridge regression as test time, slow B copy | full state | 1 | `exp/levers` | 53.3 |
| + object coords | `rldp_density_oc` | feature index | **object coords** | 1 | `exp/levers` | 87.7 |
| all three | `rldp_density_full` | regression | **object coords** | 1 | `exp/levers` | 79.7 |
| Density-RLDP, exact-reward actor (withheld) | `rldp_density_gk` / `_gk_oc` | goal kernel reward on candidates | full state / object coords | 0.01 | luis gk arm / `exp/gk_oc` | 30 / 76 (1 seed 500k) |
| Density-HILP, campaign | `hilp_density_tdjc` | whitened phi(g) index, linear reward | full state | n/a (HILP has no ortho term) | hub / `exp/ce_audit_retrain_20260911` | 71.1 |
| Density-HILP + regression tasks | `hilp_density_reg` / `hilp_density_reg_oc` | regression | full state / object coords | n/a | port in progress (2026-09-13) | pending |
| Density-HILP, exact-reward actor (withheld) | `hilp_density_gk` / `_gk_oc` | goal kernel | full state / object coords | n/a | luis gk arm / `exp/gk_oc` | 74 / 92 (1 seed 500k) |

Missing control for the object-coordinate rows: plain RLDP and plain HILP with `backward_input=hit_spec`
at the same settings (`rldp_oc`, `hilp_oc` configs exist; 1M/3-seed runs at ortho 1 not yet run).

## Ablation knobs (implemented 2026-09-12, not yet run)

- `measure_density_loss=sq` on `density_fb_z`: same softmax model, squared error on probabilities instead of cross-entropy.
- `fb_target_scale=0.01` on `fb_tdj`: FB's squared loss with the immediate term scaled by (1-gamma).
