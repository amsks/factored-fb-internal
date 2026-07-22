# Jaxify coverage audit — gciql_flow.py / configs/gciql knobs vs impls/

Verdict: **no live gap.** Every research knob used by the deleted JAX shim
(`tools/wandb_mode_shim/gciql_flow.py`) and `configs/gciql/*.yaml` is either
covered by an `impls/` critic/actor, renamed, a dead-path drop, or an explicitly
deferred feature. `impls/` therefore fully replaces the shim for in-scope work.

| Knob | Role | impls/ status | Note |
|---|---|---|---|
| value_arch | bilinear vs mlp value head | covered | encoded by critic choice (iql_bilinear / centered_bilinear / simple_sf) + `value_arch` on centered/simple_sf |
| value_latent_dim | goal-bottleneck rank d | covered | `value_latent_dim` in bilinear critic configs |
| bc_coeff | flow-BC distillation weight | covered | `bc_coeff` in configs/actor/flowbc.yaml |
| project_z | sphere-norm z/psi | covered | `project_z` across critics + utils/fb_math |
| fb_forward_z | z-indexed forward | covered | `fb_forward_z` in fb_forward_bilinear / sf_feature |
| fwd_two_tower | two-tower phi | covered | `phi_module='two_tower'` in fb_forward_bilinear |
| sf_feature_td, sf_coef, sf_base, sf_ortho_coef, sf_reward_regress, sf_joint_psi | SF-feature family | covered | critics/sf_feature.py |
| actor_cond, actor_z_from_critic, actor_z_infer | actor conditioning | covered | iql_bilinear / fb_forward_bilinear |
| expectile, double_q, iql_ortho_coef | IQL head scalars | covered | bilinear critics |
| centered_measure, centered_measure_coef, center_b_samples | centering/dueling | covered | `mode` enum + `centered_measure_coef` in centered_bilinear |
| measure_coef, measure_legs, measure_ortho_coef, measure_stopgrad_phi, measure_scale_norm, measure_test_states, measure_insample | FB successor-measure family | covered (structural) | fb_forward_bilinear `objective=measure/mix/isfb` — most are UNVERIFIED/dead paths per T7/T8 + measure-cell campaigns |
| measure_only | measure objective select | renamed | `objective=measure` |
| psi_base, simplesf_ground_psi | psi grounding | covered | iql_bilinear / simple_sf |
| isfb, alpha, awr_beta | isfb objective | covered | critics/isfb.py + fb_forward_bilinear isfb terms |
| flow_levels | flow-actor ODE steps | renamed | `flow_steps` in flowbc + fql actors |
| actor_kind | actor type select | covered | the actor NAME (flowbc/ddpgbc/awr/fql) via the combo seam |
| anneal_ramp, anneal_measure, anneal_lr_warmup, anneal_iql_fade, anneal_simplesf_ground | curriculum anneal | **dead-path drop** | anneal campaign concluded the measure fixed point is bad on cube (L_IQL was propping the basin) |
| measure_z_random_frac | random-z frac in measure | **dead-path drop** | measure family negative |
| freeze_critic_ckpt | frozen-critic ablation | **drop** (experimental) | `impls/main.py` has restore_path/restore_epoch; the freeze ablation is reconstructible, not a core knob |
| subgoal_steps, low_actor_rep_grad | HIQL knobs | **out of scope** | HIQL is not a bridge agent |
| value_split | scene factored value | **deferred** | scene domain not ported (cube-single only) |
| encoder, p_aug, frame_stack | pixel/DrQ | **deferred** | impls/ is state-only |

## Baseline

Recorded at cleanup start (see the cleanup plan `docs/plans/2026-07-22-jaxify-cleanup.md`):
`tests/impls/` full suite = **353 passed, 1 skipped** (CPU, ~6 min). This is the gate the
final verification must reproduce. Per-deletion tasks use `pytest --collect-only` (imports
every test module + its deps, catching any broken import from a deletion) as a fast gate,
with the full 353-pass run at the end.
