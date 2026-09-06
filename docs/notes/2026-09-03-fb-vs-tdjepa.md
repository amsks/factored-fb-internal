# seam jax FB vs td_jepa FB — line-by-line audit (2026-09-03)

Scope: the parts the controller had **not** already checked. Already verified elsewhere and
re-confirmed here in passing: FB TD loss (off-diag/diag forms), ortho loss, target-tower
usage, the 50/50 z-mix law, `infer_cond` (z ∝ rᵀB then project), and
`project_z = √d·normalize`.

Sources
* seam: `impls/critics/fb.py`, `impls/configs/critic/{fb,fb_ref}.yaml`,
  `impls/actors/{flowbc,ddpgbc}.py`, `impls/configs/actor/flowbc.yaml`,
  `impls/utils/{networks,fb_math,datasets,evaluation}.py`, `impls/agents/__init__.py`,
  `impls/main.py`, `env/ogbench.py`, `scripts/launch_hilp_state_and_fb_audit.sh`
* reference: `~/git/baselines/td_jepa/metamotivo/agents/fb/{agent,model}.py`,
  `.../fb/flow_bc/{agent,model}.py`, `metamotivo/{nn_models,normalizers}.py`,
  `metamotivo/agents/td3/nn_models.py`, `metamotivo/evaluations/ogbench.py`,
  `metamotivo/envs/ogbench.py`, `metamotivo/data_loading/base.py`,
  `scripts/train/proprio/launch_fb_{ogbench,dmc}.py`
* the 0.420 cell: `origin/gciql-fb-bridge` (`configs/agent/fb{,_flowbc}.yaml`,
  `configs/train.yaml`, `evals/ogbench.py`,
  `scripts/rescuing_fb/launch_missing_baselines.sh`)

---

## 1. Observation normalization — NO DIVERGENCE (hypothesis defused)

`FBAgent.update` does run a running normalizer on obs/next_obs every step
(`agents/fb/agent.py:186-189`: one train-mode call to update the statistics, then an
`eval_mode` no-grad call to apply them). But **both proprio launchers pin it to identity**:

* `scripts/train/proprio/launch_fb_ogbench.py:33-35` → `"obs_normalizer": {"name": "IdentityNormalizerConfig"}`
* `scripts/train/proprio/launch_fb_dmc.py:29-31` → same

and `normalizers.py:33-35` builds a bare `nn.Identity` for that config. `BatchNormNormalizer`
exists (`normalizers.py:16-30`) but is never selected by the FB launchers; `RGBNormalizer`
is the pixel path. So for every number td_jepa reports on OGBench **and** DMC proprio, the
normalizer is a no-op and `obs` reaches the towers raw.

The seam normalizes nothing (`impls/utils/datasets.py:120-141` emits raw float32 rows; grep
for `normaliz` across `impls/` returns only Q-normalizers and √d row-normalizers). **This is
therefore parity, not a divergence.** The mixed proprio/object-scale worry does not apply to
the reference either — it eats the same unnormalized 19-dim cube state we do, and its own
normalization of scale happens inside the towers (`LayerNorm`+`tanh` as the first stage of
every embedding, and `√d`-normalize on B's output).

Corollary: adding an obs normalizer to the seam would be a *departure* from the reference,
not a fix.

## 2. Actor family

### 2a. flow_bc (the OGBench/cube path) — FAITHFUL

`FBFlowBCAgent.update_actor` (`fb/flow_bc/agent.py:53-107`) vs
`FlowBCActor.loss` (`impls/actors/flowbc.py:56-88`), term by term:

| term | td_jepa | seam | verdict |
|---|---|---|---|
| flow-matching BC | `x_t=(1-t)x_0+t x_1`, `vel=x_1-x_0`, `mse(vf(obs,x_t,t), vel)` (`:61-69`) | identical, `boot["flow_x0"]/["flow_t"]` drawn in `sample_bootstrap` (`:90-104`) | match |
| Q term | `left_enc` under `no_grad`; `actor_in = obs` (actor_encode_obs False); `Qs=(F·z).sum(-1)`; `Q=mean over heads` (pessimism 0); `-Q.mean()` (`:72-80`) | `critic.q_fn(..., stop_encoder_grad=True)`, `targets_uncertainty` mean, `-q_coeff*Q.mean()` (`fb.py:171-181`) | match |
| distillation BC | `target = compute_flow_actions(obs, noises)` under `no_grad`, **same `noises`** as the one-step actor input; `mse(actor_actions, target)` (`:85-88`) | `_rollout(...)` under `stop_gradient`, same `noise` (`flowbc.py:80-82`) | match |
| loss shape | `actor_loss = (-Q.mean()/sg[Qs.abs().mean()]) + bc_coeff*bc_error`, **then** `+ bc_flow_loss` (`:89-91`) | identical, `jnp.abs(Qs).mean()` over the **per-head** `Qs` (`:84-86`) | match |
| flow rollout | 10 Euler steps `a += v/steps`, clamp ±1 (`:109-116`) | identical (`flowbc.py:36-42`) | match |
| optimizers | actor lr 1e-4, actor_vf lr 3e-4, separate Adams (`flow_bc/agent.py:41-46`) | `_LR_KEY["actor"]="lr_actor"`, `actor_vf` → `actor.lr_actor_vf` (`agents/__init__.py:165,217`); yaml 1e-4 / 3e-4 | match |
| update order | `update_fb` steps F/B/left_enc, **then** `update_actor` sees post-step params | seam stages the critic first, actor reads `online()` post-step (`agents/__init__.py:250-284`) | match |
| bootstrap action | `sample_action_from_norm_obs` = one-step noise-conditioned actor on raw `next_obs` (`flow_bc/agent.py:48-51`, called at `agent.py:249`) | `sample_bootstrap` → `_actor(next_obs, cvec, noise)` (`flowbc.py:90-97`) | match |

Config: `bc_coeff 3.0`, `flow_steps 10`, `lr_actor_vf 3e-4`, actor 512×2 embedding 2,
actor_vf 512×4 — the seam's `impls/configs/actor/flowbc.yaml` matches
`launch_fb_ogbench.py:42-50` exactly.

### 2b. TD3 actor (DMC path) — DIVERGES, but off the cube path

`FBAgent.update_td3_actor` (`agent.py:327-355`):
`dist = Actor(obs, z, actor_std=0.2)` (TruncatedNormal, fixed std),
`a = dist.sample(clip=stddev_clip=0.3)` — a **noisy** action into the Q term —
`bc_error = mse(a, data_action)`, `actor_loss = -Q.mean()/sg|Qs|.mean() + bc_coeff*bc_error`.
Note `actor_std` is a **constant** (`FBModelConfig.actor_std: float = 0.2`), not a schedule;
there is no stddev annealing anywhere in this repo.

`impls/actors/ddpgbc.py` differs in four ways: (i) BC is `-alpha·log_prob(data_action)` under
a Gaussian actor, not MSE to the data action; (ii) the Q term uses `dist.mode()` by default
(`q_noise_std` defaults 0.0, so the reference's clipped sample is off); (iii) the actor net is
a plain `concat([obs, cvec]) → MLP` rather than the reference's two-embedding
`Actor` (the faithful one, `FBTd3Actor`, exists in `impls/utils/networks.py:250-266` but is
unused); (iv) `q_normalize` divides by `|q̄|.mean()` of the head-mean rather than `|Qs|.mean()`
over per-head values, plus a `1e-6` floor. **Irrelevant to the cube-single cell** (that runs
`--actor flowbc`); it matters only if a DMC/ExoRL TD3-parity arm is ever built.

## 3. Networks — FAITHFUL, including inits

* `BackwardMap` (`nn_models.py:177-196`): `Linear→LayerNorm→Tanh`, `(hidden_layers-1)×[Linear→ReLU]`,
  `Linear(z_dim)`, `Norm()` = `√d·F.normalize`. `PhiMap` (`impls/utils/networks.py:56-75`) is
  the same sequence with `LayerNorm(epsilon=1e-5)` (torch's default; flax's is 1e-6, and the
  seam overrides it correctly) and `psm_norm`. The `hidden_layers == 0` shallow special case
  (`nn_models.py:189-190`) is **not** ported, but OGBench uses `deep` (4 layers) — only the DMC
  `shallow` left-encoder would need it.
* `simple_embedding` (`nn_models.py:199-205`) ↔ `_simple_embedding` (`networks.py:78-85`):
  identical, and **`embedding_layers: int = 2` exists on the reference side too**
  (`ForwardArchiConfig.embedding_layers`, `nn_models.py:116`; `ActorArchiConfig`, `:141`;
  `NoiseConditionedActorArchiConfig`, `:423`). The premise that we carry an embedding stage
  they lack is wrong — both sides have exactly two embedding layers producing `hidden//2` each,
  concatenated to `hidden`.
* `ForwardMap` (`nn_models.py:208-242`) ↔ `_ForwardTower`/`ForwardMap` (`networks.py:134-183`):
  same `embed_z([obs,z])` / `embed_sa([obs,action])`, same `cat([sa, ze])` order, same
  `hidden_layers×[Linear→ReLU] + Linear(z_dim)` trunk, `num_parallel=2` independent heads
  (`ensemblize` splits param rngs, `networks.py:17-27`).
* `NoiseConditionedActor` (`nn_models.py:428-449`) ↔ `networks.py:88-108`: same
  `embed_z([obs,z,noise])`, `embed_s([obs,noise])`, `cat([s,z])`, `tanh` head.
* `VectorField` (`agents/td3/nn_models.py:125-141`) ↔ `FlowVectorField` (`networks.py:111-126`):
  `cat([obs,a,t]) → Linear→GELU`, `(L-1)×[Linear→GELU]`, `Linear(adim)`. The seam correctly
  uses exact (erf) GELU, matching `nn.GELU()`.
* **Init**: `weight_init` (`nn_models.py:60-77`) gives `nn.Linear` orthogonal **gain 1** and
  `DenseParallel` orthogonal **relu gain √2** (including the final, un-ReLU'd output layer).
  The seam mirrors this bug-for-bug: `_ORTH1` on all non-ensembled Dense, `_ORTH_RELU` on every
  Dense inside the ensembled forward tower (`networks.py:31-32, 146-154`). LayerNorm falls into
  the `reset_parameters` branch → ones/zeros, which is flax's default.

Architecture config parity for cube (`launch_fb_ogbench.py:36-67` vs
`impls/configs/critic/fb.yaml`): F 512×2 emb 2, B 512×4 norm, left_encoder 512×4 norm,
`L_dim 50`, `z_dim 50`, `norm_z true`, `num_parallel 2`, `actor_encode_obs false`
(the seam's actor takes raw obs, `_obs_feat` is identity on state), batch 256, γ 0.99,
τ_f = τ_b = 0.005, lr_f = lr_b = lr_actor = 1e-4. **All identical.**

## 4. Update loop — everything else

| item | td_jepa | seam | verdict |
|---|---|---|---|
| amp / autocast | `FBModelConfig.amp=False` default, launchers don't set it → bf16 autocast disabled | n/a (fp32) | no divergence |
| `clip_grad_norm` | default `0.0` → passed as `None` (`agent.py:199`), launchers don't set it | none | no divergence |
| terminated-masked discount | `discount = γ·~terminated` (`agent.py:184`), with `terminated = 1-discount_col` from the episode npz (`data_loading/base.py:50`); OGBench play episodes never terminate | `terminals` hard-zeroed, constant γ (`datasets.py:139`) | no divergence in practice |
| optimizer grouping | backward+bw_encoder on `lr_b`; forward+**left_encoder**+fw_encoder on `lr_f` (`agent.py:89-102`) | `_LR_KEY`: `left_encoder → lr_f` (`agents/__init__.py:137`) | match |
| polyak | F and **left_encoder** at `f_target_tau`, B at `b_target_tau` (`agent.py:221-224`) | `FBCritic.TARGETS` identical (`fb.py:78-80`) | match |
| z per step | one `sample_mixed_z` shared by the FB and actor updates, detached | `cond` computed once in `Agent.update`, closed over as a constant (no grad path) | match |
| `weight_decay` | 0.0 | `optax.adam` (no wd) | match |
| aux `q_loss` | `torch.linalg.solve`, off by default | `pinv`, off by default | match (off) |
| `use_mix_rollout`, `z_buffer_size`, `update_z_every_step` | online-only, unused offline | n/a | n/a |
| compile / cudagraphs | perf only | jit | n/a |
| `project_z` eps | `√d·F.normalize` = `z/max(‖z‖,1e-12)` | `z/(‖z‖+1e-12)` (`fb_math.py:4-9`) | numerically nil |
| data | full play dataset (`load_n_episodes` default 1000 = 1000×1000 rows), consecutive `(s,a,s')` per episode file | `ogbench.make_env_and_datasets(compact_dataset=True)` → `valids` masks trajectory boundaries (`datasets.py:84-90,105-107`) | match |

## 5. The eval protocol — the one genuine, unmeasured difference

td_jepa's OGBench FB number is a **reward-inference zero-shot** number:

1. sample 10 000 transitions from the training buffer;
2. relabel with `cube_reward_fn(next.physics, ...)` = `successes − num_cubes`
   (`envs/ogbench.py:30-38`), i.e. `−1` unless the cube is within 4 cm of `goal_xyzs`;
3. `rewards += shift_reward = 1` (`evaluations/ogbench.py:97`, set at
   `launch_fb_ogbench.py:174`) → a **0/1 indicator of cube-at-goal**;
4. `z = project_z(Σ r·B(s'))` (`model.py:134-145`);
5. roll 10 episodes per task over 5 tasks; success = **any step** succeeded
   (`evaluations/ogbench.py:61`).

The seam's `fb_ref` cube runs (`scripts/launch_hilp_state_and_fb_audit.sh:50-53,67`) pass no
`--infer_cond_episodes`, so eval takes the env's goal frame and
`FBCritic.map_eval_cond` sets **z = project_z(B(goal_obs))** (`impls/critics/fb.py:86-97`,
goal from `info['goal']` at `impls/utils/evaluation.py:73-86`), 20 episodes per task,
success read off the **final** info row.

These are different task specifications. `B(goal_obs)` is a Dirac at one full 19-dim state,
**including the arm pose** the scripted goal happens to have; the relabelled z is an average of
`B` over every dataset state whose *cube* is at the target, arm free. The success criterion
scores only the cube — so the goal-Dirac z asks the policy for strictly more than the metric
rewards. (The opposite argument also exists: the Dirac z is in-distribution for the 50 %
`B(s')` half of the training z law, while the averaged z is off it — this is the exact concern
recorded in the bridge's `reward_z_ratio` comment. Which effect dominates on cube is an
empirical question nobody has run.)

**This matters because the 0.420 cell was measured under the td_jepa protocol.** The bridge's
torch FB (`origin/gciql-fb-bridge`) runs `OGBenchEvaluator` with
`eval_relabel_size: 10_000`, `eval_shift_reward: 1.0`, `eval_n_episodes: 10`
(`configs/train.yaml:28-30`) and any-step success (`evals/ogbench.py:94-95`), at
`STEPS=500000` and `ortho_coef` 1.0 for "Tuned FB" / 1000 for "FB (reference reimpl)". Its
agent yaml (`configs/agent/fb.yaml`, `fb_flowbc.yaml`) is the same td_jepa OGBench recipe the
seam's `fb.yaml` mirrors. So **0.420 vs 0.11–0.23 is not a training-length or architecture
contrast — the two cells differ in the eval's task specification, and in little else that we
could find.**

Blocking detail: the seam has **no OGBench reward-relabel path at all**. `FBDataset` zeroes
rewards for unlabeled play data (`datasets.py:125-128`), so `--infer_cond_episodes` on cube
infers `z = project_z(0) = 0` and trips the no-op guard (`impls/main.py:136-138`). Relabelling
exists only for ExoRL (`env/exorl_env.py`). The ingredients for the OGBench version are
present but unwired: `add_info=True` loads `qpos` (`env/ogbench.py:73-75`), `qpos[14:17]` is the
cube xyz, and `env.unwrapped.cur_task_info['goal_xyzs']` is the target
(`impls/utils/evaluation.py:37-59` already reads it for the pixel density path).

---

## 6. Verdict — ranked divergences

| # | divergence | kind | severity | note |
|---|---|---|---|---|
| 1 | **Eval z protocol**: seam `z = B(goal_obs)` (goal Dirac, full state incl. arm pose) vs reference/bridge `z = Σ r·B(s')` over 10k relabelled samples with `shift_reward=1` (cube-position indicator) | harness/protocol | **HIGH** | the only substantive difference between the seam cell and the 0.420 cell; not runnable today |
| 2 | **No OGBench relabel path** in the seam, so protocol 1 cannot even be tested (`--infer_cond_episodes` is inert: play rewards are all zero) | code (missing) | **HIGH** | the fix for #1 |
| 3 | `ortho_coef` default **1.0** vs the cube sweep `{100, 1000}`, and the reference number is a **best-of-sweep** over `ortho × lr_b × 10 seeds` | config | **MED** | `fb_ref` already tests 1000 → 0.15–0.23, so ortho alone is ruled out; `lr_b=1e-5` is still untested |
| 4 | Success/episode accounting: **any-step** success over 10 eps × 5 tasks vs seam **final-info** success over 20 eps | protocol | **MED** | biases the seam low wherever an episode touches the goal and drifts off; near-nil if `terminate_at_goal` holds |
| 5 | Training length **500k vs 1M** (`launch_fb_ogbench.py:16`) | config | **MED** | applies to td_jepa's published numbers only — the 0.420 bridge cell was also 500k |
| 6 | `ddpgbc` ≠ their TD3 actor: log-prob BC vs MSE-to-data-action, `dist.mode()` vs clipped sample, concat-MLP vs two-embedding actor, `|q̄|` vs `|Qs|` normalizer | code | **LOW** | off the cube path entirely (cube runs `flowbc`); blocks a DMC parity arm |
| 7 | `PhiMap` lacks the `hidden_layers == 0` shallow special case | code | **LOW** | only the DMC `shallow` left-encoder needs it |
| 8 | `project_z` eps: `‖z‖+1e-12` vs `max(‖z‖,1e-12)` | code | **NIL** | |
| 9 | Obs normalization | — | **NONE** | both identity on OGBench and DMC proprio |
| 10 | `embedding_layers: 2` | — | **NONE** | the reference has it too (`nn_models.py:116`) |
| 11 | amp/autocast, `clip_grad_norm`, terminated-masked discount, optimizer grouping, polyak taus, update order, one-z-per-step, dataset boundaries | — | **NONE** | all verified equivalent at the shipped configs |

Bottom line: **the seam's FB critic, flow-BC actor and networks are a faithful port; the
config is a bit-for-bit match of td_jepa's OGBench BASE_CFG apart from `ortho_coef`.** The
gap to 0.420 is therefore almost certainly *not* code-structural in the trained object — it
is either the eval's task specification (#1/#2) or the sweep-and-seed accounting (#3/#5), and
#1 is the only one that has never been measured on either side.

## 7. Minimal `fb_tdj` parity arm

One critic yaml plus one eval path:

1. **`impls/configs/critic/fb_tdj.yaml`** = `fb.yaml` with `ortho_coef: 1000` (i.e. `fb_ref`),
   swept against `lr_b: 1.0e-5`; actor `flowbc` (`bc_coeff 3.0` already); `--train_steps 1000000`.
2. **OGBench reward-relabel + reward-inference eval** (the load-bearing part). In
   `env/ogbench.py`: force `add_info=True` when a new `eval_relabel_size` key is set, and add
   a relabel fn `r = (‖qpos[14:17] − cur_task_info['goal_xyzs']‖ < 0.04) − 1` (per cube,
   summed, minus `num_cubes`) **plus `shift_reward = 1`**; sample 10 000 rows per task, feed
   `(next_observations, rewards)` to `agent.infer_eval_cond`, and run with
   `--infer_cond_episodes 10000 --eval_on_inferred_cond` so the goal frame is never forwarded.
3. **Match the accounting**: 10 eval episodes per task, 5 tasks, success = any step
   (`--eval_episodes 10`; add an any-step reducer to `impls/utils/evaluation.py` if
   `terminate_at_goal` is off for this env build).

Everything else — architecture, inits, losses, taus, learning rates, batch size, optimizer
grouping, update order — is already at parity and should be left alone.
