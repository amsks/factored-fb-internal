# FB re-implementation audit against two references (ExoRL walker, RND data)

Read-only audit, 2026-09-17. Ours = `/home/adityamohan/git/Factored-FB-seam`.
Touati = `/home/adityamohan/git/baselines/RLU/controllable_agent/url_benchmark` (official "Does zero-shot RL exist?").
TDJ = `/home/adityamohan/git/baselines/td_jepa` (metamotivo FB, the DMC launcher `scripts/train/proprio/launch_fb_dmc.py`).

Abbreviations in citations: `fb.py` = `impls/critics/fb.py`; `ddpgbc.py` = `impls/actors/ddpgbc.py`;
`networks.py` = `impls/utils/networks.py`; `fb_ddpg.py` = Touati `agent/fb_ddpg.py`; `fb_modules.py` = Touati
`agent/fb_modules.py`; `agent.py` = TDJ `metamotivo/agents/fb/agent.py`; `model.py` = TDJ
`metamotivo/agents/fb/model.py`; `nn_models.py` = TDJ `metamotivo/nn_models.py`.

## 0. A premise correction that matters for the verdict

The two "default-config" walker sets did NOT train 1M steps. Both `exp/walker_dfb/dmc-walker_fb_ddpgbc_seed_0`
and `exp/walker_dfb/dmc-walker_fb_tdj_ddpgbc_seed_0` end at step 500000 (last `eval.csv` row is step 500000;
checkpoints `params_100000 … params_500000`; `scripts/launch_walker_dfb.sh:31` passes `--train_steps 500000`).
The reference-config set `exp/walker_fbref/*` trained 1M (`scripts/launch_walker_fbref.sh:15`, checkpoints up to
`params_1000000`). So the 920/804/400/985 vs 357/49/16/853 comparison confounds a 2x budget difference with the
five configuration differences listed in section 3. Every conclusion below about the default-config collapse is
subject to that.

## 1. LOSS, line by line

| item | ours | Touati | TDJ | verdict |
|---|---|---|---|---|
| target measure | `tM = einsum(F_tgt(le_tgt(s'), z, a'), B_tgt(s'))`, then `mean − c·unc` over heads, `stop_gradient` — `fb.py:133-138` (c = `fb_pessimism_penalty` = 0 in every run ⇒ **mean of heads**) | `target_M = min(target_M1, target_M2)` — `fb_ddpg.py:311-315` (**min of heads**) | `get_targets_uncertainty(target_Ms, fb_pessimism_penalty)` → mean − 0·unc = **mean** — `agent.py:250-253, 357-372` | ours = TDJ; Touati differs (min) |
| target B input | `goal = next_obs` → `B_tgt(next_obs)` — `fb.py:125,135` | `next_goal = batch.next_obs` → `backward_target_net(next_goal)` — `fb_ddpg.py:440, 312` | `goal = bw_encoder(next_obs)` (identity on state) → `_target_backward_map(goal)` — `agent.py:170, 251` | match |
| online B input | `B(next_obs)` — `fb.py:143` | `backward_net(next_goal)` = next_obs — `fb_ddpg.py:319` | `_backward_map(goal)` = next_obs — `agent.py:258` | match |
| bootstrap action | actor mode at **raw** `next_obs`, + clipped N(0,0.2) noise, clip 0.3, re-clipped to [-1,1] — `ddpgbc.py:111-113, 36-52` (0.2/0.3 from the launch overrides) | `dist.sample(clip=0.3)` at std 0.2, actor on raw next_obs — `fb_ddpg.py:308-310` | `dist.sample(clip=stddev_clip)` at `actor_std` 0.2; actor input is **`_target_left_encoder(next_obs)`** because `actor_encode_obs=True` — `agent.py:247-249`, `model.py:54` | ours = Touati on the input; TDJ-DMC feeds the actor the encoded state |
| off-diagonal term | `0.5·Σ_heads Σ_offdiag (M − γ·tM)² / (B²−B)` — `fb.py:145-146` | `0.5·Σ_heads mean_offdiag (M − γ·tM)²` — `fb_ddpg.py:322-324` | `0.5·Σ_{heads,offdiag} diff² / off_diag_sum` — `agent.py:261-262` | identical (all three) |
| diagonal term | `−mean(diag(M − γ·tM))·P` — `fb.py:147` | `−Σ_heads mean(diag M)` — `fb_ddpg.py:325` | `−mean(diag(diff))·P` — `agent.py:263` | ours = TDJ; Touati omits the constant `γ·diag(tM)` — gradient-identical |
| 1/2 factors | 0.5 on off-diag only | same | same | match |
| batch-average vs per-sample diagonal | batch mean | batch mean | batch mean | match |
| stop-gradients | `stop_gradient` on target; target nets on s'; online nets get grads for F, B, left encoder in one `value_and_grad` — `fb.py:138`, `impls/agents/__init__.py:316-330` | `torch.no_grad()` on target; single `fb_loss.backward()` over F and B — `fb_ddpg.py:307, 383` | `torch.no_grad()`; one backward over F, B, left encoder — `agent.py:246, 292` | match |
| ortho regularizer form | `0.5·Σ_offdiag (BBᵀ)²/(B²−B) − mean(diag BBᵀ)`, on **B(next_obs)** (online) — `fb.py:148-151` | `mean_offdiag (BBᵀ)² − 2·mean(diag BBᵀ)` on B(next_obs) — `fb_ddpg.py:344-347` | `0.5·Σ_offdiag (BBᵀ)²/off_diag_sum − mean(diag)` on B(goal=next_obs) — `agent.py:267-270` | ours = TDJ; **Touati's form is exactly 2× TDJ's**, so Touati coef c ≡ TDJ coef 2c |
| ortho on which rows | full B×B covariance of B(next_obs), off-diagonal squared + negative diagonal (not the d×d feature covariance) | same | same | match |
| ortho coefficient | `ortho_coef` 1.0 (fb.yaml:10) / 1000 (fb_tdj.yaml:17); fbref override 1 | dataclass default **0.0** (`fb_ddpg.py:69`); the value the paper ran is not in this repo (no launch yaml/sh sets it; `base_config.yaml` is hydra plumbing only) | 1 in BASE_CFG, swept {0.1, 1, 10} — `launch_fb_dmc.py:63, 77` | fb_tdj's 1000 is 100–1000× above the TDJ sweep |
| two F heads / min | `num_parallel` 2, ensembled towers with separate params — `networks.py:167-183` | F1, F2 heads sharing the preprocessing embeddings — `fb_modules.py:182-184, 191-201` | `num_parallel=2` via `DenseParallel`, separate embeddings per head — `nn_models.py:116, 226-233` | ours = TDJ (per-head embeddings); Touati shares embeddings |
| z mixing | with prob `train_goal_ratio` 0.5, z = `project_z(B(next_obs)[perm])` (online B, pre-step), else Gaussian projected to the √d sphere — `fb.py:188-199` | `mix_ratio` 0.5, z = normalized `B(batch.obs[perm])` (**current obs**, not next) — `fb_ddpg.py:451, 460-485` | `train_goal_ratio` 0.5, `B(goal[perm])` with goal = next_obs — `agent.py:145-156` | ours = TDJ; Touati mixes from B(obs) |
| z normalization | `√d · z/(‖z‖+1e-12)` — `impls/utils/fb_math.py:5-9` | `√d · F.normalize(z)` — `fb_ddpg.py:224-232, 483-484` | `√d · F.normalize(z)` — `model.py:123-126` | match (eps placement nil) |
| discount | constant γ (terminals zeroed, `impls/utils/datasets.py:177`); 0.99 in fb.yaml:7, **0.98 by override** in every walker run | `batch.discount = γ·1` (DMC never terminates) — `in_memory_replay_buffer.py:171`; γ default 0.99 (`pretrain.py:68`) | `γ·(1−terminated)`, terminated always False on DMC — `agent.py:184`; γ 0.98 — `launch_fb_dmc.py:62` | ours = TDJ |
| target update (τ) | F, left encoder at `f_target_tau`; B at `b_target_tau` — `fb.py:79-81`, `impls/agents/__init__.py:352-353`; 0.005 default, 0.001 fbref | one `fb_target_tau` 0.01 for F and B — `fb_ddpg.py:49, 500-503` | 0.001 / 0.001 (DMC), left encoder rides F's τ — `agent.py:221-224`, `launch_fb_dmc.py:64-65` | fbref = TDJ; default 5× TDJ; Touati 10× TDJ |
| learning rates | Adam per group: F/left encoder `lr_f`, B `lr_b`, actor `lr_actor` — `impls/agents/__init__.py:184, 231, 266-277`; 1e-4 / 1e-4 / **3e-4** (`impls/configs/actor/ddpgbc.yaml:1` wins over `fb.yaml:20`; saved config confirms 0.0003) | Adam, all 1e-4 (`lr_coef` 1) — `fb_ddpg.py:47-48, 146-151` | Adam, `lr_f`=`lr_b`=`lr_actor`=1e-4 — `agent.py:23-25, 89-108`; `lr_b` swept {1e-4, 1e-5} | **actor lr 3× both references** |
| batch size | 256 (fb.yaml:3); 1024 fbref | 1024 — `fb_ddpg.py:65` | 1024 — `launch_fb_dmc.py:61` | default 4× smaller |
| update frequency / order | every step: sample z and a' at pre-step params, critic step, actor step at post-step critic, polyak — `impls/agents/__init__.py:382-396, 316-353` | offline: `update_every_steps` forced to 1 (`train_offline.py:63`); FB then actor then polyak — `fb_ddpg.py:493-503` | FB then actor then polyak — `agent.py:201-224` | match |
| auxiliary Q loss | off (`q_loss_coef` 0) — `fb.py:154-169` (pinv where the refs solve/invert) | off (`q_loss` False) — `fb_ddpg.py:79, 330-340` | off (0.0) — `agent.py:35, 274-287` | match (all off) |
| reward-inference z | `z = project_z(Σ_i r_i · B(s'_i))` over N rows drawn from the buffer's `next_observations` with the relabelled reward at s' — `fb.py:201-206`, `impls/main.py:478-487`, `impls/critics/base.py:63-66`; N = 50 000 for `fb` (`scripts/exorl_paper_eval.py:89-97` has no `fb` entry → 50k at :275-277) | `z = normalize(rᵀB(next_obs)/N)`, N = `num_inference_steps` **5120**, reward relabelled at the sampled next state's physics — `fb_ddpg.py:55, 188-222`, `in_memory_replay_buffer.py:163-170`, `pretrain.py:163-184` | `z = project_z(Σ r·B(next_obs))`, reward relabelled from `next.physics`; N = **10 000** in the DMC launcher (50 000 is the eval-class default) — `model.py:134-145`, `metamotivo/evaluations/dmc.py:30, 86-115`, `launch_fb_dmc.py:190-192` | same estimator (1/N is absorbed by the projection); sample count 50k vs 5120 / 10k |

Ridge/least-squares: none of the three uses it for FB; all three use the plain reward-weighted mean of B.

## 2. ARCHITECTURE

| item | ours (as run) | Touati (code defaults, `preprocess=True, add_trunk=False`) | TDJ DMC | verdict |
|---|---|---|---|---|
| F input | `left_encoder(obs)` (L_dim-wide, √L-normalized) — `fb.py:141-142` | **raw obs** (24-d), no encoder — `fb_ddpg.py:125-127` | `left_encoder(obs)`: **one `Linear(24→256)` then `√256·normalize`** ("shallow": `hidden_layers 0, norm True`) — `nn_models.py:185-192`, `launch_fb_dmc.py:49-55, 141` | fbref: see left-encoder row |
| F embeddings | `embed_z([obs_feat,z])` and `embed_sa([obs_feat,a])`, each Dense(h)→LN→tanh→Dense(h/2)→ReLU; concat; `hidden_layers`×[Dense(h)→ReLU]; Dense(z_dim) — `networks.py:134-164` | `obs_z_net`, `obs_action_net`: Linear(1024)→LN→Tanh→Linear(512)→ReLU; concat (1024); per head Linear(1024)→ReLU→Linear(50) — `fb_modules.py:167-184, 191-201` | identical structure to ours, `hidden 1024, hidden_layers 1, embedding_layers 2` — `nn_models.py:199-242` | ours ≡ TDJ; ≡ Touati up to shared-vs-per-head embeddings |
| F sizes | default 512 wide, 2 trunk layers; fbref 1024 wide, 1 trunk layer (checkpoint: `embed_*_0 (2,·,1024)`, `fs_0 (2,1024,1024)`, `fs_1 (2,1024,50)`) | 1024 hidden, 512 feature, 1 trunk layer/head | 1024 × 1 | fbref = TDJ = Touati shape; default differs |
| LayerNorm placement | after the first Dense of each embedding (eps 1e-5) | after the first Linear of each preprocessing mlp | same | match |
| B network | `PhiMap`: Dense(h)→LN→tanh→(L−1)×[Dense(h)→ReLU]→Dense(z)→`√d·x/‖x‖` — `networks.py:56-75, 49-53`; default 512×4, fbref 256×2 | `mlp(obs, 526, ntanh, 526, relu, 50)` then `√d·normalize` — `fb_modules.py:222, 229-232` | Linear→LN→Tanh→(L−1)×[Linear→ReLU]→Linear(z)→Norm; 256×2 — `nn_models.py:185-196`, `launch_fb_dmc.py:43-48` | fbref ≡ TDJ; Touati 526 wide |
| left encoder with `hidden_layers: 0` | `PhiMap(hidden_layers=0)`: the loop `range(-1)` is empty, so it is **Dense(24→256)→LN→tanh→Dense(256→256)→norm** — `networks.py:66-74`; confirmed in the fbref checkpoint (`left_encoder/Dense_0 (24,256)`, `LayerNorm_0`, `Dense_1 (256,256)`) | n/a | `if cfg.hidden_layers == 0: seq = [nn.Linear(obs, z_dim)]` then Norm — a **single linear layer** — `nn_models.py:189-192` | **mismatch**: ours has an extra LN+tanh+Dense that the reference does not |
| observation preprocessing | none (`FBDataset` emits raw float32 rows — `impls/utils/datasets.py:158-180`) | none (`nn.Identity` encoder for states — `fb_ddpg.py:106-109`) | `IdentityNormalizerConfig` — `launch_fb_dmc.py:29-31`, `agent.py:186-189` | match |
| actor network | `MotivoPolicy`: `embed_z([obs,z])`, `embed_s(obs)`, each Dense(1024)→LN→tanh→Dense(512)→ReLU; concat; 1×[Dense(1024)→ReLU]; Dense(6); **tanh on the mean** — `impls/actors/_motivo_nets.py:44-78` (`actor.motivo_archi=true`, `hidden_dims=[1024]`) | `obs_net`, `obs_z_net` each Linear(1024)→LN→Tanh→Linear(512)→ReLU; concat; policy Linear(1024)→ReLU→Linear(6); tanh — `fb_modules.py:92-94, 107, 123-124` | `Actor`: same two-embedding shape, 1024 × 1 — `nn_models.py:253-277`, `launch_fb_dmc.py:38-42` | match in shape; init differs (ours `variance_scaling` `_motivo_nets.py:26` vs orthogonal gain 1 both refs) |
| actor input | raw obs + z (`_obs_feat` is identity on state) — `ddpgbc.py:21-28, 90` | raw obs + z — `fb_ddpg.py:258-262` | **`left_encoder(obs)` + z** (`actor_encode_obs=True`; OGBench launcher sets it False, DMC launcher does not) — `agent.py:331-334`, `model.py:54, 84, 116`, `launch_fb_ogbench.py:67` | ours = Touati; differs from TDJ-DMC |
| actor loss | `−mean_heads Q(s, clip(μ+ε))` (c=0 → mean; `actor_pessimism_penalty` 0.5 would give the min), ε = clipped N(0,0.2) clip 0.3 (`q_noise_std`), no BC (`alpha` 0), no Q-normalizer (`q_normalize` false) — `ddpgbc.py:85-103` | `−mean min(Q1,Q2)` at `dist.sample(clip=0.3)`, std 0.2; no BC — `fb_ddpg.py:395-406` | `−mean Q` with Q = mean over heads (pessimism 0) at `dist.sample(clip=0.3)`; `bc_coeff` 0 ⇒ no BC and no normalizer — `agent.py:334-346` | ours = TDJ; Touati uses **min** in the actor |
| gradient to encoder from actor | `stop_encoder_grad=True` — `ddpgbc.py:95`, `fb.py:174-176` | n/a (no encoder) | `left_enc` under `no_grad` — `agent.py:331-332` | match |
| z in actor loss | same z as the FB step (from `cond`) — `ddpgbc.py:87` | same z — `fb_ddpg.py:497` | same z — `agent.py:212-216` | match |
| policy noise at target | std 0.2 / clip 0.3 (`boot_noise_*` overrides) — `ddpgbc.py:111-113` | 0.2 / 0.3 (`stddev_schedule "0.2"`, `stddev_clip 0.3`) — `fb_ddpg.py:60-61, 308-310` | `actor_std` 0.2 / `stddev_clip` 0.3 — `model.py:52`, `agent.py:34, 228-232` | match |
| truncation of noisy actions | hard `jnp.clip` (zero gradient outside [-1,1]) — `ddpgbc.py:52, 91` | `TruncatedNormal._clamp`: **straight-through** clamp (gradient passes) — `utils.py:164-185` | same straight-through clamp — `nn_models.py:573-594` | minor deviation in the actor loss only (see §5) |
| actor type | deterministic mean (tanh) + fixed Gaussian noise (TD3-style); `const_std` true so `mode()` is the mean | TD3-style truncated normal with fixed std | same | match |
| eval action | `dist.mode()` — `ddpgbc.py:132-134`; called with `temperature=0` — `env/unified.py:502-504` | `dist.mean` under `eval_mode=True` — `fb_ddpg.py:267-268`, `pretrain.py:391-395` | `act(mean=True)` — `model.py:128-132`, `metamotivo/envs/utils/rollout.py:11-17` | match |

## 3. HYPERPARAMETERS

Columns: (A) our `fb.yaml` default as run in `exp/walker_dfb` (`config.json`), (B) our `launch_walker_fbref.sh`
run (`exp/walker_fbref/.../config.json`), (C) Touati code defaults for offline DMC (`fb_ddpg.py:37-82`,
`pretrain.py:52-85`, `train_offline.py:40-54`), (D) TDJ `launch_fb_dmc.py` BASE_CFG. `!` marks a mismatch against
the column the run claims to reproduce (D); `!!` marks a mismatch against both references.

| knob | (A) default run | (B) fbref run | (C) Touati | (D) TDJ DMC | flag |
|---|---|---|---|---|---|
| gradient steps | **500 000** (`launch_walker_dfb.sh:31`, eval.csv) | 1 000 000 | 1 000 000 (`num_grad_steps`) | 3 000 000 | A: 1/6 of D, 1/2 of C; B: 1/3 of D |
| dataset | 5000 RND episodes (`env/domains/dmc-walker.yaml:35`) | 5000 | `replay_buffer_episodes` 5000 | `load_n_episodes` 5000 | match |
| batch | 256 | 1024 | 1024 | 1024 | A `!!` |
| z_dim | 50 | 50 | 50 | 50 | match |
| L_dim | 50 | 256 | — (no encoder) | 256 | A `!` |
| F | 512 × 2 trunk | 1024 × 1 | 1024 hidden / 512 feature / 1 trunk | 1024 × 1 | A `!!` |
| B | 512 × 4 | 256 × 2 | 526 × 2 | 256 × 2 | A `!!` |
| left encoder | 512 × 4 → 50, norm | "256 × 0" = 2 Dense + LN + tanh → 256, norm | none | 1 Linear → 256, norm | A `!!`; B `!` (extra nonlinearity) |
| discount | 0.98 (override; yaml 0.99) | 0.98 | 0.99 (code default) | 0.98 | match D |
| τ F / B | 0.005 / 0.005 | 0.001 / 0.001 | 0.01 / 0.01 | 0.001 / 0.001 | A `!!` (5× D, ½ C) |
| lr F / B | 1e-4 / 1e-4 | 1e-4 / 1e-4 | 1e-4 / 1e-4 | 1e-4 / {1e-4, 1e-5} | match |
| lr actor | **3e-4** | **3e-4** | 1e-4 | 1e-4 | A, B `!!` |
| ortho coef | 1.0 (`fb`) / 1000 (`fb_tdj`) | 1 | 0.0 default (2× form) | 1 (sweep 0.1/1/10) | fb_tdj `!!` |
| heads (num_parallel) | 2 | 2 | 2 | 2 | match |
| target reduction | mean | mean | min | mean | = D |
| actor Q reduction | mean | mean | min | mean | = D |
| z mix ratio / source | 0.5 / B(next_obs) | same | 0.5 / B(obs) | 0.5 / B(next_obs) | = D |
| actor std / clip | 0.2 / 0.3 | same | 0.2 / 0.3 | 0.2 / 0.3 | match |
| actor arch | Motivo two-embedding 1024 × 1 | same | two-embedding 1024/512 | two-embedding 1024 × 1 | match |
| actor input | raw obs | raw obs | raw obs | encoded obs | B `!` vs D |
| BC term | none (alpha 0) | none | none | none (bc_coeff 0) | match |
| aux Q loss | off | off | off | off | match |
| obs normalizer | none | none | none | identity | match |
| inference samples | 50 000 | 50 000 | 5120 | 10 000 | more samples than either; harmless for a mean estimator |
| eval episodes | 10 | 10 | 10 | 10 | match |
| update every | 1 | 1 | 1 | 1 | match |
| weight init | orthogonal (F √2, others 1); actor `variance_scaling` | same | orthogonal gain 1 all | orthogonal (DenseParallel √2, others 1) | actor init differs (minor) |

## 4. VERDICT

### (a) The default-config collapse on walker (357 / 49 / 16 / 853)

First: this run is at 500k steps, the reference-config run at 1M (section 0). Part of the gap is budget. The
fbref run at its own 500k checkpoint (`exp/walker_fbref/.../eval.csv`, walk task, step 500000) would be the
budget-matched comparison and should be read before attributing anything to configuration.

Setting the budget aside, the two runs differ in exactly five things (their saved `config.json`s are otherwise
identical: same actor, same noise, same discount 0.98, same learning rates, same z_dim, same ortho 1):

1. **Left encoder: 512 wide, 4 layers, output 50-d on the √50 sphere** (A) vs a 256-d near-linear projection (B).
   In (A) the forward map never sees the state; it sees a 50-dimensional normalized code trained only by the FB
   loss, and the bootstrap target uses the polyak copy of that code. On walker the reward-relevant signal is
   velocities and orientations that a dense-reward policy has to resolve finely; a 50-d unit-sphere bottleneck
   in front of F is the most structural of the five changes. This is the TDJ OGBench ("deep") recipe transplanted
   to DMC; TDJ's own DMC launcher offers a "deep" variant only at 2 layers (`launch_fb_dmc.py:141-151`), never 4.
   Ranked first on mechanism, not on a measured ablation.
2. **Batch 256 vs 1024.** The FB loss is a contrastive loss over the B×B matrix of (state-action, goal) pairs;
   batch 256 gives 16× fewer off-diagonal pairs per step and 4× fewer gradient samples over the run. Both
   references use 1024. Ranked second.
3. **τ 0.005 vs 0.001** for F, B and the left encoder. 5× faster target tracking; on this domain the earlier
   note recorded large seed-to-seed spread at this τ (walker 148/483/297). Ranked third; plausibly the cause of
   instability rather than of the mean level.
4. **B 512×4 vs 256×2, F 512×2 vs 1024×1.** Capacity/shape; least likely to matter on its own.
5. **`fb_tdj` only: ortho 1000** vs the TDJ sweep {0.1, 1, 10} (Touati form would make that 2000). At 1000 the
   B-covariance term dominates the FB term by three orders and B is driven to be orthonormal on the batch
   regardless of dynamics; this is the extra drop from 357/49/16/853 to 233/32/31/125 and needs no other
   explanation.

The decisive test is five single-knob runs from (B): (B − left encoder → deep 512×4/50), (B − batch → 256),
(B − τ → 0.005), (B − B/F shapes), (B + ortho 1000), all at 1M.

### (b) The reference-config run vs published FB

920 / 804 / 400 / 985 is at or above the published Touati numbers (~900 / 800–820 / 380 / 960) at 1M steps,
so there is no remaining gap to explain against Touati. Against TDJ's own FB (3M steps, best of a
lr_b × ortho × 5-seed sweep) the run still deviates in:

1. **Budget**: 1M vs 3M (`launch_fb_dmc.py:16`).
2. **Left encoder is not the reference's single linear layer** (`networks.py:66-74` vs `nn_models.py:189-192`);
   ours adds LN → tanh → Dense(256→256) before the sphere normalization. Same width, one extra nonlinearity.
3. **Actor input is the raw state, not the encoded state** (`ddpgbc.py:90, 111-113` vs `agent.py:248, 333`,
   `model.py:54`). Both the actor loss and the bootstrap action are affected.
4. **Actor learning rate 3e-4 vs 1e-4** (`impls/configs/actor/ddpgbc.yaml:1` vs `agent.py:25`).
5. Actor init `variance_scaling` vs orthogonal (`_motivo_nets.py:26` vs `nn_models.py:60-64`).
6. Inference samples 50k vs 10k; eval episodes match (10).

Against Touati's code the run is not a port at all: Touati uses min over heads for both target and actor
(`fb_ddpg.py:315, 405`), τ 0.01 (`:49`), a 2× ortho form (`:344-347`), z-mix from B(obs) (`:460`), no encoder,
shared F embeddings, B width 526, and γ 0.99 by code default. Those are TDJ-vs-Touati differences, not bugs.

**Is the reference-config run a faithful port?** Of TDJ's DMC recipe: yes for the loss, the FB towers, the z
law, the noise, the discount, the τ's, the batch and the inference estimator; no on items 2–5 above. None of
2–5 is expected to be large, and the run already matches published Touati at a third of TDJ's budget.

## 5. BUGS (relative to both references)

No wrong stop-gradient, wrong target, wrong sign or wrong normalization was found in `impls/critics/fb.py` or
`impls/actors/ddpgbc.py`. Three deviations are worth recording; none is a correctness bug in the loss.

1. **`PhiMap` with `hidden_layers: 0` is not a linear map.** `networks.py:66-74`:
   ```python
   x = nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(x)
   x = nn.LayerNorm(epsilon=1e-5)(x)
   x = jnp.tanh(x)
   for _ in range(self.hidden_layers - 1):      # range(-1): empty
       ...
   x = nn.Dense(self.z_dim, kernel_init=_ORTH1)(x)
   ```
   TDJ `nn_models.py:189-190` replaces the whole sequence with `[nn.Linear(obs_dim, z_dim)]` when
   `hidden_layers == 0`. The `launch_walker_fbref.sh:13` left encoder therefore has one more nonlinear layer than
   the reference "shallow" encoder (checkpoint confirms `Dense_0 (24,256)`, `LayerNorm_0`, `Dense_1 (256,256)`).
   Affects only configs that ask for zero hidden layers.

2. **Hard clip instead of straight-through clamp on the noisy actor action.** `ddpgbc.py:91-94, 52`:
   ```python
   q_actions = jnp.clip(dist.mode(), -1, 1)
   if self.q_noise_std > 0.0:
       q_actions = _smooth(q_actions, boot["actor_q_rng"], self.q_noise_std, self.q_noise_clip)
   # _smooth: return jnp.clip(a + eps, -1.0, 1.0)
   ```
   Both references (`utils.py:171-174`, `nn_models.py:582-585`) clamp with `x - x.detach() + clamped.detach()`,
   so the actor mean still receives the Q-gradient for coordinates whose noisy value left [-1, 1]; `jnp.clip`
   zeroes it. `networks.py:35-38` already has `truncated_clamp` implementing the reference behaviour, unused
   here. With a tanh mean and 0.3-clipped noise this touches only near-saturated coordinates.

3. **Reward-inference z is `Σ r·B` not `(1/N) Σ r·B`** (`fb.py:205` vs `fb_ddpg.py:216`). Identical under
   `norm_z: true` (every run), would differ from Touati if `norm_z` were ever false.

Also verified as equivalent, for the record: the diagonal term uses `diff` (TDJ) rather than `M` (Touati) —
the extra `γ·diag(tM)` is a stop-gradient constant; `project_z` eps placement; constant-γ vs
`γ·(1−terminated)` on a domain that never terminates; z drawn once per step from the pre-step online B and
shared by the FB and actor updates; actor stepped after the critic and reading post-step critic params;
polyak after the actor step; reward rows aligned to `next_observations` and relabelled at the next state's
physics (`env/exorl_env.py:375-398, 406-425` vs `in_memory_replay_buffer.py:163-170` and
`evaluations/dmc.py:95-112`).
