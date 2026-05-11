# Faithful FB on OGBench — Design

**Date:** 2026-05-11
**Status:** Approved for planning
**Reference:** `../../../../td_jepa/metamotivo/` (Meta Platforms td_jepa, FB and FlowBC agents)

---

## 1. Overview

Restore Factored-FB to a **faithful port of td_jepa's FB and FlowBC agents on state-based
OGBench**. The current repo was ported from td_jepa but accumulated several mathematical
divergences that prevent learning. This cleanup fixes the math, restores the
architectural pieces that were dropped (left_encoder, obs_normalizer wiring), aligns
defaults with td_jepa's OGBench launch config, and removes scaffolding that was added
speculatively (Factored-FB extension hook, test suite).

Goal: `python train.py domain=antmaze-medium-navigate-v0` and
`python train.py domain=cube-single-play-v0 agent=fb_flowbc` reproduce td_jepa-style
results on state-based OGBench, approaching the FB rows of `expected_results.md`.

We **keep** the existing Hydra/YAML config surface and `train.py` loop shape — this is a
math/architecture cleanup, not a config-system rewrite.

---

## 2. Success criterion

Runs should approach td_jepa's reported state-based OGBench FB numbers
(from `expected_results.md`, "OGBench (avg)" rows):

| Domain          | Target success |
|-----------------|----------------|
| antmaze-mn      | ~73            |
| antmaze-ln      | ~37            |
| antmaze-ms      | ~70            |
| antmaze-ls      | ~50            |
| antmaze-me      | ~52            |
| cube-single     | ~50            |
| cube-double     | ~3             |
| scene           | ~13            |
| puzzle-3x3      | ~5             |

Manipulation rows use FlowBC; antmaze rows use standard FB.

---

## 3. Bugs being fixed

The eight correctness divergences against td_jepa, in priority order:

1. **FB loss formulation.** Current code does Q-style MSE on the diagonal only plus a
   squared off-diagonal penalty. Correct formulation is full M-matrix TD with a
   **negative** mean-diagonal contrastive term:
   `diff = M − γ · target_M` over the full `[E, B, B]` tensor;
   `fb_diag = −mean(diag(diff)) * E`, `fb_offdiag = 0.5·sum((diff·mask)²)/sum(mask)`.
2. **Actor Q is wrong.** Current uses `Q = F · B(obs)`. Correct: `Q = F · z`.
3. **Ortho loss is wrong shape.** Current regresses `Bᵀ B` to identity in `z×z`.
   Correct: contrastive `B Bᵀ` in `batch×batch` with negative diagonal mean.
4. **Reward inference at eval is wrong.** Current does regularized least-squares on
   `B(obs)` with `|r|>1e-6` filter (drops almost everything in sparse-reward tasks).
   Correct: `z = Σ r · B(next_obs)` then `project_z`.
5. **Relabel uses observation, not physics.** Reward functions are written for `qpos`
   (full physics state). Using `observation` gives wrong rewards for cube/scene/puzzle.
6. **Broken loader import.** `train.py` imports `load_ogbench_dataset` but the file
   exports `load_ogbench_data`. Code does not run as-is.
7. **FlowBC actor is the wrong mechanism.** Current is a `NoiseConditionedActor + MSE-BC`.
   Correct: flow-matching velocity field (`_actor_vf`) trained on FM loss;
   the noise-conditioned actor distills the FM ODE rollout.
8. **Mixed-z sampling.** Current packs `B(next_obs[:N])` at fixed positions without
   projection. Correct: shuffle next_obs, pass through `B` and `project_z`, mask
   per-sample with `train_goal_ratio`.

---

## 4. File-level changes

### 4.1 Add

- `agents/fb/flow_bc/model.py` — `FBFlowBCModel` subclass of `FBModel`. Builds an
  additional `_actor_vf` (flow-matching velocity field) network.
- Inside `agents/fb/model.py`: `_left_encoder`, `_obs_normalizer`, `_bw_encoder`,
  `_fw_encoder` modules wired into the call chain.
- Inside `nn_models.py`: `IdentityNNConfig` and `IdentityNN` (passthrough module with
  `output_space` property), needed for the encoder/normalizer slots when running on
  state observations.

### 4.2 Modify

- **`agents/fb/agent.py`** — replace `_fb_loss`, `_ortho_loss`, `_actor_loss`, and the
  `mixed_z` call with the td_jepa formulations (see §5). Add pairwise-L1 pessimism
  helper. Optimizer groups become three: forward (`_forward_map + _left_encoder`),
  backward (`_backward_map`), actor (`_actor`).
- **`agents/fb/model.py`** — wire encoders/normalizer. Fix `infer_z` /
  `reward_inference` to use td_jepa's simple linear estimator. Add
  `actor_encode_obs` switch on `FBModelConfig`. Add `_target_left_encoder` and
  soft-update it alongside F and B targets.
- **`agents/fb/flow_bc/agent.py`** — replace with td_jepa's flow-matching trainer:
  FM loss on `_actor_vf`, RL loss on `_actor` with `Q = F·z`, distillation BC loss
  from the FM ODE rollout. Drop the "argmax-Q over N samples" eval heuristic.
- **`data/ogbench.py`** — rename `load_ogbench_data` → `load_ogbench_dataset`. Place
  `physics` under `batch["next"]["physics"]` with `phys[1:]` slicing per episode.
- **`evals/ogbench.py`** — relabel on `batch["next"]["physics"]`. Drop the
  `|r|>1e-6` filter. Default `relabel_size` to 10_000. Add `shift_reward` field
  (default 1.0). Call `agent.model.reward_inference(next_obs=..., reward=...)`.
- **`configs/train.yaml`, `configs/agent/*.yaml`, `configs/domain/*.yaml`** — bump
  defaults to td_jepa's OGBench launch values (see §7).
- **`nn_models.py`** — confirm or add `VectorField` config matching
  `actor_vf: {hidden_dim: 512, hidden_layers: 4}`; verify the existing
  `SimpleVectorFieldArchiConfig` builds the right module.
- **`train.py`** — wire renamed loader, agent build via the updated configs,
  expose `actor_encode_obs` plumbing.

### 4.3 Delete

- `tests/` (entire directory).
- README section "Extending to Factored-FB" and any speculative scaffolding for
  subclass extension.
- `configs/experiment/smoke_test.yaml` if it references removed fields after the
  config rewrite. Otherwise keep.
- Any `__pycache__` clutter in commits.

### 4.4 Keep as-is

- `envs/ogbench.py`, `envs/rollout.py`, `envs/wrappers.py`, `envs/gym_spaces.py`
- `buffers/transition.py` (DictBuffer)
- `normalizers.py`, `base_config.py`, `base_model.py`, `utils.py`
- `agents/base.py`
- Overall Hydra layout in `configs/`

---

## 5. Math — FB agent (`agents/fb/agent.py`)

### 5.1 `_fb_loss`

Full M-matrix TD with contrastive diagonal. Pseudocode below assumes the top of
`update()` has already done what td_jepa's `aug` + `enc` do — namely
`obs = _fw_encoder(_normalize(obs))`, `next_obs = _fw_encoder(_normalize(next_obs))`,
and `goal = _bw_encoder(_normalize(next_obs))`. For state-based OGBench all three
of `_obs_normalizer`, `_fw_encoder`, `_bw_encoder` are Identity, so this is a no-op
in practice but the call-chain matches td_jepa for future pixel work.

Also: `discount` arrives at the loss as shape `[B, 1]` (= γ · `~terminated`, with
`terminated` shaped `[B, 1]` from the buffer). This is critical — broadcast against
`target_M` of shape `[B, B]` must apply the discount per-row (per source index `i`),
not per-column. Verify shape at the top of `_fb_loss`.

```python
with torch.no_grad():
    next_left = model._target_left_encoder(next_obs)
    actor_in  = next_left if cfg.actor_encode_obs else next_obs
    next_act  = model._actor(actor_in, z, actor_std).sample(clip=stddev_clip)
    target_Fs = model._target_forward_map(next_left, z, next_act)   # [E, B, z_dim]
    target_B  = model._target_backward_map(goal)                    # [B, z_dim]; goal = _bw_encoder(_normalize(next_obs))
    target_Ms = target_Fs @ target_B.T                              # [E, B, B]
    _, _, target_M = pairwise_l1_pessimism(target_Ms, fb_pessimism) # [B, B]

left  = model._left_encoder(obs)
Fs    = model._forward_map(left, z, action)                         # [E, B, z_dim]
B_out = model._backward_map(goal)                                   # [B, z_dim]
Ms    = Fs @ B_out.T                                                # [E, B, B]

diff       = Ms - discount * target_M                               # discount=γ·(1−done) per-sample, broadcast over B,B
fb_offdiag = 0.5 * (diff * off_diag_mask).pow(2).sum() / off_diag_sum
fb_diag    = -torch.diagonal(diff, dim1=1, dim2=2).mean() * E
fb_loss    = fb_offdiag + fb_diag
```

- `off_diag_mask = 1 - I_B`, precomputed once at `__init__` against `cfg.batch_size`.
  This couples batch size to a fixed value (same constraint as td_jepa).
- The negative diagonal mean is the contrastive trick from the FB paper; constrained
  by the off-diag squared term it pulls `M_ii` toward correct successor-feature scale.

### 5.2 `_ortho_loss`

```python
Cov               = B_out @ B_out.T                          # [B, B], NOT B.T @ B
orth_loss_diag    = -Cov.diag().mean()
orth_loss_offdiag = 0.5 * (Cov * off_diag_mask).pow(2).sum() / off_diag_sum
ortho_loss        = orth_loss_offdiag + orth_loss_diag
```

Added into `fb_loss` weighted by `ortho_coef`.

### 5.3 `_actor_loss`

```python
with torch.no_grad():
    left = model._left_encoder(obs)
actor_in    = left if cfg.actor_encode_obs else obs
dist        = model._actor(actor_in, z, actor_std)
actor_act   = dist.sample(clip=stddev_clip)
Fs          = model._forward_map(left, z, actor_act)
Qs          = (Fs * z).sum(-1)                                # [E, B]
_, _, Q     = pairwise_l1_pessimism(Qs, actor_pessimism)
actor_loss  = -Q.mean()
```

### 5.4 `pairwise_l1_pessimism`

Port `get_targets_uncertainty` from td_jepa:

```python
def pairwise_l1_pessimism(preds, penalty):
    # preds: [E, ...]
    mean = preds.mean(dim=0)
    diffs = (preds.unsqueeze(0) - preds.unsqueeze(1)).abs()    # [E, E, ...]
    scaling = preds.shape[0] ** 2 - preds.shape[0]
    unc = diffs.sum(dim=(0, 1)) / scaling
    return mean, unc, mean - penalty * unc
```

### 5.5 `mixed_z` (lives on `FBModel`)

```python
z       = self.sample_z(B)                                # [B, z_dim], ||·||=√z_dim
perm    = torch.randperm(B)
goals   = self.project_z(self._backward_map(next_obs[perm]))
mask    = (torch.rand((B, 1)) < train_goal_ratio)
z       = torch.where(mask, goals, z)
```

### 5.6 `reward_inference` (lives on `FBModel`)

```python
@torch.no_grad()
def reward_inference(self, next_obs, reward):
    B = self.backward_map(next_obs)        # passes through normalizer + bw_encoder
    z = (reward.T @ B).squeeze(0)          # [z_dim]
    return self.project_z(z.unsqueeze(0)).squeeze(0)
```

No least-squares solve, no regularization, no zero-reward filter.

---

## 6. Architecture — `FBModel` call chain

```
obs ──► _obs_normalizer ──► _fw_encoder ──► _left_encoder ──► _forward_map
                          └► _bw_encoder ──► _backward_map
                          └► (if actor_encode_obs: _left_encoder else: passthrough) ──► _actor
```

- `_obs_normalizer` — `IdentityNormalizerConfig` for state OGBench. Wired in so the
  call chain matches td_jepa.
- `_fw_encoder`, `_bw_encoder` — `IdentityNNConfig` for state OGBench. Each module
  must advertise an `output_space` property returning the (passthrough) gym Box, so
  downstream modules can size their input layer.
- `_left_encoder` — `BackwardArchi` MLP. Defaults for OGBench launch:
  `hidden_dim=512`, `hidden_layers=4`, `norm=True`, `L_dim=50`.
- Targets: `_target_forward_map`, `_target_backward_map`, `_target_left_encoder`.
  All three soft-updated with `f_target_tau` / `b_target_tau` (left_encoder uses
  `f_target_tau`).

`actor_encode_obs: false` matches td_jepa's OGBench FB launch — the actor sees raw
(normalized, fw-encoded) obs, not the left-encoded representation.

---

## 7. Config defaults

### 7.1 `configs/agent/fb.yaml` (and base values in `train.yaml`)

```yaml
hidden_dim: 512
z_dim: 50
L_dim: 50
batch_size: 256
discount: 0.99
ortho_coef: 1.0
f_target_tau: 0.005           # td_jepa launch overrides default 0.01 → 0.005
b_target_tau: 0.005
fb_pessimism_penalty: 0.0
actor_pessimism_penalty: 0.0
stddev_clip: 0.3
actor_std: 0.2
train_goal_ratio: 0.5
weight_decay: 0.0
clip_grad_norm: 0.0
actor_encode_obs: false

f:            {hidden_dim: 512, hidden_layers: 2, num_parallel: 2}
b:            {hidden_dim: 512, hidden_layers: 4, norm: true}
left_encoder: {hidden_dim: 512, hidden_layers: 4, norm: true}
actor:        {hidden_dim: 512, hidden_layers: 2}
```

### 7.2 `configs/agent/fb_flowbc.yaml` (overrides on fb.yaml)

```yaml
flow_steps: 10
lr_actor_vf: 3.0e-4
bc_coeff: 0.3                              # antmaze default; manipulation overrides to 3.0
actor:    {name: noise_conditioned_actor, hidden_dim: 512, hidden_layers: 2}
actor_vf: {hidden_dim: 512, hidden_layers: 4}
```

### 7.3 Per-domain overrides (`configs/domain/`)

- `antmaze_medium.yaml`, `antmaze_large.yaml`, etc.: `agent=fb`, `ortho_coef`
  default 1.0 (sweep range 100/1000), `lr_b` default 1e-4 (sweep 1e-4/1e-5).
- `cube_single.yaml`, `cube_double.yaml`, `scene.yaml`, `puzzle_3x3.yaml`:
  `agent=fb_flowbc`, `bc_coeff=3.0`, `ortho_coef` default 1.0 (sweep 100/1000),
  `lr_b` default 1e-4 (sweep 1e-4/1e-5).

---

## 8. Data and eval pipeline

### 8.1 `data/ogbench.py::load_ogbench_dataset`

- Rename the function so `train.py`'s import resolves.
- Per-episode slicing: `obs[:-1]`, `act[:-1]`, `obs[1:]` for `next.observation`,
  `disc[1:]==0` for `next.terminated`, **`phys[1:]` for `next.physics`** (reward
  functions read the *next-state* physics, since reward is defined on the state we
  transitioned into).

### 8.2 `evals/ogbench.py::OGBenchEvaluator`

- `relabel_size: int = 10_000` (configurable).
- `shift_reward: float = 1.0`.
- Per task, per eval call:
  ```python
  batch     = self.offline_buffer.sample(relabel_size)
  rewards   = relabel_fn(batch["next"]["physics"].cpu().numpy(), batch["action"].cpu().numpy())
  rewards  += shift_reward
  rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
  z         = agent.model.reward_inference(
                  next_obs=batch["next"]["observation"].to(device),
                  reward=rewards_t,
              )
  st, infos = rollout(env, agent, n_episodes=n_episodes, ctx=z)
  ```
- No zero-reward filter; the linear estimator handles sparse rewards correctly when
  combined with `shift_reward=1`.

---

## 9. FlowBC trainer (`agents/fb/flow_bc/agent.py`)

Three loss terms per actor update step:

```python
# 1. Flow-matching loss (trains _actor_vf)
x_1 = action
x_0 = torch.randn_like(x_1)
t   = torch.rand((B, 1), device=action.device)
x_t = (1 - t) * x_0 + t * x_1
vel = x_1 - x_0
bc_flow_loss = (model._actor_vf(obs, x_t, t) - vel).pow(2).mean()

# 2. RL loss (trains _actor)
with torch.no_grad():
    left = model._left_encoder(obs)
actor_in  = left if cfg.actor_encode_obs else obs
noises    = torch.randn_like(action)
a_actor   = model._actor(actor_in, z, noises)
Fs        = model._forward_map(left, z, a_actor)
Qs        = (Fs * z).sum(-1)                                  # [E, B]
_, _, Q   = pairwise_l1_pessimism(Qs, actor_pessimism)
rl_loss   = -Q.mean()

# 3. Distillation BC (trains _actor against FM rollout)
bc_error = torch.tensor(0.0, device=action.device)
if cfg.bc_coeff > 0:
    with torch.no_grad():
        target = compute_flow_actions(obs, noises)            # ODE rollout
    bc_error = (a_actor - target).pow(2).mean()
    bc_loss  = cfg.bc_coeff * bc_error
    actor_loss = (rl_loss / Qs.abs().mean().detach()) + bc_loss
else:
    actor_loss = rl_loss

actor_loss = actor_loss + bc_flow_loss
```

`compute_flow_actions` is `flow_steps` Euler steps integrating `_actor_vf(obs, x, t)`
from `x = noises`, clamped to `[-1, 1]` at the end.

Eval (`act`) — sample a single noise, run `_actor(obs, z, noise)`. No
test-time argmax-Q search.

Optimizer: add `actor_vf_optimizer` (Adam, `lr_actor_vf=3e-4`) alongside the
existing actor optimizer.

---

## 10. Out of scope

Not in this cleanup:

- DMC support (envs, eval, tasks).
- Pixel observations (`DrQEncoder`, `Augmentator`, pixel rollout path).
- Non-FB agents (`byol`, `hilp`, `icvf`, `laplacian`, `rldp`, `sf`, `td3`, `td_jepa`).
- `torch.compile`, `cudagraphs`, AMP autocast.
- Parallel replay buffer.
- `exca` launcher, slurm, `tyro` CLI.
- Safetensors / Pydantic-driven checkpoint format.
- CSV logger.
- `q_loss_coef` auxiliary Q loss (td_jepa supports it; its OGBench launch uses 0).
- Online-rollout config (`use_mix_rollout`, `update_z_every_step`, `z_buffer_size`).
- Multi-seed sweep launcher (Hydra `--multirun` is sufficient).
- Test suite (deleted).

---

## 11. Risks and sharp edges

- **`f_target_tau` value.** td_jepa's `FBAgentTrainConfig` default is `0.01`, but its
  OGBench launch overrides to `0.005`. Use `0.005`.
- **Batch-size coupling.** The precomputed `off_diag_mask` ties batch size to
  `cfg.batch_size`. Same constraint as td_jepa; non-matching batches will error.
- **`actor_std` in FlowBC.** Unused by `NoiseConditionedActor`. Keep the field on
  `FBModelConfig` for API uniformity; FlowBC's actor ignores it.
- **Episode boundaries.** `disc[1:]==0` defines `next.terminated`. Must align with
  `obs[1:]` for `next.observation` and `phys[1:]` for `next.physics`. Off-by-one
  here corrupts every transition.
- **`infer_z` numerical scale.** Magnitude of the linear estimator is governed by
  reward × `B` magnitude; `project_z` re-normalizes direction to `||z||=√z_dim`.
  With `shift_reward=1` on sparse rewards, every state's `B(s_next)` contributes,
  and the success-state mass biases `z` toward goal-like directions. Intended.
- **Checkpoint incompatibility.** State-dict layout changes (new `_left_encoder`
  params, renamed maps). Existing `checkpoints/*.pt` will not load. Acceptable —
  nothing has trained successfully yet.
- **README rewrite.** Current README documents the broken behavior (LS reward
  inference, MSE-BC FlowBC, "Extending to Factored-FB" hook). All three sections
  need to change. Implementation plan should include README as a deliverable.
