---
name: rlmini envs integration
overview: Integrate the updated `rlmini_envs` Gymnasium/factored-observation wrapper into Factored-FB, then prove small offline datasets work for Box2D, AirHockey, and Libero.
todos:
  - id: validate-rlmini-wrapper
    content: Validate `rlmini_envs.make(..., obs_keys/select_obs, gym_to_gymnasium=True)` for Box2D, AirHockey, and Libero, including flat Box observations, continuous Box actions, reset/step signatures, render, and success info.
    status: pending
  - id: add-backend-registry
    content: Introduce an `env_family` registry and route `train.py` through it while keeping `ogbench` behavior unchanged.
    status: pending
  - id: share-episode-loader
    content: Extract the generic episode-to-`DictBuffer` logic from `data/ogbench.py` into a shared loader module.
    status: pending
  - id: harden-libero-metadata
    content: Patch `../rlmini_envs` or account for Libero missing factor metadata (`object_range`, `object_dynamics`, instancing/proximity) so it composes with `FactoredObservationWrapper` and `GymnasiumWrapper`.
    status: pending
  - id: implement-rlmini-backend
    content: Add `rlmini` env, dataset, and evaluator adapters targeting the wrapped Gymnasium environments first.
    status: pending
  - id: random-action-dataset
    content: Generate random-action rollouts for Box2D, AirHockey, and Libero into Factored-FB episode `.npz` files as the first offline dataset source.
    status: pending
  - id: policy-dataset-path
    content: Leave a clean extension point for future online CRL/policy rollouts to write the same offline episode format.
    status: pending
  - id: add-configs-and-smokes
    content: Add one `rlmini` domain config, dataset conversion smoke tests, and a CPU agent-update smoke.
    status: pending
isProject: false
---

# Integrate rlmini_envs Cleanly

## Recommendation
Use a two-repo adapter approach. Keep the environment normalization in `../rlmini_envs`, then keep this repo responsible for training, offline dataset loading, and evaluation. The updated `rlmini_envs` wrapper structure is the right direction: `rlmini_envs.make(...)` now supports observation selection (`obs_keys`, `select_obs`, `obs_slices`, `obs_flat`) and can compose `FactoredObservationWrapper` before `GymnasiumWrapper`.

Structurally this should work for Box2D, and likely AirHockey's Box2D backend, because both expose continuous `Box` actions plus factor metadata (`all_names`, `object_sizes`, `object_range`, `object_dynamics`). Libero needs one more compatibility check: it exposes continuous actions and factored state, but currently appears to define `object_sizes` without matching `object_range` / `object_dynamics`. Either Libero should add those metadata fields or the wrappers should provide safe defaults.

Today `train.py` is directly coupled to OGBench via `[data/ogbench.py](data/ogbench.py)`, `[envs/ogbench.py](envs/ogbench.py)`, and `[evals/ogbench.py](evals/ogbench.py)`. The effective transfer setup is to introduce a tiny `env_family` switch, keep `ogbench` as the default backend, and add `rlmini` as a parallel backend that consumes the wrapped Gymnasium environments.

## Proposed Shape
- In `../rlmini_envs`, finish hardening the generic wrapper path:
  - Use `rlmini_envs.make(env, variant=..., gym_to_gymnasium=True, select_obs=True, obs_flat=True, ...)` as the public integration API.
  - Ensure `FactoredObservationWrapper` returns a stable flat `Box` observation and preserves enough factor metadata for debugging.
  - Ensure `GymnasiumWrapper` exposes standard `reset(seed) -> (obs, info)` and `step(action) -> (obs, reward, terminated, truncated, info)`.
  - Ensure all three target envs have continuous `Box` action spaces. If any variant is discrete, it should be excluded from the first Factored-FB pass.
  - Add wrapper tests in `../rlmini_envs` for Box2D, AirHockey, and Libero, mirroring the existing Box2D composition test.
- In this repo, add a backend registry, likely `[envs/registry.py](envs/registry.py)`, with methods for `create_env`, `load_dataset`, `make_evaluator`, `list_eval_tasks`, and optional `get_relabel_fn`.
- Move the generic transition loader out of `[data/ogbench.py](data/ogbench.py)` into something like `[data/episodes.py](data/episodes.py)`, because the `DictBuffer` contract is already general: episode files become `observation`, `action`, and `next.{observation, terminated, action, physics?}` tensors.
- Keep `[envs/ogbench.py](envs/ogbench.py)`, `[data/ogbench.py](data/ogbench.py)`, and `[evals/ogbench.py](evals/ogbench.py)` as the OGBench backend implementation so existing runs remain unchanged.
- Add parallel `rlmini` files, for example `[envs/rlmini.py](envs/rlmini.py)`, `[data/rlmini.py](data/rlmini.py)`, and `[evals/rlmini.py](evals/rlmini.py)`, translating `rlmini_envs` into the repo's existing Gymnasium/offline-buffer/evaluator expectations.
- Add `env_family: ogbench` to `[configs/train.yaml](configs/train.yaml)`, then add one or more `[configs/domain](configs/domain)` entries for `rlmini` environments.

## Data Contract To Enforce
The key transfer decision is the offline dataset format. I would standardize every `rlmini_envs` dataset into the repo's episode layout:

```text
datasets/<domain>/buffer/episode_000000_<length>.npz
  observation: [T, obs_dim]
  action: [T, act_dim]
  physics: optional [T, phys_dim] or flattened factored state mirror
  reward: [T, 1]
  discount: [T, 1]
  pixels: optional [T, H, W, C]
```

For the first transfer, store the flattened selected factored state as `observation`. Store the same vector as `physics` only as a compatibility/debug mirror unless a better environment-native physics vector exists. That gives Factored-FB a normal observation tensor and preserves enough metadata for future relabeling/diagnostics. The `rlmini` evaluator should use environment-native rewards and success info first; it should not reuse OGBench qpos relabeling or `eval_shift_reward: 1.0`.

Random-action data should be generated by a new rollout/export script that repeatedly calls the wrapper, samples `env.action_space.sample()`, and writes complete episodes. Later CRL or another online policy can call the same writer API, replacing only the action source.

## First Vertical Slice
Start with wrapped Box2D and CPU-only smoke coverage, then add AirHockey and Libero:
- In `../rlmini_envs`, run or add wrapper-contract tests for:
  - Box2D `goal-ctrl-target` or `fixed-goal-ctrl-target-128-small`.
  - AirHockey `default` using its Box2D backend first.
  - Libero `default`, after adding missing factor metadata or safe wrapper defaults.
- Install `../rlmini_envs` editable from this repo's venv during local development, not by vendoring its source into Factored-FB.
- In this repo, add domain configs such as `rlmini_box2d_goal_target`, `rlmini_air_hockey_default`, and `rlmini_libero_default`.
- Add a random-action exporter under `[scripts/data](scripts/data)` that writes tiny datasets, for example 3 episodes x 10 to 20 steps per env.
- Add tests that load each tiny dataset into `DictBuffer`, sample a batch, and run 1 to 2 `agent.update(...)` calls on CPU.
- Add a simple `rlmini` evaluator that rolls out the learned policy and reports mean return plus `success` when present in `info`.

## Test Plan
- In `../rlmini_envs`, add wrapper unit tests that instantiate each target env with `gym_to_gymnasium=True`, `select_obs=True`, and `obs_flat=True`; assert `reset` returns an `np.ndarray`, `observation_space.contains(obs)` is true after dtype normalization if needed, `action_space` is a continuous `Box`, and one random `step` returns the Gymnasium 5-tuple.
- In this repo, add a small random-data generation test that writes temporary episode `.npz` files for each env and checks required keys/shapes: `observation`, `action`, `reward`, `discount`, and optional `physics`.
- In this repo, add a loader smoke that reads those files through the shared episode loader and verifies `DictBuffer.sample(...)` returns tensors compatible with `make_agent(...)`.
- In this repo, add a CPU training smoke per env with very small overrides: `device=cpu`, `num_train_steps=1`, `batch_size` small, `eval_every=0`, `save_every` disabled or larger than the smoke.
- Gate AirHockey/Libero tests with dependency checks so base CI can still run without optional heavy extras.

## Risks To Control
- Reward relabeling is the biggest mismatch: OGBench evaluation assumes `next.physics`, task-specific reward functions, and `eval_shift_reward: 1.0`.
- Action alignment matters: `[data/ogbench.py](data/ogbench.py)` currently applies an EXORL-style shift, so the `rlmini` loader/extractor must decide explicitly whether that shift is correct for its source data.
- Discrete-action `rlmini_envs` should wait. The current FB agents assume continuous Box actions, so start with continuous Box2D and only add discrete support after the backend works.
- Dict observations should be flattened in the `rlmini_envs` wrapper, not inside agents.
- Libero likely needs metadata hardening before wrapper composition works. The safest fix is to make `FactoredObservationWrapper` provide default dynamics/instancing/proximity fields for selected factors, and/or add proper `object_range` / `object_dynamics` fields in Libero.
- AirHockey has both Box2D and robosuite-style paths. Use the Box2D backend for the first test to avoid pulling heavy robosuite rendering/dependency issues into the initial integration.
- Pixel observations should follow the existing HWC-on-disk, CHW-in-buffer pattern if visual environments are transferred.
- Leave the JAX/GCIQL path alone initially; this should target the PyTorch `train.py` stack first.
