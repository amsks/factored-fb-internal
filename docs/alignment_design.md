# Alignment design: Factored-FB ⇄ rlmini-envs ⇄ new envs/collectors

**Branch:** `tl-alignment` (umbrella) → `cc-rlmini-envs` (env-support sub-branch)
**Status:** design (no algorithm code touched yet)
**Goal:** one training/eval pipeline in Factored-FB style, one env interface in rlmini-envs
style, one on-disk data format — so OGBench, rlmini-envs, Metaworld, and ManiSkill all flow
through the *same* agents (FB / FB+FlowBC / RLDP / TD-MPC2 / GC baselines) unchanged.

---

## 1. The seam: where Factored-FB is coupled to OGBench today

Only **four** touchpoints in Factored-FB know about OGBench. Everything above them (agents,
`nn_models`, `buffers/transition.DictBuffer`, `train.py`'s loop, checkpointing) is already
env-agnostic. Align these four and the rest comes for free.

| # | Touchpoint | File | OGBench assumption | rlmini-envs equivalent |
|---|---|---|---|---|
| 1 | env creation | `envs/ogbench.py::create_ogbench_env` | `make_env_and_datasets`; flat-vector obs | `rlmini_envs.make(env, variant, …)` → dict obs `{raw_state, flat_state, factored_state}` |
| 2 | rollout obs | `envs/rollout.py::rollout` | `obs` is a flat `np.ndarray`; `info["success"]` | adapter flattens dict → vector; `info["success"]` already present (Metaworld) |
| 3 | buffer schema | `data/ogbench.py` + `buffers/transition.py` | `next/physics` (qpos) for relabeling | `achieved_goal` from `get_achieved_goal_state` / factored `Goal` factor |
| 4 | relabel + eval | `evals/ogbench.py` + `envs/ogbench.py::get_relabel_fn` | per-domain `reward_fn(qpos, action, target)` | generic goal-distance reward on `achieved_goal`; `goal_space`, `goal_based` |

Key insight: **rlmini's `factored_state` is strictly richer than OGBench's `physics`.** OGBench
hand-writes `cube_reward_fn(qpos[...,14:17], …)` with magic slices; rlmini already names those
factors (`Object1`, `Goal`, …) and exposes `object_sizes`, `object_range`,
`get_achieved_goal_state`, `goal_space`. So the alignment *removes* magic slices rather than
adding a parallel hack — and it hands the FB-cube-failure analysis (interaction variables) a
first-class structured signal.

---

## 2. Target architecture

```
                    ┌─────────────────────────── agents/ (UNCHANGED) ───────────────────────────┐
                    │  FBAgent · FBFlowBCAgent · RLDPAgent · TDMPC2Agent · (GCIQL/CRL via JAX)  │
                    └───────────────▲───────────────────────────────▲──────────────────────────┘
                                    │ batch (DictBuffer)             │ act(obs_vec, z|goal)
                    ┌───────────────┴───────────────┐   ┌────────────┴───────────────┐
                    │ data/ (unified loader)        │   │ evals/ (goal evaluator)     │
                    │  load_dataset → DictBuffer    │   │  GoalEvaluator: relabel via │
                    │  next/achieved_goal (was phys)│   │  achieved_goal + goal_space │
                    └───────────────▲───────────────┘   └────────────▲───────────────┘
                                    │ unified trajectories (.npz + meta.json)      │ EnvAdapter
        ┌───────────────────────────┴───────────────┐          ┌──────────────────┴──────────────────┐
        │ collectors/ (NEW)                          │          │ envs/registry.py  (NEW dispatch)     │
        │  TrajectoryCollector: policy→trajectories  │          │  ogbench: | rlmini: | metaworld: |   │
        │  TDMPC2Collector · Random · Scripted       │          │  maniskill:  → one EnvAdapter proto  │
        └───────────────▲────────────────────────────┘          └──────────────────▲──────────────────┘
                        │ rolls out in                                              │ wraps
                        └───────────────── EnvAdapter (one protocol) ───────────────┘
                                            │        │         │          │
                                        OGBench   rlmini   Metaworld   ManiSkill
                                       (current) (via make) (rlmini)   (direct adapter)
```

One new **`EnvAdapter`** protocol is the spine. Everything env-specific lives behind it; agents
and the training loop never see an env family.

---

## 3. Component-by-component alignment

### 3a. Environment interface — `EnvAdapter` (new, rlmini-shaped)

The minimal contract the rollout + evaluator need. rlmini already provides all of it; OGBench is
retrofitted; Metaworld/ManiSkill implement it.

```python
# envs/adapter.py  (NEW)
class EnvAdapter(Protocol):
    observation_space: gym.spaces.Box      # FLAT policy obs (post FactoredObservationWrapper)
    action_space:      gym.spaces.Box
    goal_space:        gym.spaces.Box | None
    factor_schema:     dict                 # {name: size} from object_sizes (for meta.json + analysis)

    def reset(self, seed=None) -> tuple[np.ndarray, dict]: ...          # (obs_vec, info)
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        # info MUST carry: "success" (bool), "achieved_goal" (np[g_dim]),
        #                  optional "factored_state" (dict)
    def render(self) -> np.ndarray: ...
    def achieved_goal(self, *, obs=None, factored=None) -> np.ndarray:  # goal-relevant slice
    def set_task(self, task) -> None:       # rlmini variant / OGBench singletask / MW task
```

- `envs/rlmini.py::make_rlmini_env(name, variant, obs_keys, …)` = thin wrapper over
  `rlmini_envs.make(...)` with `obs_flat=True` (uses `FactoredObservationWrapper` → flat Box).
  It stashes `factored_state` and `achieved_goal = get_achieved_goal_state(state)` into `info`
  every step. rlmini's `GymnasiumWrapper.step` already returns the gymnasium 5-tuple, so
  `envs/rollout.py` needs **zero changes** once obs is flat.
- `envs/ogbench.py` gets a small `OGBenchAdapter` shim: `achieved_goal = physics` (or the
  goal-relevant slice), `set_task` = pick `singletask-taskN`. The existing reward fns move behind
  the generic relabeler (§3c) as one registered special case.
- `envs/registry.py::make_env(cfg)` dispatches on a `env_family` prefix in the domain string:
  `"ogbench:cube-single-play-v0"`, `"rlmini:Metaworld/pick-place"`,
  `"maniskill:PickCube-v1"`. Default (bare name) = `ogbench:` for back-compat.

### 3b. Dataflow / data format — one trajectory schema (superset)

**On disk**, per-episode (keeps OGBench's `<root>/<domain>/buffer/episode_*.npz` layout so the
existing loader path and `/dev/shm` habits carry over), plus a dataset-level `meta.json`:

```
episode_000123_00500.npz
  observation     [T, obs_dim]   float32   # flat policy obs
  action          [T, act_dim]   float32
  achieved_goal   [T, g_dim]     float32   # GENERALIZES OGBench "physics"
  reward          [T]            float32   # env/task reward (0 for reward-free play)
  terminal        [T]            bool
  timeout         [T]            bool       # truncated
  success         [T]            bool       # info["success"] when available
  # optional structured block (object-centric; enables the FB interaction analysis):
  factored/<Name> [T, size]      float32   # e.g. factored/Object1, factored/Goal

meta.json
  { env_family, env_name, variant, obs_dim, act_dim, goal_dim,
    factor_names, factor_sizes, achieved_goal_factors: ["Object1","Goal"],
    goal_low, goal_high, success_eps, horizon,
    collector: {policy:"tdmpc2", ckpt, exploration_noise}, git_sha }
```

- `data/unified.py::load_dataset(root, ...) -> DictBuffer` replaces the OGBench-specific loader.
  It maps episodes → transitions and fills `next/achieved_goal` (was `next/physics`).
  **`buffers/transition.DictBuffer` is unchanged** — it already stores arbitrary nested dicts.
- `data/ogbench.py::load_ogbench_dataset` becomes a thin call into `load_dataset` with an
  `physics→achieved_goal` rename shim, so **all existing FB/OGBench runs keep working**.
- Rename discipline: everywhere the code reads `batch["next"]["physics"]` (evaluator, TD-MPC2
  agent, `scripts/data/cube_density.py`) it reads `batch["next"]["achieved_goal"]`. The OGBench
  loader writes both keys during a deprecation window.

### 3c. Relabeling + evaluation — generic goal relabeler

Today: `get_relabel_fn(domain, task)` returns one of 4 hand-written `reward_fn(qpos, action,
target)`. Replace with one generic function + a per-env registration:

```python
# evals/relabel.py  (NEW)
def goal_distance_reward(achieved_goal, action, *, target, eps, proj=None):
    ag = achieved_goal[..., proj] if proj else achieved_goal
    d  = np.linalg.norm(ag - target, axis=-1, keepdims=True)
    return (d < eps).astype(np.float32) - 1.0        # sparse {-1, 0}, matches OGBench convention
```

- Target goals come from **hindsight over the offline data** (sample achieved_goal of reached
  states) or from the env's `goal_space`. `eps`, `proj` (which achieved_goal dims to compare)
  live in `meta.json`, not in code.
- `evals/goal_eval.py::GoalEvaluator` generalizes `OGBenchEvaluator`:
  - **FB / FB+FlowBC**: unchanged inference — relabel → `reward_inference(next_obs, reward)` → z
    → rollout. The only change is `next.physics` → `next.achieved_goal`.
  - **GC agents (GCIQL/CRL)**: need an explicit goal observation; source it as the highest-reward
    `next_obs` (already implemented in `_goal_observation`, just generalized off `achieved_goal`).
  - Success = `info["success"]` (rlmini/Metaworld give it directly; OGBench already does).
- The 4 OGBench reward fns become registered entries `{domain: (proj_slice, eps)}` — kept for
  exact-repro, but no longer the only path.

### 3d. Training pipeline — `train.py` stays, calls get dispatched

`train.py` currently hard-calls `create_ogbench_env`, `load_ogbench_dataset`, `OGBenchEvaluator`.
Change those three lines to registry factories keyed by `cfg.env_family`; **agents, optimizers,
logging, checkpointing untouched.** New Hydra config group `configs/env/*.yaml` (parallel to
`configs/domain/*.yaml`) carries `{env_family, env_name, variant, obs_keys, data_root,
success_eps, goal_factors}`. Existing `domain/*.yaml` keep working (default `env_family=ogbench`).

### 3e. Data collection — `collectors/` with TD-MPC2 (new)

```python
# collectors/base.py  (NEW)
class TrajectoryCollector:
    def __init__(self, env: EnvAdapter, policy, recorder): ...
    def collect(self, n_episodes, exploration) -> None:   # writes unified .npz + meta.json
```

- `collectors/tdmpc2_collector.py`: wraps the **generic** TD-MPC2 core
  (`agents/tdmpc2/core.py`, "algorithmically identical to upstream") — *not* the OGBench-cube
  `TDMPC2Agent` wrapper, which is coupled to `physics[...,14:17]`. It online-trains on the env's
  own dense reward (Metaworld/ManiSkill both provide one), then rolls out with planning +
  exploration noise to produce diverse "play"-style offline data. Also ships `RandomCollector`
  and `ScriptedCollector` (Metaworld has oracle policies) for baselines/mixtures.
- New env families for collection:
  - **Metaworld**: already in rlmini (`rlmini:Metaworld/<task>`) → reuse the rlmini adapter.
  - **ManiSkill**: not in rlmini. Add a direct `envs/maniskill.py::ManiSkillAdapter` implementing
    the same `EnvAdapter` protocol (ManiSkill is gymnasium-native with structured obs +
    `info["success"]`). Optionally contribute a `ManiSkill` env back to rlmini later so it also
    gets a factored view — but the adapter unblocks us without that.
- Output feeds straight into `data/unified.load_dataset` — collection and training share one
  format, so "collect with TD-MPC2, train FB on it" is two commands.

---

## 4. Branch & phasing plan

- **`tl-alignment`** (this branch) — umbrella: this doc + the `EnvAdapter` protocol +
  `envs/registry.py` + the `train.py` dispatch refactor. OGBench path stays byte-identical
  (guarded by the existing `tests/`), so nothing regresses.
- **`cc-rlmini-envs`** — branched from `tl-alignment`: `envs/rlmini.py`, rlmini registry entries,
  `data/unified.py`, `evals/goal_eval.py` + `evals/relabel.py`, a rlmini smoke. PR → `tl-alignment`.
- Follow-on branches off `tl-alignment`: `tl-collectors-tdmpc2`, `tl-maniskill`.

| Phase | Branch | Deliverable | Smoke gate |
|---|---|---|---|
| P0 | tl-alignment | `EnvAdapter`, `envs/registry.py`, train.py dispatch, this doc | existing OGBench FB smoke + tests still green |
| P1 | cc-rlmini-envs | rlmini adapter + unified loader + goal evaluator | FB trains N steps on `rlmini:Metaworld/reach` from a tiny collected set; eval logs `success` |
| P2 | tl-collectors-tdmpc2 | `collectors/` + TD-MPC2 collector + Metaworld wiring | TD-MPC2 collects play data on Metaworld; FB trains on it end-to-end |
| P3 | tl-maniskill | `ManiSkillAdapter` + ManiSkill TD-MPC2 collection | same end-to-end on a ManiSkill task; unified wandb metrics |

Each phase keeps the two venvs we already built (`.venv` PyTorch, `/dev/shm/.venv-jax` JAX);
rlmini/Metaworld/ManiSkill install into `.venv` as extras.

---

## 5. Decisions (LOCKED 2026-07-09)

1. **On-disk format** → **per-episode `.npz` + `meta.json`.** Keep OGBench's
   `<root>/<domain>/buffer/episode_*.npz` layout; existing loader path and `/dev/shm` workflow
   carry over. (h5py chunked deferred; revisit only if a collection outgrows npz.)
2. **ManiSkill** → **direct `envs/maniskill.py::ManiSkillAdapter`** implementing the `EnvAdapter`
   protocol. No upstream rlmini change required to unblock collection.
3. **Scope** → **PyTorch FB family only** (FB / FB+FlowBC / RLDP / TD-MPC2). JAX GCIQL/CRL stay
   on their native OGBench path; porting `impls/main.py` is a separate later effort.
4. **Relabel semantics** → **share the data layer only.** Unify `achieved_goal` storage +
   loading, but preserve FB's reward-inference-`z` path and GC's explicit-goal path exactly as
   today — protects the reproduced numbers (FB 0.47 / GCIQL 0.76 / CRL 0.15 / CRL+FlowBC 0.26).

## 6. P0 concrete task list (branch `tl-alignment`)

Order chosen so the OGBench path stays byte-identical at every commit (existing `tests/` are the
regression gate). Nothing here touches `agents/` or `nn_models.py`.

1. `envs/adapter.py` — `EnvAdapter` Protocol + a tiny `AdapterEnv` base (flat obs, `info` carries
   `success`/`achieved_goal`/`factored_state`).
2. `envs/registry.py` — `make_env(cfg)` + `family, name = parse_domain(cfg.domain)`; register
   `ogbench:` → wrap current `create_ogbench_env` in an `OGBenchAdapter`
   (`achieved_goal := physics`). Bare names default to `ogbench:`.
3. `data/unified.py` — `load_dataset(root, …) -> DictBuffer` writing `next/achieved_goal`;
   `write_episode(path, traj, meta)` + `meta.json` I/O. `data/ogbench.py::load_ogbench_dataset`
   becomes a shim that also emits `next/physics` (deprecation alias) so eval + TD-MPC2 keep working.
4. `evals/relabel.py` — `goal_distance_reward(...)` + a registry of the 4 OGBench special cases
   (`{domain: (proj_slice, eps)}`) for exact repro.
5. `evals/goal_eval.py` — `GoalEvaluator` = `OGBenchEvaluator` with `physics`→`achieved_goal` and
   relabel via `evals/relabel.py`; FB/GC branches unchanged. Keep `OGBenchEvaluator` as a subclass
   alias.
6. `train.py` — replace the 3 construction calls with `envs/registry` + `data/unified` +
   `evals/goal_eval` dispatch on `cfg.env_family` (default `ogbench`). Add `configs/env/*.yaml`
   group; existing `configs/domain/*.yaml` unchanged.
7. `tests/test_alignment_ogbench_parity.py` — assert the `ogbench:` path produces identical
   batches/eval keys to the pre-refactor loader on a tiny fixture; run existing FB smoke.

**P0 exit gate:** `make smoke` + the OGBench FB smoke + full `tests/` green; a 2k-step
`ogbench:cube-single-play-v0` FB run logs the same `eval/success` shape as before.
Then branch `cc-rlmini-envs` for P1 (rlmini adapter + Metaworld).

---

## 7. Revised env architecture (LOCKED 2026-07-09, round 2)

Supersedes §3a's per-family-adapters-in-Factored-FB sketch. **Env definitions live in the rlmini
repo; Factored-FB consumes them through one `EnvAdapter`.** Clean repo boundary:

| Concern | rlmini-envs repo | Factored-FB (`cc-rlmini-envs`) |
|---|---|---|
| Env definitions (Metaworld ✔, ManiSkill NEW, OGBench NEW/phased) | ✅ `Environment` subclasses | — |
| factored_state / `Goal.get_achieved_goal` / goal_space / success | ✅ native | consumed |
| `EnvAdapter` (single class) | — | ✅ wraps `rlmini_envs.make(...)` |
| Collector (TD-MPC2 / random / scripted) → npz+meta | — | ✅ drives a rlmini env |
| data / evals / agents | — | ✅ unchanged |

**Decisions (round 2):**
1. **`EnvAdapter` + collector live in Factored-FB** — rlmini stays pure, reusable env code. The
   adapter is the consumer-side seam (knows `DictBuffer`, npz schema, FB needs).
2. **First end-to-end env = Metaworld** — already a rlmini env, zero new env code; fastest proof
   of EnvAdapter + TD-MPC2 collector + FB-training.
3. **OGBench-into-rlmini deferred** — keep OGBench on its native path (data exists, results
   reproduced 0.47/0.76/0.15/0.26). Add `Environments/OGBench/` later and cross-check the
   rlmini-OGBench eval against the native `OGBenchEvaluator` before switching anything.

**Consequence for §3a/§3d:** `envs/registry.py` collapses to
`make_env(cfg) = EnvAdapter(rlmini_envs.make(name, variant, obs_flat=True, …))` for the
`rlmini:` family; the `ogbench:` family keeps calling `create_ogbench_env` (native) until the
deferred rlmini-OGBench port. Family dispatch mostly moves *into* rlmini's `make`.

**#1 integration risk — CHECKED & RESOLVED (2026-07-09).** Dry-run
`uv pip install --dry-run -e ../rlmini-envs[metaworld]` against the current `.venv` resolves, but
**forces `mujoco 3.10.0 → 3.3.0`** (Metaworld 3.1.1 pins mujoco 3.3). That would downgrade the
mujoco under the reproduced OGBench results. **Resolution: a separate `.venv-rlmini`** (mujoco 3.3
+ metaworld + rlmini + torch + Factored-FB), mirroring the repo's existing two-stack idiom
(`.venv` torch-FB, `.venv-jax` JAX). Pick the venv by env family:
- `.venv` (mujoco 3.10) → OGBench train+eval, untouched → reproduced numbers stay valid.
- `.venv-rlmini` (mujoco 3.3) → Metaworld/ManiSkill collection **and** train+eval.
FB agent code is pure torch → runs identically in either; npz format crosses freely. Revisit a
single shared venv only if OGBench is later confirmed working on mujoco 3.3.

## 8. Revised P1 plan (branch `cc-rlmini-envs`, off `tl-alignment` after P0)

1. **Dep dry-run** — resolve `rlmini-envs[metaworld]` against `.venv`; decide shared vs
   `.venv-rlmini`. (Gate before any code.)
2. `envs/rlmini_adapter.py::EnvAdapter` — wrap `rlmini_envs.make(...)`; expose flat obs +
   `info["success"/"achieved_goal"/"factored_state"]`, `goal_space`, `factor_schema`,
   `achieved_goal()`, `set_task(variant)`.
3. `envs/registry.py` — add `rlmini:` family → `EnvAdapter`.
4. `collectors/base.py` + `collectors/tdmpc2_collector.py` — drive a rlmini Metaworld env with the
   generic TD-MPC2 core; write unified npz + `meta.json` (`achieved_goal` from
   `Goal.get_achieved_goal`).
5. **End-to-end smoke** — collect a small Metaworld `reach`/`pick-place` set with TD-MPC2, then FB
   trains N steps on it via `data/unified.load_dataset`, eval logs `success`.

Deferred branches: `tl-maniskill` (new rlmini `ManiSkill` env + adapter reuse),
`tl-ogbench-rlmini` (OGBench-as-rlmini-env + eval-parity cross-check).
