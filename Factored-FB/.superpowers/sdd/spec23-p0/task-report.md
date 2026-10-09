# Spec 23 P0 feasibility-probe plumbing — task report

Branch `gciql-fb-bridge`. Scope: minimum plumbing for P0 (`point_mass_maze`,
RND buffer, seed 0) per `PAPER/Multistep-FB/specs/23-humanoid-eval.md`
sections 3–5. No training launched. No vendored code edited. No commits made.

## 1. Data acquisition (spec 4.1)

`git clone --depth 1 https://github.com/denisyarats/exorl` to
`~/exorl_data/exorl` (commit `a3fb07a420939280aa0918150923dcca7e82bf2a`,
2022-02-07), then `./download.sh point_mass_maze rnd` from that checkout
(downloads from `https://dl.fbaipublicfiles.com/exorl/point_mass_maze/rnd.zip`,
523,838,175 bytes, ~13s at 38.7 MB/s; unzip ~4s).

- **Location (outside the repo, per convention):**
  `~/exorl_data/exorl/datasets/point_mass_maze/rnd/buffer/` (raw episodes)
  and `~/exorl_data/exorl/datasets/point_mass_maze/rnd/video/`.
- **Episode count:** 10,000 files, `episode_000000_1000.npz` …
  `episode_009999_1000.npz`.
- **Size:** buffer dir 515 MB on disk (520,696,627 bytes summed); video dir
  5.4 MB (not used).
- **Steps:** 10,000 episodes × 1000 transitions = **10,000,000** — matches
  spec 4.1's "full 10M primary" budget exactly; no subsampling needed for P0.
- Also cloned `https://github.com/facebookresearch/controllable_agent`
  (commit `5a9950b07f1edcb4bddd04f5819379541793d975`) to `~/exorl_data/controllable_agent`
  — the official FB / "Does Zero-Shot RL Exist?" reference implementation,
  needed for the ε provenance (section 5 below), not a training dependency.

## 2. Obs-layout verification (MANDATORY, spec 4.2) — VERIFIED, matches spec's expectation exactly

Spec expected: `(x, y)` at dims `0:2` of a 4-dim obs. **Confirmed true**, by
three independent checks, not just the source-code name:

1. **Real downloaded data.** `episode_000000_1000.npz`:
   `observation` shape `(1001, 4)`, `physics` shape `(1001, 4)`, and
   `physics == observation` row-for-row (point_mass_maze's `Physics.state()`
   is exactly `[qpos, qvel]` = `[x, y, vx, vy]`). First row velocity is
   exactly `[0, 0]` (a fresh reset).
2. **Real env instantiation, from source.** Cloned exorl's
   `custom_dmc_tasks/point_mass_maze.py`: `get_observation()` returns
   `OrderedDict(position=physics.position(), velocity=physics.velocity())`,
   and `dmc.py`'s `flat_observation=True` concatenates `OrderedDict` values
   in insertion order → `[x, y, vx, vy]`.
3. **Real env instantiation, executed** (both from the original exorl
   checkout and from the file this repo now vendors — see below):
   `env.reset()` → `observation = {'position': [x, y] (2,), 'velocity': [vx, vy] (2,)}`,
   `action_spec` bounds already `[-1, 1]` (the XML's own `ctrlrange`, so
   ExoRL's `action_scale.Wrapper(-1, 1)` is an identity map for this domain).

No disagreement found anywhere — hit_spec `0:2:<eps>` is frozen as
specified.

**Vendored** (new, `third_party/exorl_dmc_tasks/`, MIT, PROVENANCE.md
included): `point_mass_maze.py` (one documented one-line patch: XML path
resolution made `__file__`-relative since this isn't installed under
exorl's own `custom_dmc_tasks` package layout) + the 4 task XMLs. The
`<include file="./common/*.xml">` directives resolve against
`dm_control.suite.common.ASSETS`, which ships byte-identical copies under
those exact keys (verified empirically) — exorl's own
`custom_dmc_tasks/common/` dir was not needed and was not vendored.

## 3. Conversion (spec 4.2) — `scripts/exorl/convert_exorl.py`

ExoRL episode schema (from `exorl/replay_buffer.py`, verified against real
files): each `episode_<idx>_<N>.npz` stores `observation`/`action` arrays
of length **N+1** — a "dummy first transition": `action[0]` is a `[0, 0]`
padding row; the N real transitions are `(observation[t], action[t+1],
observation[t+1])` for `t = 0..N-1`.

The converter reproduces this exactly and writes the repo's **non-compact**
Dataset format — `observations`, `actions`, `next_observations`,
`terminals` (all explicit, float32) — so no `next_observations` inference
from `idx+1` is ever needed at load time. This was a deliberate choice to
sidestep the off-by-one defect class already seen in this codebase (compact
datasets infer `next_observations[t] = observations[t+1]`, which is only
boundary-safe if `valids`/terminal bookkeeping is exactly right —
`third_party/ogbench/ogbench/utils.py`'s `load_dataset(compact_dataset=True)`
is the shape of that trap). Deterministic split: the **last** `val_episodes`
files (by sorted filename / collection order) become val, so early- and
late-training collection behavior aren't mixed across the split.

**P0 conversion run** (full 10,000 episodes, `val_episodes=100`):
```
n_episodes_total=10000  n_episodes_train=9900  n_episodes_val=100
n_transitions_train=9,900,000  n_transitions_val=100,000
obs_dim=4  action_dim=2
source_bytes=520,696,627  train_bytes=435,601,040  val_bytes=4,401,040
```
Wall time: **5.2s** for the full conversion.

Output (outside the repo):
```
~/exorl_data/converted/point_mass_maze_rnd_seed0/point_mass_maze_rnd.npz       (416 MB, train)
~/exorl_data/converted/point_mass_maze_rnd_seed0/point_mass_maze_rnd-val.npz   (4.2 MB, val)
~/exorl_data/converted/point_mass_maze_rnd_seed0/point_mass_maze_rnd.stats.json
```

**Round-trip + boundary tests** (`tests/test_exorl_convert.py`, 4 tests, all pass):
- `test_synthetic_round_trip_exact` — byte-exact reproduction of a
  synthetic buffer's `observation`/`action` arrays after conversion, train
  and val splits both checked.
- `test_terminal_structure_matches_episode_boundaries` — terminals land
  exactly at cumulative episode-length boundaries; additionally shells out
  to the jax venv (`tests/jax_subprocess.py`, since `GCDataset` lives under
  `third_party/ogbench/impls`, which needs jax and whose `utils` package
  name collides with the repo's own top-level `utils.py` if imported from
  `.venv`) to instantiate the REAL vendored `GCDataset` on the converted
  data and assert its recovered `terminal_locs`/`initial_locs` match.
- `test_no_cross_episode_leakage_at_boundary` — explicitly checks the
  off-by-one failure mode: a terminal row's `next_observations` is the true
  final state of its OWN episode, not the next episode's first observation.
- `test_real_episode_round_trip` — same round-trip check against the real
  downloaded `episode_000000_1000.npz` (skipped, not failed, if the dataset
  isn't present on the machine running pytest).

**Full-dataset GCDataset load smoke check** (jax venv, CPU): loaded the
real 9.9M-transition train file, constructed `GCDataset`, confirmed
`terminal_locs[-1] == size - 1`, `n_episodes == 9900`, and sampled a batch
of 256 — all fields finite, positions/actions in expected ranges
(`observations`/`next_observations` in `[-0.29, 0.29]`, `actions` in
`[-1, 1]`), no NaNs.

## 4. Eval adapter (spec 4.2/4.4) — `scripts/exorl/point_mass_maze_env.py`

Thin, standalone dm_control adapter, parallel to (imports nothing from,
edits nothing in) `third_party/ogbench/impls/utils/evaluation.py`. No
`humenv`/`metamotivo` import anywhere. Provides:
- `PointMassMazeEnv`: `reset()`/`step()` gym-like wrapper over the vendored
  task, flattening `{position, velocity}` into the verified `[x,y,vx,vy]`
  order explicitly (not relying on dict-iteration order matching).
- `rollout(env, policy, goal_xy, eps, horizon)`: runs up to `horizon`
  steps (default 1000, matching the ExoRL episode length), returns
  observations/actions/distances plus `success_any_time`/`success_final`.
- `random_policy(...)`: uniform-in-`[-1,1]` action sampler.
- `MAZE_20_GOALS`, `GOAL_EPS` — see section 5.

**Exercised on a random policy** (`tests/test_exorl_point_mass_maze_env.py`,
4 tests, all pass): reset layout sane (velocity zero, position in the
XML-randomized start range), action spec bounds `[-1,1]`, a full 1000-step
rollout has all-finite observations/distances, actions respect the action
spec, position stays within the maze walls (`|x|,|y| <= 0.35`). Also ran a
**quantitative** random-policy check (60 rollouts, 20 goals × 3 episodes
each, full 1000-step horizon): any-time success rate **1.7%** (1/60),
min-distance-to-goal mean **0.33**, median **0.35**, range
`[0.019, 0.64]`. This is the expected signature for a random walk in a
0.6×0.6 maze against a 0.03-radius target — non-zero (occasional lucky
approach), far from anything resembling a trained policy. No NaNs, no
degenerate all-0/all-1 successes.

## 5. Goal-success definition + ε (spec 4.4)

**Definition:** `success_any_time` = the point mass's `(x,y)` ever comes
within `eps` of the goal at any point in the episode (bounded implicit
episode length 1000). `success_final` also computed (final-step proximity)
for a stricter variant.

**ε = 0.03**, sourced from the *code*, not a paper PDF (PDF extraction of
arXiv 2209.14935 failed cleanly — see below):
`url_benchmark/goals.py`, class `MazeMultiGoal`, in
`https://github.com/facebookresearch/controllable_agent` @
`5a9950b07f1edcb4bddd04f5819379541793d975` — the official reference
implementation for "Does Zero-Shot Reinforcement Learning Exist?" (arXiv
2209.14935), which is exactly the paper spec 23 section 8 cites for the
"maze (20 goals)" claim. `MazeMultiGoal` defines **exactly 20 goals** (5
per room × 4 rooms — matches spec's "20 goals" claim verbatim) and scores
them with `rewards.tolerance(distance, bounds=(0, target_size), margin=
target_size)` where **`target_size = 0.03`**. The 20 goal coordinates are
reproduced as data (not code) in `MAZE_20_GOALS`.

**Honest caveat, stated per the task's requirement:** the upstream
protocol itself does not report a hard binary success rate — it sums the
dense tolerance reward over the episode and separately logs the final-step
distance (`eval_maze_goals()` in `url_benchmark/pretrain.py`). Our binary
`success_any_time`/`success_final` is a **derived** metric reusing the same
full-credit radius (`target_size = 0.03`) as that dense reward's
`bounds=(0, target_size)`, not a copied binary-success number (none is
published). This matches spec 4.4's own framing ("success = proximity
within the published tolerance") as closely as the source material allows.

## 6. Launch command — NOT fully wired; here is exactly what's blocking it

`configs/gciql/point_mass_maze_rnd_seed0_measure_density_d50.yaml` (new)
records the intended P0 density-arm recipe: mirrors
`cube_single_state_measure_density_d50.yaml` byte-for-byte except
`env_name` and `measure_hit_spec: "0:2:0.03"`. **It is not runnable via
`python run_gciql.py --config-name point_mass_maze_rnd_seed0_measure_density_d50`
today.** Root cause: `third_party/ogbench/impls/utils/env_utils.py`
(vendored, must not be edited per `third_party/ogbench/PROVENANCE.md`)
hardcodes `ogbench.make_env_and_datasets(dataset_name, compact_dataset=True)`,
which unconditionally calls `gymnasium.make(env_name)` — there is no
OGBench-registered gymnasium id for `point_mass_maze`, and the wrapper
doesn't expose the underlying `dataset_path`/`dataset_only` kwargs that
`ogbench.make_env_and_datasets` itself actually supports (which WOULD
allow loading an arbitrary `.npz`, env-free). Fixing this needs either a
new parallel training entry point (reusing the vendored `agents`/
`Dataset`/`GCDataset` as library imports, driving training and calling
`scripts/exorl/point_mass_maze_env.py`'s adapter for eval instead of
`utils/evaluation.py`) or an env_utils extension point — that is exactly
spec 23a's stated job ("implementation contract to be written as spec 23a
AFTER P0 passes"), not something buildable as "a converter script, an
adapter, tests, plus at most a config" without either editing vendored
code or building the general integration this task was explicitly scoped
to avoid. **This is the one item I could not close** — flagging it plainly
rather than faking a launch command.

Everything the eventual spec-23a launcher needs is ready and verified:
the converted dataset (section 3), the layout + hit_spec (section 2), the
eval adapter (section 4), and the pinned hyperparameters/ε (this config +
section 5).

## 7. Blocker closed (follow-up task, 2026-08-11)

STATUS: **CLOSED. P0 is launchable and runs end-to-end.**

**Extension point.** `tools/wandb_mode_shim/exorl_env.py` (new) wraps
`utils.env_utils.make_env_and_datasets` via the same
monkeypatch-through-`sitecustomize.py` mechanism already used for agent
injection / DrQ registration (one new guarded block appended to
`sitecustomize.py`). Dispatch: `env_name` starting with `"exorl:"` routes
to a small registry (`EXORL_REGISTRY`, keyed `point_mass_maze_rnd_seed0`)
that loads the converted `.npz` pair and builds `_ExoRLPointMassMazeEnv`, a
gym-shim around the already-tested `scripts/exorl/point_mass_maze_env.py`
adapter satisfying exactly the contract vendored `evaluate()`/`main.py`
need (`reset(options=...)`, `step()`, `render()`, `.unwrapped.task_infos`,
20 published maze goals, `info['goal']` = full 4-dim `[gx,gy,0,0]`,
success = any-time proximity ≤ 0.03 on the verified 0:2 block). Every other
`env_name` calls straight through to the original vendored function with
unchanged arguments. `third_party/ogbench/` was **not** edited; neither was
`tools/wandb_mode_shim/gciql_flow.py`. All ExoRL-specific imports
(`point_mass_maze_env`, `dm_control`) are deferred into the registry
builder, not done at `sitecustomize` load time, so non-ExoRL runs pay zero
import cost.

`run_gciql.py` got one small additive change: `build_ogbench_argv` now
emits `--eval_tasks=`/`--video_episodes=` only if the config sets those
keys (none do today), needed to shrink the smoke eval below.

**Second blocker found only during verification (not part of the original
gap):** `/dev/shm/.venv-jax` has `mujoco==3.3.0`, but `dm_control==1.0.41`
(already installed there) declares `Requires-Dist: mujoco>=3.8.1`, so any
`dm_control.mujoco.engine.Physics.from_xml_string` call in that venv fails
(`AttributeError: 'MjModel' object has no attribute 'B_colind'`) —
pre-existing unmet dependency, unrelated to my design, and NOT triggered by
OGBench's own manipspace envs (which don't go through that dm_control code
path). Upgrading `mujoco` venv-wide was rejected as out-of-scope (shared,
live-training venv). Fix: `exorl_env._import_point_mass_maze_env()`
detects a `mujoco<3.8.1` already cached in `sys.modules`, purges it, and
re-imports `mujoco` with the repo's own `.venv` (which already has a
working `mujoco==3.8.1` + the same `dm_control==1.0.41`) temporarily
prepended to `sys.path`, then removes that path immediately. `numpy` is
imported first specifically so it resolves from the jax venv throughout;
verified empirically that numpy's identity is unchanged before/after. This
executes only inside the ExoRL registry builder — cube/scene/puzzle keep
using the jax venv's own `mujoco==3.3.0`, confirmed below. No package was
installed or modified on disk anywhere.

**Smoke run (P0, real entrypoint, GPU 5 idle / `JAX_PLATFORMS=cpu`,
`wandb_mode=offline`, output under the scratchpad):**
```
CUDA_VISIBLE_DEVICES=5 JAX_PLATFORMS=cpu .venv/bin/python run_gciql.py \
  --config-name point_mass_maze_rnd_seed0_measure_density_d50 \
  output_root=<scratch> wandb_mode=offline train_steps=300 log_interval=100 \
  eval_interval=300 eval_episodes=1 +eval_tasks=1 +video_episodes=0 \
  save_interval=100000 run_group=spec23-p0-smoke
```
Log shows `MuJoCo library version is: 3.8.1` (overlay engaged), converted
dataset loaded, training to step 300, and a working goal-conditioned eval
(`evaluation/goal1_success`: 0.0 at step 1 → 1.0 at step 300). `train.csv`,
all finite, no NaN:
```
step=100: density_ce=4.595  density_ess_median=50.989  density_k_self=0.0605
step=200: density_ce=4.564  density_ess_median=57.708  density_k_self=0.0626
step=300: density_ce=4.485  density_ess_median=47.834  density_k_self=0.0606
```
(Gate G0a per spec section 5 — median ESS ≥ 4, CE finite, no NaN — passes
even at this toy step count.)

**Regression check (existing OGBench config unaffected), same harness:**
```
CUDA_VISIBLE_DEVICES=5 JAX_PLATFORMS=cpu .venv/bin/python run_gciql.py \
  --config-name cube_single_state_measure_density_d50 \
  output_root=<scratch> wandb_mode=offline train_steps=300 log_interval=100 \
  eval_interval=300 eval_episodes=1 +eval_tasks=1 +video_episodes=0 \
  save_interval=100000 run_group=spec23-p0-regress-cube-single
```
Log shows `MuJoCo library version is: 3.3.0` (no overlay — confirms the
default path is untouched), `--env_name=cube-single-play-v0` builds via
`gymnasium.make` exactly as before, trains 300 steps, produces the same
metric shape (`density_ce`/`density_ess_median`/`density_k_self` all
finite, `evaluation/task1_horizontal_success` present). Argv for both runs
is otherwise byte-identical in structure to a pre-existing run (verified
directly via `scripts/dev/test_run_gciql.py`, unchanged, still passing).

**Tests:**
```
.venv/bin/python -m pytest tests/test_exorl_convert.py \
  tests/test_exorl_point_mass_maze_env.py \
  tests/test_exorl_env_extension_point.py \
  tests/test_wandb_mode_shim.py -q
# 14 passed
.venv/bin/python scripts/dev/test_run_gciql.py   # RUN_GCIQL ARGV OK
```
New file `tests/test_exorl_env_extension_point.py` (4 tests, jax-subprocess
harness): (1) non-`exorl:` names reach the ORIGINAL `ogbench.make_env_and_datasets`
with unchanged `(dataset_name, compact_dataset=True)` args (stubbed, byte-identical
default-path proof); (2) an unregistered `exorl:` key raises `ValueError`
and never touches `ogbench.make_env_and_datasets`; (3) `frame_stack` for
ExoRL raises `NotImplementedError` (not silently mishandled); (4)
end-to-end real-dataset load + a working `reset`/`step` env contract
(skips if the converted dataset isn't on the box). Two pre-existing,
unrelated collection failures noted and confirmed NOT caused by this work:
`tests/test_run_gciql_visual.py::test_runner_script_exists_and_is_executable`
(a shell script deleted in commit `07deac3`, before this session) and
`tests/test_extract_ogbench_visual.py` (references a missing
`scripts/extract_ogbench.py`, also pre-existing).

**Files changed/added this task:**
- `tools/wandb_mode_shim/exorl_env.py` (new) — the extension point
- `tools/wandb_mode_shim/sitecustomize.py` (+15 lines, one new guarded
  block, same pattern as its neighbors)
- `run_gciql.py` (+8 lines, optional `eval_tasks`/`video_episodes`
  passthrough, no-op for every existing config)
- `configs/gciql/point_mass_maze_rnd_seed0_measure_density_d50.yaml` —
  `env_name` flipped to `exorl:point_mass_maze_rnd_seed0`, header comment
  updated from "NOT YET LAUNCHABLE" to launchable
- `tests/test_exorl_env_extension_point.py` (new, 4 tests)
- `third_party/ogbench/` and `tools/wandb_mode_shim/gciql_flow.py`: **not
  touched**

**P0 launch command** (real, full recipe — 500k steps, no truncation
flags, matches spec section 5's "500k steps" cell exactly):
```
CUDA_VISIBLE_DEVICES=<free GPU> .venv/bin/python run_gciql.py \
  --config-name point_mass_maze_rnd_seed0_measure_density_d50 \
  wandb_mode=online
```

## Files created (all new; nothing vendored or shared was edited)

- `third_party/exorl_dmc_tasks/` — vendored point_mass_maze task + XMLs +
  `PROVENANCE.md` + `LICENSE`
- `scripts/exorl/convert_exorl.py` — ExoRL → GCDataset converter
- `scripts/exorl/point_mass_maze_env.py` — dm_control eval adapter
- `tests/test_exorl_convert.py` — round-trip + boundary tests (4)
- `tests/test_exorl_point_mass_maze_env.py` — adapter smoke tests (4)
- `configs/gciql/point_mass_maze_rnd_seed0_measure_density_d50.yaml` — P0
  recipe (not yet launchable, see section 6)
- Converted dataset (outside repo):
  `~/exorl_data/converted/point_mass_maze_rnd_seed0/*.npz`
- Raw ExoRL clone + downloaded buffer (outside repo):
  `~/exorl_data/exorl/`
- Reference clone for ε provenance (outside repo, not a runtime dependency):
  `~/exorl_data/controllable_agent/`
