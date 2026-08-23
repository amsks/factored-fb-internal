# Density-FB

A JAX implementation of a critics annd actors on [OGBench](https://github.com/seohongpark/ogbench) 
cube manipulation. All value representations can be  composed with flow, DDPG+BC, AWR and one-step actors 
and run under a single training loop, so they can be compared on equal footing.

---

## The Methods

`impls/agents/__init__.py` composes a critic (the value representation) with an actor (the
policy) over per-group Flax `TrainState`s. `check_combo` in `impls/critics/base.py`
validates the pair before building it, and `tests/impls/test_boot_contract.py` builds
every registered pair.

The `CRITICS` and `ACTORS` dicts in `impls/agents/__init__.py` are the authoritative
lists. Critics are the OGBench family (`gciql`, `gcivl`, `hiql`, `crl`, `qrl`), bilinear
GCIQL (`iql_bilinear`), and the successor-measure family (`fb`, `density_fb`, `psm`,
`rldp`). Actors are `flowbc` (flow matching with a distilled one-step actor), `bcflow`
(the same actor with the value term switched off, i.e. a pure-BC control), `ddpgbc`,
`ddpgbc_density`, `awr` and `fql`.

Two composition rules are worth knowing before reading a result table:

- `flowbc` conditions on whatever the critic publishes as `cond_vec`. For `fb` that is
  the latent z; for goal critics it is the goal, which makes the pair a
  goal-conditioned flow-BC actor. The flow field stays goal-agnostic — only the one-step
  actor is conditioned.
- `awr` uses a native `V(s)` when the critic provides one, and otherwise falls back to the
  policy-baseline advantage `A = Q(s,a) − Q(s, π(s))`, so it composes with value-less
  critics too.

---

## Installation

Python 3.11. `ogbench` — and `jax`, `flax`, `optax`, `numpy` and `gymnasium` through its
`[train]` extra — is installed editable from the vendored source in `ogbench`.

```bash
make install          # uv venv then requirements.txt
```

Verify with `python -c 'import jax; print(jax.devices())'`. 

---

## Running

```bash
python -m impls.main \
    --critic fb --actor flowbc --domain cube_single \
    --seed 0 --train_steps 1000000 --save_dir exp/
```

`impls/utils/config_utils.py:load_config` composes the config from the sibling groups
under `impls/configs/{critic,actor,domain}/`. `--override key=value` is repeatable.
Checkpointing and restore go through `--save_dir`, `--restore_path` (a glob over run
dirs) and `--restore_epoch`.

`make run` runs the default pair;

### Online

`--online` collects from the environment during training instead of reading a fixed
dataset. Default off; the offline path is bit-identical when it is not set.

```bash
MUJOCO_GL=egl python -m impls.main \
    --critic density_fb --actor flowbc --domain cube_single \
    --seed 0 --train_steps 1000000 --save_dir exp/ \
    --online=True --online_start_steps 5000 --online_buffer_size 200000
```

`--online_env_steps_per_update` sets the env-step to gradient-step ratio (1.0 default;
0.5 is one env step every other update). `--online_action_noise` scales the exploration
noise, `--online_start_steps` is a uniform-random warmup, and `--online_seed_offline`
prefills the buffer from the offline dataset.

The buffer stores whole episodes and is served through the same `make_dataset` the
offline path uses, so goal-conditioned critics get real hindsight goals rather than
placeholders. It is checkpointed alongside the agent; `--restore_path` restores both, and
`--online_restore_buffer=false` opts back into starting with an empty buffer.

Collection is conditioned on the eval goal, so the buffer only covers what a
goal-directed policy reaches; `--online_explore_goal_prob` (default 0) draws some episode
goals from the buffer instead.

### Tests

Run the suite **one file at a time**. `pytest tests/impls` in a single process aborts
partway through -- XLA retains a compiled executable per jitted `Agent.update`, and the
suite builds enough critic x actor pairs to exhaust the process.

```bash
for f in tests/impls/test_*.py; do
    MUJOCO_GL=egl JAX_PLATFORMS=cpu PYTHONPATH=$PWD python -m pytest "$f" -q
done
```

`JAX_PLATFORMS=cpu` is required by `tests/impls/test_boot_contract.py`: it compares 40
critic x actor pairs against a byte-identical baseline, and GPU results differ in the last
bits. That gate is also sensitive to intra-op thread count, so it can report differences on
unmodified code under non-default `OMP_NUM_THREADS` or single-threaded XLA settings.
`MUJOCO_GL=egl` is needed by the pixel and env tests; without it they fail with
`gladLoadGL error`.

---

## Data

`ogbench.make_env_and_datasets(name, compact_dataset=True)` in `impls/utils/env_utils.py` loads
the official OGBench datasets, which OGBench downloads and caches under `~/.ogbench` on first
use. The other four families read from directories you point at with environment variables.

`env/domains/` holds 23 domains in five families. `--domain <name>` takes any of them; the
name is the yaml stem.

| family | domains | needs |
|---|---|---|
| OGBench | `cube-single`, `cube-double`, `scene`, `antmaze`, `pointmaze`, `cube-single-pixels` | nothing |
| MetaWorld / ManiSkill | `mw-pick-place`, `mw-push`, `ms-pick-cube`, `ms-push-cube` | `FB_DATA_ROOT` |
| MetaWorld (HF export) | `mw-assembly`, `mw-basketball`, `mw-bin-picking`, `mw-box-close`, `mw-button-press` | `FB_HF_EXPORT` |
| Box2D | the five `b2d-*` | `FB_COLLECTED` (no) |
| ExoRL (dm_control) | `dmc-point-mass-maze`, `dmc-walker`, `dmc-cheetah` | `EXORL_DATA_ROOT`, plus `FB_MUJOCO_SITE_PACKAGES` |

MetaWorld is split across two roots, which is easy to trip over: `mw-push` and
`mw-pick-place` come from `FB_DATA_ROOT`, the other five from `FB_HF_EXPORT`. Setting only
one silently leaves half the MetaWorld domains unresolvable.

`FB_MUJOCO_SITE_PACKAGES` is colon-separated and must point at a site-packages directory
whose `mujoco` matches the installed `dm_control`. Without it the ExoRL domains and their
tests raise a `RuntimeError` that says so. 

ExoRL is the only family with **reward-labelled** data. Its buffers carry the reward of the
task the data was collected under, so the loader relabels from the stored physics instead of
passing that column through -- a passthrough would be right for one task per domain and
silently wrong for the others.

---

## Layout

```
impls/            critics/, actors/, agents/, utils/, configs/, main.py
tests/impls/      parity and seam suite, with frozen .npz fixtures
docs/             coverage audit, specs, plans, notes
```
