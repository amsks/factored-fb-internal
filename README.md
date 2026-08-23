# Factored-FB (JAX)

A JAX implementation of a critic × actor seam for offline goal-conditioned RL on
[OGBench](https://github.com/seohongpark/ogbench) cube manipulation. The seam lets
Forward–Backward (FB) and GCIQL-family value representations be composed with flow,
DDPG+BC, AWR and one-step actors and run under a single training loop, so they can be
compared on equal footing.

The research question behind it is when future-reachability methods fail on manipulation.
The paper trees live on other branches.

---

## Method

FB factors the goal-conditioned Q-function as a bilinear form:

```
Q(s, a, z) = F(s, z, a) · B(s')
```

F predicts successor features; B maps observations into a latent goal space z. A task is
specified at test time by inferring z from relabelled offline transitions,
`z = project(Σ rᵢ · B(next_obsᵢ))`, with no fine-tuning.

The bridge restates GCIQL and its bilinear and successor-feature variants in the same
factored API, so one loop runs all of them. Which knobs of the original research shim map
to which critic is recorded in
[`docs/jaxify-coverage-audit.md`](docs/jaxify-coverage-audit.md).

---

## The seam

`impls/agents/__init__.py` composes a critic (the value representation) with an actor (the
policy) over per-group Flax `TrainState`s. `check_combo` in `impls/critics/base.py`
validates the pair before building it, and `tests/impls/test_boot_contract.py` builds
every registered pair.

The `CRITICS` and `ACTORS` dicts in `impls/agents/__init__.py` are the authoritative
lists. Critics cover the FB family (`fb`, `isfb`, `td_fb`, `density_fb`, `psm`, `rldp`),
the GCIQL family and its bilinear and successor-feature variants (`gciql`, `gcivl`,
`hiql`, `iql_bilinear`, `simple_sf`, `sf_feature`, `centered_bilinear`,
`fb_forward_bilinear`), and the contrastive and quasimetric baselines (`crl`, `qrl`,
`vip`). Actors are `flowbc` (flow matching with a distilled one-step actor), `bcflow`
(the same actor with the value term switched off, i.e. a pure-BC control), `ddpgbc`,
`ddpgbc_density`, `awr` and `fql`.

Two composition rules are worth knowing before reading a result table:

- `flowbc` conditions on whatever the critic publishes as `cond_vec`. For `fb` and `isfb`
  that is the latent z; for goal critics it is the goal, which makes the pair a
  goal-conditioned flow-BC actor. The flow field stays goal-agnostic — only the one-step
  actor is conditioned.
- `awr` uses a native `V(s)` when the critic provides one, and otherwise falls back to the
  policy-baseline advantage `A = Q(s,a) − Q(s, π(s))`, so it composes with value-less
  critics too.

---

## Installation

Python 3.11. `ogbench` — and `jax`, `flax`, `optax`, `numpy` and `gymnasium` through its
`[train]` extra — is installed editable from the vendored source in `third_party/ogbench`.

```bash
make install          # uv venv at /dev/shm/.venv-jax, then requirements.txt
```

Verify with `/dev/shm/.venv-jax/bin/python -c 'import jax; print(jax.devices())'`. The
venv lives in tmpfs and does not survive a reboot.

`impls/utils/torch_to_flax.py` reads the parity fixtures and remaps weights; despite the
name it imports no torch.

---

## Running

```bash
/dev/shm/.venv-jax/bin/python -m impls.main \
    --critic fb --actor flowbc --domain cube_single \
    --seed 0 --train_steps 1000000 --save_dir exp/
```

`impls/utils/config_utils.py:load_config` composes the config from the sibling groups
under `impls/configs/{critic,actor,domain}/`. `--override key=value` is repeatable.
Checkpointing and restore go through `--save_dir`, `--restore_path` (a glob over run
dirs) and `--restore_epoch`.

`make run` runs the default pair; `make smoke` runs five training steps on CPU.

---

## Data

`ogbench.make_env_and_datasets(name, compact_dataset=True)` in
`impls/utils/env_utils.py` loads the official OGBench datasets, which OGBench downloads
and caches under `~/.ogbench` on first use. `cube_single` (`cube-single-play-v0`) is the
only domain wired up — it is the sole file in `impls/configs/domain/`.

---

## Tests

```bash
make test             # JAX_PLATFORMS=cpu pytest tests/impls/ -q
make test-fast        # collection only, catches import breakage in seconds
```

Ten frozen reference arrays in `tests/impls/fixtures/*.npz` pin numerical parity against
the original torch/shim implementation, at tolerances down to 1e-10. The generators for
the torch-side fixtures were removed along with the torch reference, so regenerating one
means checking out a pre-cleanup ref; see
[`docs/jaxify-coverage-audit.md`](docs/jaxify-coverage-audit.md).
`tests/impls/test_boot_contract.py` is the byte-identity gate over every registered
critic × actor pair.

Two tests in `test_density_fb_pixels.py` render a MuJoCo scene and fail on a headless box
with no GL context; they need `MUJOCO_GL=egl` or a display.

---

## Scope

`cube_single` is the only wired domain. The scene domain and its factored `value_split`
value are not here.

`density_fb` has a pixel/DrQ path, selected with `--critic density_fb_pixels`. It needs
the privileged object-coordinate channel described in
`docs/specs/2026-08-21-density-fb-pixels-drq.md`, because the density kernel is defined by
column slices of object coordinates and there is nothing to slice on an image. No other
critic runs on images.

The PyTorch implementation and TD-MPC2 live on other branches.

---

## Layout

```
impls/            critics/, actors/, agents/, utils/, configs/, main.py
tests/impls/      parity and seam suite, with frozen .npz fixtures
third_party/      vendored OGBench (env/dataset dep, plus jax/flax via [train])
docs/             coverage audit, specs, plans, notes
```
