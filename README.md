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

---

## Data

`ogbench.make_env_and_datasets(name, compact_dataset=True)` in `impls/utils/env_utils.py` loads 
the official OGBench datasets, which OGBench downloads and caches under `~/.ogbench` on first 
use.

---

## Layout

```
impls/            critics/, actors/, agents/, utils/, configs/, main.py
tests/impls/      parity and seam suite, with frozen .npz fixtures
docs/             coverage audit, specs, plans, notes
```
