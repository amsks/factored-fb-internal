# Factored-FB (JAX)

A clean, standalone **JAX** implementation of the **Forward–Backward (FB) ↔ GCIQL bridge**
for offline goal-conditioned RL on state-based [OGBench](https://github.com/seohongpark/ogbench)
cube manipulation.

Everything lives in one package, **`impls/`** — a critic × actor *seam* that lets FB-family
and GCIQL-family value representations be composed with flow / DDPG+BC / AWR / one-step
actors and compared on equal footing. The scientific thrust is a **failure analysis of
future-reachability methods on manipulation** (see the paper trees on the other branches).

> This branch is **code-only and state-only**. The PyTorch implementation, the pixel/DrQ
> path, the scene domain, and the PSM/RLDP/TD-MPC2 agents are **not** here — they live on
> other branches. See [Scope](#scope).

---

## Method

FB decomposes the goal-conditioned Q-function as a bilinear form:

```
Q(s, a, z) = F(s, z, a) · B(s')
```

**F** (forward map) predicts successor features; **B** (backward map) maps observations to a
latent goal space **z**. At test time a task is specified by inferring **z** as
`z = project(Σ rᵢ · B(next_obsᵢ))` over relabelled offline transitions — no fine-tuning.

The **bridge** recasts GCIQL and its bilinear/successor-feature variants in the same factored
API, so a single training loop can run FB, GCIQL, ISFB, and the bilinear/SF critics under one
actor family. Which knobs of the old research shim map to which critic is recorded in
[`docs/jaxify-coverage-audit.md`](docs/jaxify-coverage-audit.md).

---

## The critic × actor seam

`impls/agents/__init__.py` composes a **critic** (the value representation) with an **actor**
(the policy) over per-group Flax `TrainState`s, and `check_combo` (`impls/critics/base.py`)
validates the pair before building. **All 32 critic × actor pairs are valid** — every actor
composes with every critic.

**Critics:** `fb`, `gciql`, `isfb`, `iql_bilinear`, `simple_sf`, `sf_feature`,
`centered_bilinear`, `fb_forward_bilinear`
**Actors:** `flowbc` (flow-matching + distilled noise actor), `ddpgbc` (DDPG+BC),
`awr` (advantage-weighted regression), `fql` (one-step flow).

| critic \ actor | flowbc | ddpgbc | awr | fql |
|---|:---:|:---:|:---:|:---:|
| `fb` | ✅ | ✅ | ✅ᵛ | ✅ |
| `gciql` | ✅ᶜ | ✅ | ✅ | ✅ |
| `isfb` | ✅ | ✅ | ✅ | ✅ |
| `iql_bilinear` | ✅ᶜ | ✅ | ✅ | ✅ |
| `simple_sf` | ✅ᶜ | ✅ | ✅ | ✅ |
| `sf_feature` | ✅ᶜ | ✅ | ✅ᵛ | ✅ |
| `centered_bilinear` | ✅ᶜ | ✅ | ✅ᵛ | ✅ |
| `fb_forward_bilinear` | ✅ᶜ | ✅ | ✅ | ✅ |

Every pair builds and trains. Two cells use **fallback semantics** where the critic lacks the
native quantity the actor prefers:

- **ᶜ** — `flowbc` conditions on the critic's `cond_vec`. For `fb`/`isfb` that *is* the latent
  `z` (native); for goal-critics it is the **goal**, making it a goal-conditioned flow-BC actor
  (the flow field stays goal-agnostic; only the one-step actor is conditioned).
- **ᵛ** — `awr` uses a native value `V(s)` when the critic provides one; for value-less critics
  (`fb`, `sf_feature`, `centered_bilinear`) it falls back to a **policy-baseline advantage**
  `A = Q(s,a) − Q(s, π(s))`.

`fb`/`isfb`-flowbc and every V-critic-awr are **byte-identical** to before this generalization
(guarded by the parity fixtures); only the fallback cells are new.

---

## Installation

**Python ≥ 3.11.** Uses [`uv`](https://github.com/astral-sh/uv). The `ogbench` env/dataset
library — and `jax` / `flax` / `optax` / `numpy` / `gymnasium` via its `[train]` extra — is
installed **editable from the vendored source** at `third_party/ogbench`.

```bash
make install          # creates /dev/shm/.venv-jax and installs requirements.txt
# or manually:
uv venv --python 3.11 /dev/shm/.venv-jax
/dev/shm/.venv-jax/bin/pip install -r requirements.txt
```

Verify: `/dev/shm/.venv-jax/bin/python -c 'import jax; print(jax.devices())'`

> `impls/utils/torch_to_flax.py` is a **pure-JAX** weight-remapping helper (it reads the
> parity fixtures); despite the name it imports no torch.

---

## Running

Train one critic × actor pair on cube-single via the `impls.main` entrypoint:

```bash
/dev/shm/.venv-jax/bin/python -m impls.main \
    --critic fb --actor flowbc --domain cube_single \
    --seed 0 --train_steps 1000000 --save_dir exp/
```

Config is **composed** from the sibling groups under `impls/configs/{critic,actor,domain}/`
by `impls/utils/config_utils.py:load_config`. Override any key with `--override key=value`
(repeatable). Checkpoint / restore via `--save_dir`, `--restore_path` (a glob over run dirs),
and `--restore_epoch`.

Quick smoke (5 steps, CPU):

```bash
make smoke            # CRITIC=fb ACTOR=flowbc by default; override on the CLI
```

---

## Data

The loop uses the official OGBench datasets via `ogbench.make_env_and_datasets(name,
compact_dataset=True)` (`impls/utils/env_utils.py`). The currently wired domain is
`cube_single` (`cube-single-play-v0`). OGBench downloads and caches datasets under
`~/.ogbench` on first use.

---

## Tests & parity

```bash
make test             # full impls/ suite on CPU
# == XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu \
#      /dev/shm/.venv-jax/bin/python -m pytest tests/impls/ -q
```

Every critic and actor carries a **1e-10 numerical-parity** test against a reference
implementation. The reference values are frozen in `tests/impls/fixtures/*.npz` — golden
values captured at parity with the original torch/shim implementation. Those generators were
removed with the torch reference (this is a JAX-only repo); regenerating a fixture requires
checking out a pre-cleanup ref. See [`docs/jaxify-coverage-audit.md`](docs/jaxify-coverage-audit.md).

---

## Scope

- **In:** state-based OGBench cube-single; the FB↔GCIQL bridge critics and the four actors.
- **Deferred:** the pixel / DrQ path; the scene domain and its factored `value_split` value.
- **Elsewhere:** PSM, RLDP, TD-MPC2, and the PyTorch implementation live on their own branches.

---

## Layout

```
impls/            the implementation (critics/, actors/, agents/, utils/, configs/, main.py)
tests/impls/      parity + seam suite (+ frozen .npz fixtures)
third_party/      vendored OGBench (the runtime env/dataset dep + jax/flax via [train])
docs/             coverage audit, specs, plans, notes
```
