# DrQ encoder (JAX) for OGBench GCIQL/GCIVL — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give OGBench's GCIQL/GCIVL pixel baselines a DrQ-v2 visual front-end (encoder + augmentation + frame-stack + normalization) that matches FB's pixel pipeline, so the only FB-vs-baseline difference is the RL algorithm — without ever editing vendored OGBench.

**Architecture:** A new `tools/wandb_mode_shim/ogbench_drq.py` holds a Flax `DrQEncoder` (mirrors FB's PyTorch `DrQEncoder`) and a `random_shifts` augmentation. The existing `sitecustomize` shim (already on the OGBench child's `PYTHONPATH`) registers the encoder into `encoder_modules['drq']` and — gated on `OGBENCH_DRQ_FRONTEND=1` — swaps `GCDataset.augment` for random_shifts. `run_gciql.py` sets that env var when a config selects `encoder: drq`. Two new Hydra configs select the DrQ front-end for GCIQL and GCIVL.

**Tech Stack:** JAX + Flax (`/dev/shm/.venv-jax`, the OGBench track), PyTorch (`.venv`, the FB track), Hydra, pytest.

---

## Spec reference

`docs/superpowers/specs/2026-05-21-drq-encoder-ogbench-design.md` (commit `90bedf0`, status: approved).

## Critical context (verified against the codebase before planning)

- **Two isolated venvs, no overlap:** `.venv` has `torch` but **not** `jax/flax`; `/dev/shm/.venv-jax` has `jax/flax/numpy` but **not** `torch`. `/dev/shm/.venv-jax` is **tmpfs — wiped on reboot**; `make install-jax` recreates it. Consequence: every JAX-side test runs by shelling out to the jax interpreter and **skips** if it is absent. The weight-transplant parity test bridges the two: torch runs in-process (pytest is under `.venv`), jax runs in a subprocess, communicating via an `.npz` artifact.
- **Hard constraint:** never edit `third_party/ogbench` (pinned SHA `1d4140997f60c52c6fb0702ec100dc988b18c548`). All injection is via `tools/wandb_mode_shim/sitecustomize.py`, which `run_gciql.py` already places first on the child `PYTHONPATH` (`run_gciql.py:101-103`).
- **FB `DrQEncoder`** (`nn_models.py:480-519`): conv trunk = `Conv2d(in,32,3,stride=2)→ReLU→[Conv2d(32,32,3,stride=1)→ReLU]×3→Flatten`, all PyTorch default padding=0 (= Flax `padding='VALID'`); then `proj = Linear(repr_dim, 256)→LayerNorm(256)→Tanh`. It does **not** normalize internally — `RGBNorm` is applied outside (`normalizers.py:38-42`, `(x/255)-0.5`). The JAX version normalizes **inside** (mirroring how OGBench's `ImpalaEncoder` does `x/255` internally, `encoders.py:84`), so the JAX encoder does `x/255 - 0.5` first.
- **Conv spatial math** for 64×64 input: `64 →(s2)→ 31 →(s1)→ 29 → 27 → 25`; `repr_dim = 25·25·32 = 20000`.
- **FB `Augmentator`** (`nn_models.py:530-552`): `F.pad(replicate, pad)` then bilinear `grid_sample` with a per-sample **integer** shift in `[0, 2·pad]`, `align_corners=False`, `padding_mode='zeros'`. Because the shifts are integers landing exactly on padded-pixel centers and never exceed the replicate-padded region, this is **provably identical** to "replicate-pad by `pad`, then integer-crop `[shift:shift+h]`". FB's visual config uses **`pad=2`** (`configs/domain/visual_cube_single.yaml:18-20`). FB applies it **independently** per tensor (`agents/fb/agent.py:178` calls `_augmentator(obs)` and `_augmentator(next_obs)` separately).
- **OGBench's native augment** (`utils/datasets.py:283-293`): `random_crop` uses `jnp.pad(mode='edge')` (= replicate) + `dynamic_slice`, `padding=3`, and shares **one** `crop_froms` across all keys. So the only differences vs FB are `pad` (3→2) and shared→independent shifts. We replace the whole `augment` method to get FB's `pad=2` + independent per-key shifts.
- **Encoder/aug/frame_stack/p_aug config flow:** `agents/gciql.py:211-215` and `agents/gcivl.py:176-179` do `encoder_modules[config['encoder']]` wrapped in `GCEncoder(concat_encoder=...)`. `main.py:58` passes `frame_stack=config['frame_stack']` to `make_env_and_datasets`. Both `get_config()`s already declare `encoder` (placeholder str), `frame_stack` (placeholder int), `p_aug` (0.0), `dataset_class='GCDataset'` — so `--agent.encoder=drq`, `--agent.frame_stack=3`, `--agent.p_aug=1.0` are all valid overrides.
- **`GCEncoder` caveat (documented, not a bug):** `GCEncoder(concat_encoder=enc)` encodes `concat([obs, goal], axis=-1)` (`encoders.py:134`) — ~18 channels at frame_stack=3, vs FB's 9 (obs alone). Same conv architecture; channel count differs because the *algorithm* differs. Flax infers input channels, so the module handles 9 or 18.

## Deviations from the spec (flagged for review)

1. **Augment stays in JAX, in `ogbench_drq.py` (not inlined in `sitecustomize.py`).** Spec Component 2 says "in the shim"; extracting it to the importable module is what makes it unit-testable in isolation. Behaviorally identical.
2. **Augment uses integer-crop (`jnp.pad(mode='edge')` + `dynamic_slice`), not literal `grid_sample`.** Proven equivalent to FB's integer-shift `grid_sample` (see Critical context). Simpler, exact, and reuses OGBench's own `random_crop` shape of implementation.
3. **Independent per-key shifts** (FB-faithful) rather than OGBench's shared `crop_froms`.

If you disagree with any of these, stop and raise it before implementing — they are deliberate, reversible choices.

## File structure

| File | Create/Modify | Responsibility |
|------|---------------|----------------|
| `tools/wandb_mode_shim/ogbench_drq.py` | Create | Flax `DrQEncoder`; `random_shifts_batch` + `gcdataset_augment` method. Imports jax/flax/numpy. |
| `tools/wandb_mode_shim/sitecustomize.py` | Modify | Add a guarded block: register `encoder_modules['drq']`; gate-patch `GCDataset.augment`. |
| `run_gciql.py` | Modify | Extract `build_child_env(plain, base_env)`; set `OGBENCH_DRQ_FRONTEND=1` when `agent_overrides.encoder == 'drq'`. |
| `configs/gciql/cube_single_visual_drq.yaml` | Create | GCIQL + DrQ front-end. |
| `configs/gciql/cube_single_visual_gcivl_drq.yaml` | Create | GCIVL + DrQ front-end. |
| `tests/jax_subprocess.py` | Create | Test helper: locate jax python (skip if absent), run a jax snippet with the shim + OGBench impls on `PYTHONPATH`. |
| `tests/test_ogbench_drq_encoder.py` | Create | JAX encoder forward-shape test (subprocess). |
| `tests/test_ogbench_drq_augment.py` | Create | random_shifts shape/identity/determinism/passthrough (subprocess). |
| `tests/test_ogbench_drq_parity.py` | Create | Weight-transplant numerical parity (torch in-proc + jax subprocess). **Hard gate.** |
| `tests/test_ogbench_drq_injection.py` | Create | Encoder always registered; augment patched only with the flag (subprocess). |
| `tests/test_run_gciql_drq_env.py` | Create | `build_child_env` sets/omits `OGBENCH_DRQ_FRONTEND`. |
| `tests/test_drq_configs.py` | Create | The two configs build the expected `argv`. |

**Run tests with** `.venv/bin/python -m pytest` (per repo memory: `uv run` is broken). JAX-dependent tests auto-skip when `/dev/shm/.venv-jax` is absent.

---

## Task 1: Shared JAX-subprocess test helper

**Files:**
- Create: `tests/jax_subprocess.py`

- [ ] **Step 1: Write the helper**

```python
"""Helpers to run a JAX snippet in the isolated OGBench venv from pytest.

The main .venv (which runs pytest) has torch but not jax/flax; jax/flax live
only in /dev/shm/.venv-jax (tmpfs, recreated by `make install-jax`). JAX-side
tests therefore shell out to that interpreter and skip when it is absent.
"""
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SHIM_DIR = REPO / "tools" / "wandb_mode_shim"
OGBENCH_IMPLS = REPO / "third_party" / "ogbench" / "impls"
_JAX_CANDIDATES = (
    "/dev/shm/.venv-jax/bin/python",
    "/mnt/scratch1/.venv-jax/bin/python",
)


def jax_python() -> str:
    for p in _JAX_CANDIDATES:
        if Path(p).exists():
            return p
    pytest.skip("no .venv-jax interpreter found (run `make install-jax`)")


def run_jax(code: str, env_extra: dict | None = None, cwd: str | None = None):
    """Run `code` under the jax interpreter with the shim + OGBench impls on
    PYTHONPATH so `import ogbench_drq` and `import utils.*` both resolve."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SHIM_DIR), str(OGBENCH_IMPLS), env.get("PYTHONPATH", "")]
    )
    env["JAX_PLATFORMS"] = env.get("JAX_PLATFORMS", "cpu")  # tests don't need GPU
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [jax_python(), "-c", textwrap.dedent(code)],
        env=env, cwd=cwd, capture_output=True, text=True,
    )
```

- [ ] **Step 2: Sanity-check it imports**

Run: `.venv/bin/python -c "import tests.jax_subprocess as j; print('ok')"`
Expected: `ok` (no jax needed to import the helper itself).

- [ ] **Step 3: Commit**

```bash
git add tests/jax_subprocess.py
git commit -m "test(drq-ogbench): jax-subprocess test helper (skips without .venv-jax)"
```

---

## Task 2: Flax DrQ encoder

**Files:**
- Create: `tools/wandb_mode_shim/ogbench_drq.py`
- Test: `tests/test_ogbench_drq_encoder.py`

- [ ] **Step 1: Write the failing test**

```python
from tests.jax_subprocess import run_jax


def test_drq_encoder_forward_shape():
    code = """
        import jax, jax.numpy as jnp
        from ogbench_drq import DrQEncoder
        enc = DrQEncoder(feature_dim=256)
        x = jnp.zeros((4, 64, 64, 9), dtype=jnp.uint8)   # NHWC, frame_stack=3 RGB
        params = enc.init(jax.random.PRNGKey(0), x)["params"]
        out = enc.apply({"params": params}, x)
        assert out.shape == (4, 256), out.shape
        # conv trunk spatial reduction 64->31->29->27->25, 25*25*32 = 20000
        kshape = params["proj"]["kernel"].shape
        assert kshape == (20000, 256), kshape
        print("OK")
    """
    proc = run_jax(code)
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run it — expect SKIP or FAIL (module missing)**

Run: `.venv/bin/python -m pytest tests/test_ogbench_drq_encoder.py -v`
Expected: SKIP if no `.venv-jax`; otherwise FAIL with `ModuleNotFoundError: ogbench_drq`.

- [ ] **Step 3: Implement the encoder**

Create `tools/wandb_mode_shim/ogbench_drq.py`:

```python
"""JAX/Flax DrQ-v2 visual front-end for vendored OGBench (GCIQL/GCIVL).

Imported by the sitecustomize shim inside the OGBench child process (jax env).
Mirrors FB's PyTorch DrQEncoder (nn_models.py:480) and Augmentator
(nn_models.py:530) so OGBench's pixel baselines share FB's encoder, augmentation,
frame-stacking and normalization. Never edits vendored OGBench.
"""
import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class DrQEncoder(nn.Module):
    """DrQ-v2 RGB encoder (Flax), numerically matching FB's torch DrQEncoder.

    Normalizes internally (x/255 - 0.5, == FB's RGBNorm) because OGBench encoders
    own normalization (cf. ImpalaEncoder's x/255). __call__ takes (train, cond_var)
    for signature parity with ImpalaEncoder; both are ignored.
    """

    feature_dim: int = 256

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        init = nn.initializers.xavier_uniform()
        x = x.astype(jnp.float32) / 255.0 - 0.5
        x = nn.relu(nn.Conv(32, (3, 3), strides=2, padding="VALID",
                            kernel_init=init, name="conv0")(x))
        x = nn.relu(nn.Conv(32, (3, 3), strides=1, padding="VALID",
                            kernel_init=init, name="conv1")(x))
        x = nn.relu(nn.Conv(32, (3, 3), strides=1, padding="VALID",
                            kernel_init=init, name="conv2")(x))
        x = nn.relu(nn.Conv(32, (3, 3), strides=1, padding="VALID",
                            kernel_init=init, name="conv3")(x))
        x = x.reshape((*x.shape[:-3], -1))
        x = nn.Dense(self.feature_dim, kernel_init=init, name="proj")(x)
        x = nn.LayerNorm(epsilon=1e-5, name="proj_ln")(x)  # 1e-5 == torch default
        return jnp.tanh(x)
```

- [ ] **Step 4: Run the test — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_ogbench_drq_encoder.py -v`
Expected: PASS (or SKIP without `.venv-jax`).

- [ ] **Step 5: Commit**

```bash
git add tools/wandb_mode_shim/ogbench_drq.py tests/test_ogbench_drq_encoder.py
git commit -m "feat(drq-ogbench): Flax DrQ encoder + forward-shape test"
```

---

## Task 3: random_shifts augmentation

**Files:**
- Modify: `tools/wandb_mode_shim/ogbench_drq.py`
- Test: `tests/test_ogbench_drq_augment.py`

- [ ] **Step 1: Write the failing test**

```python
from tests.jax_subprocess import run_jax


def test_random_shifts_identity_at_center_shift():
    """shift == pad recovers the original image exactly (replicate-pad then crop
    [pad:pad+h] == identity)."""
    code = """
        import numpy as np, jax.numpy as jnp
        from ogbench_drq import _batched_shift
        rng = np.random.default_rng(0)
        imgs = jnp.asarray(rng.integers(0, 256, (3, 8, 8, 9), dtype=np.uint8))
        shifts = jnp.asarray(np.full((3, 2), 2, dtype=np.int32))  # pad=2
        out = _batched_shift(imgs, shifts, 2)
        assert out.shape == imgs.shape, out.shape
        assert np.array_equal(np.asarray(out), np.asarray(imgs))
        print("OK")
    """
    proc = run_jax(code)
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_random_shifts_batch_shape_determinism_passthrough():
    code = """
        import numpy as np
        from ogbench_drq import random_shifts_batch
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)
        def mk():
            return {
                "observations": rng_make_img(),
                "value_goals": rng_make_img(),
                "actions": np.zeros((4, 5), dtype=np.float32),  # non-image (2D)
            }
        import numpy as _np
        def rng_make_img():
            return _np.broadcast_to(
                _np.arange(8 * 8 * 9, dtype=_np.uint8).reshape(8, 8, 9), (4, 8, 8, 9)
            ).copy()
        keys = ["observations", "value_goals", "actions"]
        ba, bb = mk(), mk()
        random_shifts_batch(ba, keys, pad=2, rng=rng_a)
        random_shifts_batch(bb, keys, pad=2, rng=rng_b)
        # shapes preserved
        assert ba["observations"].shape == (4, 8, 8, 9)
        # deterministic under equal seeds
        assert np.array_equal(ba["observations"], bb["observations"])
        assert np.array_equal(ba["value_goals"], bb["value_goals"])
        # non-image key untouched
        assert np.array_equal(ba["actions"], np.zeros((4, 5), dtype=np.float32))
        # independent per-key: obs and goals get different shifts (very likely)
        assert not np.array_equal(ba["observations"], ba["value_goals"])
        print("OK")
    """
    proc = run_jax(code)
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run it — expect SKIP or FAIL (`_batched_shift` missing)**

Run: `.venv/bin/python -m pytest tests/test_ogbench_drq_augment.py -v`
Expected: SKIP without `.venv-jax`; otherwise FAIL with `ImportError`/`AttributeError`.

- [ ] **Step 3: Append the augmentation to `ogbench_drq.py`**

```python
def _shift_one(img, shift, pad):
    # img: (H, W, C); shift: (2,) int in [0, 2*pad]; replicate-pad then crop.
    padded = jnp.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    start = jnp.array([shift[0], shift[1], 0])
    return jax.lax.dynamic_slice(padded, start, img.shape)


@functools.partial(jax.jit, static_argnames=("pad",))
def _batched_shift(imgs, shifts, pad):
    """imgs: (N, H, W, C) uint8; shifts: (N, 2) int in [0, 2*pad]."""
    return jax.vmap(_shift_one, (0, 0, None))(imgs, shifts, pad)


def random_shifts_batch(batch, keys, pad=2, rng=None):
    """In-place FB-faithful random_shifts: replicate-pad by `pad`, integer-crop.

    Independent per-key shifts (matches FB's separate _augmentator calls). Only
    4D arrays (N,H,W,C) are augmented; other keys pass through. `rng` is an
    np.random.Generator (None -> fresh) so callers/tests can fix the seed.
    """
    g = np.random.default_rng() if rng is None else rng
    for key in keys:
        arr = batch[key]
        if getattr(arr, "ndim", 0) != 4:
            continue
        n = arr.shape[0]
        shifts = g.integers(0, 2 * pad + 1, size=(n, 2)).astype(np.int32)
        batch[key] = np.asarray(
            _batched_shift(jnp.asarray(arr), jnp.asarray(shifts), pad)
        )
    return batch


def gcdataset_augment(self, batch, keys):
    """Drop-in replacement for GCDataset.augment (pad=2, FB-faithful)."""
    random_shifts_batch(batch, keys, pad=2)


gcdataset_augment._ogbench_drq_patched = True  # sentinel for the injection test
```

- [ ] **Step 4: Run the tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_ogbench_drq_augment.py -v`
Expected: PASS (or SKIP without `.venv-jax`).

- [ ] **Step 5: Commit**

```bash
git add tools/wandb_mode_shim/ogbench_drq.py tests/test_ogbench_drq_augment.py
git commit -m "feat(drq-ogbench): FB-faithful random_shifts augment + tests"
```

---

## Task 4: Weight-transplant numerical parity (HARD GATE)

Proves the Flax encoder == FB's PyTorch `DrQEncoder` for the same input under the
same weights (max abs diff < 1e-5). torch runs in-process (`.venv`); jax runs in a
subprocess (`.venv-jax`); they communicate via a temp `.npz`.

**Files:**
- Test: `tests/test_ogbench_drq_parity.py`

- [ ] **Step 1: Write the test**

```python
import subprocess
import tempfile
import textwrap
from pathlib import Path

import numpy as np

from tests.jax_subprocess import jax_python, SHIM_DIR, OGBENCH_IMPLS
import os


def test_drq_jax_matches_fb_torch_under_transplant():
    import gymnasium
    import torch
    from nn_models import DrQEncoderArchiConfig

    torch.manual_seed(0)
    obs_space = gymnasium.spaces.Box(0, 255, (9, 64, 64), np.uint8)  # CHW, obs-alone
    enc = DrQEncoderArchiConfig(feature_dim=256).build(obs_space).eval()

    x_raw = torch.rand(2, 9, 64, 64) * 255.0          # [0,255] float, NCHW
    with torch.no_grad():
        out_t = enc(x_raw / 255.0 - 0.5).numpy()      # FB applies RGBNorm outside
    sd = enc.state_dict()
    # trunk = Sequential(Conv,ReLU,Conv,ReLU,Conv,ReLU,Conv,ReLU,Flatten) -> convs at 0,2,4,6
    # proj  = Sequential(Linear, LayerNorm, Tanh) -> Linear=proj.0, LayerNorm=proj.1
    arrs = dict(
        conv0_w=sd["trunk.0.weight"].numpy(), conv0_b=sd["trunk.0.bias"].numpy(),
        conv1_w=sd["trunk.2.weight"].numpy(), conv1_b=sd["trunk.2.bias"].numpy(),
        conv2_w=sd["trunk.4.weight"].numpy(), conv2_b=sd["trunk.4.bias"].numpy(),
        conv3_w=sd["trunk.6.weight"].numpy(), conv3_b=sd["trunk.6.bias"].numpy(),
        proj_w=sd["proj.0.weight"].numpy(),   proj_b=sd["proj.0.bias"].numpy(),
        ln_w=sd["proj.1.weight"].numpy(),     ln_b=sd["proj.1.bias"].numpy(),
        x_raw=x_raw.numpy(), out_t=out_t,
    )
    with tempfile.TemporaryDirectory() as d:
        npz = Path(d) / "drq_parity.npz"
        np.savez(npz, **arrs)

        jax_code = textwrap.dedent(
            """
            import sys, numpy as np, jax, jax.numpy as jnp
            from ogbench_drq import DrQEncoder
            d = np.load(sys.argv[1])
            x = jnp.asarray(d["x_raw"]).transpose(0, 2, 3, 1)        # NCHW -> NHWC
            enc = DrQEncoder(feature_dim=256)
            enc.init(jax.random.PRNGKey(0), x)                       # shape check only
            def ck(w):  # torch [out,in,kh,kw] -> flax [kh,kw,in,out]
                return jnp.asarray(w).transpose(2, 3, 1, 0)
            # proj: torch flatten order (C,H,W); flax reshape order (H,W,C).
            # reorder torch Linear weight columns C,H,W -> H,W,C, then transpose to [in,out].
            pw = d["proj_w"].reshape(256, 32, 25, 25)                # [out,C,H,W]
            pw = np.transpose(pw, (2, 3, 1, 0)).reshape(20000, 256)  # [H*W*C, out]
            params = {
                "conv0": {"kernel": ck(d["conv0_w"]), "bias": jnp.asarray(d["conv0_b"])},
                "conv1": {"kernel": ck(d["conv1_w"]), "bias": jnp.asarray(d["conv1_b"])},
                "conv2": {"kernel": ck(d["conv2_w"]), "bias": jnp.asarray(d["conv2_b"])},
                "conv3": {"kernel": ck(d["conv3_w"]), "bias": jnp.asarray(d["conv3_b"])},
                "proj": {"kernel": jnp.asarray(pw), "bias": jnp.asarray(d["proj_b"])},
                "proj_ln": {"scale": jnp.asarray(d["ln_w"]), "bias": jnp.asarray(d["ln_b"])},
            }
            out_j = np.asarray(enc.apply({"params": params}, x))
            diff = float(np.max(np.abs(out_j - d["out_t"])))
            print("MAXDIFF", diff)
            sys.exit(0 if diff < 1e-5 else 1)
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(SHIM_DIR), str(OGBENCH_IMPLS), env.get("PYTHONPATH", "")]
        )
        env["JAX_PLATFORMS"] = "cpu"
        proc = subprocess.run(
            [jax_python(), "-c", jax_code, str(npz)],
            env=env, capture_output=True, text=True,
        )
    assert proc.returncode == 0, (
        f"parity failed\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert "MAXDIFF" in proc.stdout
```

- [ ] **Step 2: Run it — expect PASS (or SKIP without `.venv-jax`)**

Run: `.venv/bin/python -m pytest tests/test_ogbench_drq_parity.py -v -s`
Expected: PASS; `MAXDIFF <small>` printed (< 1e-5). If it FAILS, the printed `MAXDIFF` localizes the discrepancy — re-check, in order: conv `padding='VALID'`, conv kernel transpose `(2,3,1,0)`, the proj flatten-reorder `reshape(256,32,25,25)→transpose(2,3,1,0)→reshape(20000,256)`, LayerNorm `epsilon=1e-5`, and the normalization boundary (torch gets `x/255-0.5`, jax gets raw `x_raw`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_ogbench_drq_parity.py
git commit -m "test(drq-ogbench): weight-transplant parity gate (jax == fb torch < 1e-5)"
```

---

## Task 5: sitecustomize injection

**Files:**
- Modify: `tools/wandb_mode_shim/sitecustomize.py` (append a new top-level guarded block, after the existing wandb block)
- Test: `tests/test_ogbench_drq_injection.py`

- [ ] **Step 1: Write the failing test**

```python
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS

_CHECK = """
    import utils.encoders as enc
    import utils.datasets as ds
    assert "drq" in enc.encoder_modules, "drq not registered"
    patched = getattr(ds.GCDataset.augment, "_ogbench_drq_patched", False)
    print("PATCHED", patched)
"""


def test_encoder_registered_and_augment_patched_with_flag():
    proc = run_jax(_CHECK, env_extra={"OGBENCH_DRQ_FRONTEND": "1"},
                   cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "PATCHED True" in proc.stdout


def test_encoder_registered_but_augment_native_without_flag():
    proc = run_jax(_CHECK, cwd=str(OGBENCH_IMPLS))  # no flag
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "PATCHED False" in proc.stdout  # drq registered, augment untouched
```

- [ ] **Step 2: Run it — expect SKIP or FAIL (`drq not registered`)**

Run: `.venv/bin/python -m pytest tests/test_ogbench_drq_injection.py -v`
Expected: SKIP without `.venv-jax`; otherwise FAIL (assertion on `"drq" in encoder_modules`).

- [ ] **Step 3: Append the injection block to `sitecustomize.py`**

Add at the **end** of `tools/wandb_mode_shim/sitecustomize.py` (a new, independent top-level block — must not depend on the wandb env flags):

```python
# --- DrQ visual front-end injection (independent of the wandb block above) ---
# Register the JAX DrQ encoder so configs may select encoder='drq', and — only
# when OGBENCH_DRQ_FRONTEND=1 — swap GCDataset.augment for FB's random_shifts.
# Guarded: this file is also auto-imported by non-OGBench/non-jax processes (e.g.
# the wandb-shim tests under .venv), where these imports must fail silently.
# run_gciql.py runs `python main.py` with cwd=third_party/ogbench/impls, so
# sys.path[0] is that dir and `utils.*` import here at interpreter startup.
try:
    import utils.encoders as _ogb_encoders
    from ogbench_drq import DrQEncoder as _DrQEncoder

    _ogb_encoders.encoder_modules.setdefault("drq", _DrQEncoder)

    if os.environ.get("OGBENCH_DRQ_FRONTEND") == "1":
        import utils.datasets as _ogb_datasets
        from ogbench_drq import gcdataset_augment as _drq_augment

        _ogb_datasets.GCDataset.augment = _drq_augment
except Exception:
    pass
```

- [ ] **Step 4: Run the injection tests AND the existing wandb-shim tests (regression)**

Run: `.venv/bin/python -m pytest tests/test_ogbench_drq_injection.py tests/test_wandb_mode_shim.py -v`
Expected: injection tests PASS (or SKIP without `.venv-jax`); both `test_wandb_mode_shim` tests still PASS (the new block is guarded and inert under `.venv`).

- [ ] **Step 5: Commit**

```bash
git add tools/wandb_mode_shim/sitecustomize.py tests/test_ogbench_drq_injection.py
git commit -m "feat(drq-ogbench): register drq encoder + gate augment swap in shim"
```

---

## Task 6: run_gciql sets OGBENCH_DRQ_FRONTEND

**Files:**
- Modify: `run_gciql.py:79-116` (extract `build_child_env`; call it from `main`)
- Test: `tests/test_run_gciql_drq_env.py`

- [ ] **Step 1: Write the failing test**

```python
from run_gciql import build_child_env

_BASE = dict(
    env_name="visual-cube-single-play-v0",
    wandb_mode="online",
    agent_overrides={"encoder": "drq", "frame_stack": 3, "p_aug": 1.0},
)


def test_drq_encoder_sets_frontend_flag():
    env = build_child_env(_BASE, {})
    assert env["OGBENCH_DRQ_FRONTEND"] == "1"


def test_impala_encoder_does_not_set_frontend_flag():
    plain = {**_BASE, "agent_overrides": {"encoder": "impala_small", "p_aug": 0.5}}
    env = build_child_env(plain, {})
    assert "OGBENCH_DRQ_FRONTEND" not in env


def test_build_child_env_still_sets_core_shim_vars():
    env = build_child_env(_BASE, {})
    assert env["GCIQL_FB_ENV"] == "visual-cube-single-play-v0"
    assert "tools/wandb_mode_shim" in env["PYTHONPATH"]
    assert env["MUJOCO_GL"] == "egl"
```

- [ ] **Step 2: Run it — expect FAIL (`cannot import name 'build_child_env'`)**

Run: `.venv/bin/python -m pytest tests/test_run_gciql_drq_env.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Refactor `run_gciql.py`** — extract the env build (currently `run_gciql.py:79-116`) into a pure function and add the flag.

Add this function above `main` (after `_preflight`):

```python
def build_child_env(plain: Mapping[str, Any], base_env: Mapping[str, str]) -> dict:
    """Build the OGBench child env from the resolved config + a base environment.

    Pure (no os.environ access) so it is unit-testable. Mirrors the wandb-shim
    contract documented in run_gciql's module docstring and sets
    OGBENCH_DRQ_FRONTEND=1 when the config selects the DrQ visual encoder.
    """
    env = dict(base_env)
    env["MUJOCO_GL"] = "egl"
    cvd = base_env.get("CUDA_VISIBLE_DEVICES", "0")
    env["CUDA_VISIBLE_DEVICES"] = cvd
    egl_dev = cvd.split(",")[0].strip() or "0"
    env.setdefault("MUJOCO_EGL_DEVICE_ID", egl_dev)
    shim_dir = str(REPO_ROOT / "tools" / "wandb_mode_shim")
    existing = base_env.get("PYTHONPATH")
    env["PYTHONPATH"] = shim_dir + (os.pathsep + existing if existing else "")
    env["GCIQL_FB_ENV"] = str(plain["env_name"])
    env["GCIQL_WANDB_ENTITY"] = str(plain.get("wandb_entity", "amsks"))
    env["GCIQL_WANDB_PROJECT"] = str(plain.get("wandb_project", "factored-fb"))
    mode = str(plain.get("wandb_mode", "online"))
    env.pop("GCIQL_WANDB_FORCE_MODE", None)
    if mode in ("offline", "disabled"):
        env["GCIQL_WANDB_FORCE_MODE"] = mode
        env["WANDB_MODE"] = "offline" if mode == "disabled" else mode
    # DrQ visual front-end: gate the sitecustomize augment swap on the encoder.
    overrides = plain.get("agent_overrides") or {}
    if str(overrides.get("encoder")) == "drq":
        env["OGBENCH_DRQ_FRONTEND"] = "1"
    return env
```

Then replace the inline block in `main` (the lines from `env = os.environ.copy()` through the `if mode in ("offline", "disabled"):` block, `run_gciql.py:79-116`) with:

```python
    env = build_child_env(plain, os.environ.copy())
```

Keep the comments now living in `build_child_env`; the `main` body keeps everything after (run_meta, subprocess.run) unchanged.

- [ ] **Step 4: Run the new test AND the existing runner test (regression)**

Run: `.venv/bin/python -m pytest tests/test_run_gciql_drq_env.py tests/test_run_gciql_visual.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add run_gciql.py tests/test_run_gciql_drq_env.py
git commit -m "feat(drq-ogbench): set OGBENCH_DRQ_FRONTEND when encoder=drq (build_child_env)"
```

---

## Task 7: Hydra configs for the DrQ front-end

**Files:**
- Create: `configs/gciql/cube_single_visual_drq.yaml`
- Create: `configs/gciql/cube_single_visual_gcivl_drq.yaml`
- Test: `tests/test_drq_configs.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from hydra import compose, initialize

from run_gciql import build_ogbench_argv


def _argv(name):
    with initialize(version_base="1.3", config_path="../configs/gciql"):
        cfg = compose(config_name=name)
    cfg_d = dict(cfg)
    cfg_d["output_root"] = "/tmp/out"
    return build_ogbench_argv(cfg_d)


def test_gciql_drq_config_argv():
    argv = _argv("cube_single_visual_drq")
    assert "--agent=agents/gciql.py" in argv
    assert "--agent.encoder=drq" in argv
    assert "--agent.frame_stack=3" in argv
    assert "--agent.p_aug=1.0" in argv
    assert "--agent.alpha=1.0" in argv
    assert "--run_group=factored-fb-gciql-pixel-drq" in argv
    assert "--env_name=visual-cube-single-play-v0" in argv


def test_gcivl_drq_config_argv():
    argv = _argv("cube_single_visual_gcivl_drq")
    assert "--agent=agents/gcivl.py" in argv
    assert "--agent.encoder=drq" in argv
    assert "--agent.frame_stack=3" in argv
    assert "--agent.p_aug=1.0" in argv
    assert "--agent.alpha=10.0" in argv
    assert "--run_group=factored-fb-gcivl-pixel-drq" in argv
```

- [ ] **Step 2: Run it — expect FAIL (configs missing)**

Run: `.venv/bin/python -m pytest tests/test_drq_configs.py -v`
Expected: FAIL (Hydra `MissingConfigException`).

- [ ] **Step 3: Create `configs/gciql/cube_single_visual_drq.yaml`**

```yaml
# Visual cube GCIQL with FB's DrQ visual front-end (encoder + random_shifts
# augmentation + frame_stack=3 + RGBNorm), so the only FB-vs-baseline difference
# is the RL algorithm. encoder=drq is registered, and p_aug=1.0 + the augment
# swap are wired, by tools/wandb_mode_shim (gated on OGBENCH_DRQ_FRONTEND, which
# run_gciql.py sets because encoder == 'drq'). Deltas vs cube_single_visual.yaml:
# encoder, frame_stack, p_aug, run_group.
defaults:
  - storage: shm
  - _self_
variant: visual
env_name: visual-cube-single-play-v0
agent_file: agents/gciql.py
seed: 0
train_steps: 500000
log_interval: 5000
eval_interval: 100000
save_interval: 50000
eval_episodes: 50
eval_on_cpu: 0
run_group: factored-fb-gciql-pixel-drq
agent_overrides:
  alpha: 1.0
  batch_size: 256
  encoder: drq
  frame_stack: 3
  p_aug: 1.0
wandb_mode: online
```

- [ ] **Step 4: Create `configs/gciql/cube_single_visual_gcivl_drq.yaml`**

```yaml
# Visual cube GCIVL with FB's DrQ visual front-end. Identical to
# cube_single_visual_drq.yaml except the agent (agents/gcivl.py), alpha (10.0,
# OGBench's GCIVL visual recipe), and run_group.
defaults:
  - storage: shm
  - _self_
variant: visual
env_name: visual-cube-single-play-v0
agent_file: agents/gcivl.py
seed: 0
train_steps: 500000
log_interval: 5000
eval_interval: 100000
save_interval: 50000
eval_episodes: 50
eval_on_cpu: 0
run_group: factored-fb-gcivl-pixel-drq
agent_overrides:
  alpha: 10.0
  batch_size: 256
  encoder: drq
  frame_stack: 3
  p_aug: 1.0
wandb_mode: online
```

- [ ] **Step 5: Run the test — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_drq_configs.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/gciql/cube_single_visual_drq.yaml configs/gciql/cube_single_visual_gcivl_drq.yaml tests/test_drq_configs.py
git commit -m "feat(drq-ogbench): GCIQL/GCIVL DrQ-front-end Hydra configs + argv tests"
```

---

## Task 8: Full suite + behavioral smoke run (manual gate)

This task has no new code — it verifies the whole pipeline end-to-end. The smoke
run is **manual** (needs a GPU and `.venv-jax`); do not script it as a unit test.

- [ ] **Step 1: Run the entire test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all DrQ tests PASS or SKIP (skip only if `.venv-jax` is absent); no regressions. (Per repo memory, ~6 unrelated modules may fail on missing `pandas` — those are pre-existing, not caused by this work. Confirm the failure list matches that baseline.)

- [ ] **Step 2: GCIQL DrQ smoke run (~2k steps, single seed)**

Ensure `/dev/shm/.venv-jax` exists (else `make install-jax`) and a healthy GPU is free (spec: GPUs 3,5,6,7; avoid degraded 0,1,2,4). Then:

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python run_gciql.py \
    --config-name cube_single_visual_drq \
    storage=shm seed=0 wandb_mode=disabled \
    train_steps=2000 eval_interval=1000 eval_episodes=2 save_interval=2000 \
    run_group=drq-smoke
```

Expected / check:
- starts without error (encoder `drq` resolves; no `KeyError: 'drq'`),
- training loss decreases over the 2k steps,
- the eval at step 1000/2000 executes (no `isinstance`/wandb crash, no shape error from the 18-channel concat),
- throughput is healthy (~20–25 it/s, **not** the ~20 s/it degraded-GPU pathology),
- a checkpoint `params_2000.pkl` is written under `/dev/shm/gciql_outputs/.../drq-smoke/`.

- [ ] **Step 3: GCIVL DrQ smoke run (~2k steps)**

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python run_gciql.py \
    --config-name cube_single_visual_gcivl_drq \
    storage=shm seed=0 wandb_mode=disabled \
    train_steps=2000 eval_interval=1000 eval_episodes=2 save_interval=2000 \
    run_group=drq-smoke
```

Expected: same health checks as Step 2, with `--agent=agents/gcivl.py`.

- [ ] **Step 4: Record outcome**

Note the two smoke-run results (it/s, loss trend, checkpoint path) in the run log
or a short notes file. Do **not** launch the full seeds-0–4 / 500k experiment from
this plan — that's a separate launch decision (see spec "Experiment plan").

---

## Self-review (completed during planning)

- **Spec coverage:** Component 1 encoder → Task 2 + parity Task 4; Component 2 augment → Task 3; Component 3 injection → Task 5 + run_gciql gating Task 6; Component 4 configs → Task 7; testing items 1–4 → Tasks 2,3,4,8; GCEncoder caveat → documented (Critical context), exercised by the 18-channel smoke run (Task 8). Experiment plan → intentionally out of scope (Task 8 Step 4).
- **Placeholders:** none — every code/test step is complete; the only manual step (smoke run) is explicitly non-automated with concrete commands and pass criteria.
- **Type/name consistency:** `DrQEncoder(feature_dim=256)`, param keys `conv0..conv3`/`proj`/`proj_ln`, `_batched_shift(imgs, shifts, pad)`, `random_shifts_batch(batch, keys, pad, rng)`, `gcdataset_augment(self, batch, keys)`, `build_child_env(plain, base_env)`, `OGBENCH_DRQ_FRONTEND` — used identically across Tasks 2–7.
