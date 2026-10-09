# Multi-env measure eval (spec 14) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score the four unscored torch scene / cube-double checkpoints on the measure axis (non-negativity, calibration, zero-shot transfer), which today is a cube-single-only axis.

**Architecture:** One registry module (`scripts/rescuing_fb/env_geometry.py`) owns "the measure coordinate and region rule for env X" as a `HitSpec` of per-observation-block `(start, stop, eps)` triples combined as a **conjunction**. `FullMeasure` is generalized from a single 3-D L2 ball to that conjunction, using a Chebyshev KDTree on per-block-scaled coordinates as an exact **superset** filter. The three torch eval scripts then drop their hardcoded `CUBE_OBS`/`EPS` and read the spec from each checkpoint's own recorded `cfg.hit_spec`.

**Tech Stack:** Python 3.11, numpy, scipy `cKDTree`, torch (eval only), pytest. Interpreter is `.venv/bin/python` — `uv run` is broken in this repo.

## Global Constraints

- Spec: `PAPER/rescuing-fb/specs/14-multienv-measure-eval.md`. Read it before Task 1.
- Tests run as `.venv/bin/python -m pytest` (never `uv run`, never bare `pytest`).
- **Backwards compatibility is a hard requirement.** Every cube-single number already in `results/spec10_torchfb/` must be reproducible by the new code. The existing 8 tests in `tests/test_full_measure.py` must pass **unmodified**.
- Trained hit specs, verbatim — these are the ground truth and must match the checkpoints' recorded `cfg.hit_spec` exactly:
  - `cube-single-play-v0`: `19:22:0.4`
  - `cube-double-play-v0`: `19:22:0.4,28:31:0.4`
  - `scene-play-v0`: `19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6`
- Region multipliers replace absolute half-widths: `MULTS = [1.0, 2.0, 4.0]`. On cube-single (eps 0.4) these reproduce the archived half-widths 0.4 / 0.8 / 1.6 exactly.
- Checkpoints under evaluation: `~/exp_spec10/torchfb/{p1,p2}_{scn,cdbl}_s42` (seed 43 chains are training now under tmux `menv43`).
- Commit style: no Claude co-author line.
- `PAPER/`, `results/`, `analysis/` are gitignored artifacts — never `git add` them.

---

### Task 1: `env_geometry.py` — HitSpec and the env registry

**Files:**
- Create: `scripts/rescuing_fb/env_geometry.py`
- Test: `tests/test_env_geometry.py`

**Interfaces:**
- Consumes: nothing (leaf module, pure numpy).
- Produces:
  - `HitSpec.parse(spec: str) -> HitSpec`, `HitSpec.single(a: int, b: int, eps: float) -> HitSpec`
  - `HitSpec.blocks: tuple[tuple[int, int, float], ...]`, `HitSpec.dim: int`, `HitSpec.radii: np.ndarray` (per-**column** eps, length `dim`)
  - `HitSpec.coords(obs: np.ndarray) -> np.ndarray` `[n, dim]`
  - `HitSpec.scaled(coords: np.ndarray) -> np.ndarray` `[n, dim]`
  - `HitSpec.contains(coords, center, mult=1.0) -> np.ndarray[bool]` (conjunctive L2 balls)
  - `HitSpec.box_contains(coords, center, mult=1.0) -> np.ndarray[bool]` (conjunctive L∞ boxes)
  - `EnvGeometry(env_name, hit_spec, probe_centers, dataset_name)`
  - `GEOMETRY: dict[str, EnvGeometry]`, `geometry_for(domain: str, hit_spec: str = "") -> EnvGeometry`
  - `resolve_dataset(data_path: str, domain: str) -> str`
  - `episode_bounds(terminals: np.ndarray) -> np.ndarray` `[n_ep, 2]` start/stop row indices
  - `hindsight_rows(bounds, start_row, rng, k) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_env_geometry.py
"""Spec 14 §3.1 — one registry for the measure coordinate and region rule."""
import numpy as np
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "rescuing_fb"))

from env_geometry import (HitSpec, GEOMETRY, geometry_for, resolve_dataset,
                          episode_bounds, hindsight_rows)

SCENE = "19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6"


def test_parse_matches_agent_semantics():
    """Block-for-block agreement with agents/fb/measure/agent.py's parser."""
    spec = HitSpec.parse(SCENE)
    assert spec.blocks == ((19, 22, 0.4), (28, 30, 0.5), (32, 34, 0.5),
                           (36, 37, 0.72), (38, 39, 0.6))
    assert spec.dim == 3 + 2 + 2 + 1 + 1


def test_parse_rejects_bad_blocks():
    for bad in ["19:19:0.4", "22:19:0.4", "19:22:0", "19:22:-1", ""]:
        with pytest.raises((AssertionError, ValueError)):
            HitSpec.parse(bad)


def test_radii_are_per_column():
    spec = HitSpec.parse("19:22:0.4,28:30:0.5")
    assert np.allclose(spec.radii, [0.4, 0.4, 0.4, 0.5, 0.5])


def test_coords_concatenates_blocks():
    spec = HitSpec.parse("1:3:0.5,5:6:0.5")
    obs = np.arange(24, dtype=np.float64).reshape(3, 8)
    got = spec.coords(obs)
    assert got.shape == (3, 3)
    assert np.allclose(got[0], [1, 2, 5])


def test_contains_is_conjunctive_not_joint_l2():
    """Two blocks each just inside their own radius = HIT, even though the
    joint 4-D L2 norm exceeds either radius. This is the whole point."""
    spec = HitSpec.parse("0:2:1.0,2:4:1.0")
    center = np.zeros(4)
    pts = np.array([[0.9, 0.0, 0.9, 0.0],    # both blocks in  -> True
                    [1.1, 0.0, 0.0, 0.0],    # block 0 out     -> False
                    [0.0, 0.0, 1.1, 0.0]])   # block 1 out     -> False
    assert spec.contains(pts, center).tolist() == [True, False, False]


def test_contains_respects_per_block_radii_and_mult():
    spec = HitSpec.parse("0:1:1.0,1:2:2.0")
    center = np.zeros(2)
    pts = np.array([[0.0, 1.5], [1.5, 0.0]])
    assert spec.contains(pts, center).tolist() == [True, False]
    assert spec.contains(pts, center, mult=2.0).tolist() == [True, True]


def test_box_contains_is_per_block_linf():
    spec = HitSpec.parse("0:2:1.0")
    center = np.zeros(2)
    pts = np.array([[0.9, 0.9], [1.1, 0.0]])
    # L-inf box admits the corner point that the L2 ball would reject
    assert spec.box_contains(pts, center).tolist() == [True, False]
    assert spec.contains(pts, center).tolist() == [False, False]


def test_registry_specs_are_the_trained_ones():
    assert GEOMETRY["cube-single-play-v0"].hit_spec.blocks == ((19, 22, 0.4),)
    assert GEOMETRY["cube-double-play-v0"].hit_spec.blocks == ((19, 22, 0.4), (28, 31, 0.4))
    assert GEOMETRY["scene-play-v0"].hit_spec == HitSpec.parse(SCENE)


def test_registry_probe_center_policy():
    assert GEOMETRY["cube-single-play-v0"].probe_centers == "random"
    assert GEOMETRY["cube-double-play-v0"].probe_centers == "hindsight"
    assert GEOMETRY["scene-play-v0"].probe_centers == "hindsight"


def test_geometry_for_prefers_checkpoint_spec():
    """The checkpoint's own hit_spec wins over the registry default, so a
    checkpoint trained at a non-default width scores against what it learned."""
    g = geometry_for("cube-double-play-v0", hit_spec="19:22:0.9,28:31:0.9")
    assert g.hit_spec.blocks == ((19, 22, 0.9), (28, 31, 0.9))
    g0 = geometry_for("cube-double-play-v0")
    assert g0.hit_spec.blocks == ((19, 22, 0.4), (28, 31, 0.4))


def test_geometry_for_rejects_unknown_env():
    with pytest.raises(KeyError):
        geometry_for("puzzle-4x4-play-v0")


def test_resolve_dataset_prefers_recorded_then_falls_back(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".ogbench" / "data").mkdir(parents=True)
    (home / ".ogbench" / "data" / "scene-play-v0.npz").write_bytes(b"x")
    monkeypatch.setenv("HOME", str(home))

    recorded = tmp_path / "shm"
    recorded.mkdir()
    # fallback: recorded dir has no top-level npz (the real /dev/shm layout)
    assert resolve_dataset(str(recorded), "scene-play-v0").startswith(str(home))

    (recorded / "scene-play-v0.npz").write_bytes(b"x")
    assert resolve_dataset(str(recorded), "scene-play-v0").startswith(str(recorded))


def test_resolve_dataset_errors_loudly_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "nowhere"))
    with pytest.raises(FileNotFoundError):
        resolve_dataset(str(tmp_path / "also-nowhere"), "scene-play-v0")


def test_episode_bounds_and_hindsight_rows():
    terminals = np.zeros(10, bool)
    terminals[[3, 9]] = True          # episodes rows 0..3 and 4..9
    b = episode_bounds(terminals)
    assert b.tolist() == [[0, 4], [4, 10]]

    rng = np.random.default_rng(0)
    rows = hindsight_rows(b, start_row=5, rng=rng, k=3)
    assert len(rows) == 3
    assert set(rows.tolist()) <= {6, 7, 8, 9}      # strictly after, same episode


def test_hindsight_rows_at_episode_end_returns_empty():
    b = np.array([[0, 4]])
    rng = np.random.default_rng(0)
    assert len(hindsight_rows(b, start_row=3, rng=rng, k=3)) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_env_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'env_geometry'`

- [ ] **Step 3: Write the module**

```python
# scripts/rescuing_fb/env_geometry.py
"""Spec 14 §3.1 — the single definition of "the measure coordinate and region
rule" for an environment.

Every eval script in both stacks imports this instead of redeclaring
`CUBE_OBS = slice(19, 22)` / `EPS = 0.4`. Manipspace success criteria are
CONJUNCTIONS (cube-double: both cubes in place; scene: cube AND drawer AND
window AND both buttons), so a region is a tuple of per-observation-block balls
combined with AND -- mirroring `agents/fb/measure/agent.py`'s `hit_spec` and the
jax `measure_hit_spec` exactly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HitSpec:
    """Conjunction of per-block balls. blocks = ((start, stop, eps), ...)."""

    blocks: tuple[tuple[int, int, float], ...]

    @classmethod
    def parse(cls, spec: str) -> "HitSpec":
        """Parse "a:b:eps,..." — the same grammar the agents parse."""
        blocks = []
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            a, b, e = part.split(":")
            a, b, e = int(a), int(b), float(e)
            assert 0 <= a < b and e > 0.0, f"bad hit_spec block {part!r}"
            blocks.append((a, b, e))
        assert blocks, f"empty hit_spec {spec!r}"
        return cls(tuple(blocks))

    @classmethod
    def single(cls, a: int, b: int, eps: float) -> "HitSpec":
        return cls(((int(a), int(b), float(eps)),))

    @property
    def dim(self) -> int:
        return sum(b - a for a, b, _ in self.blocks)

    @property
    def radii(self) -> np.ndarray:
        """Per-COLUMN eps, length `dim` (each block's eps repeated)."""
        return np.concatenate([np.full(b - a, e) for a, b, e in self.blocks])

    def coords(self, obs: np.ndarray) -> np.ndarray:
        """[n, obs_dim] -> [n, dim], blocks concatenated in spec order."""
        obs = np.atleast_2d(np.asarray(obs, np.float64))
        return np.concatenate([obs[:, a:b] for a, b, _ in self.blocks], axis=1)

    def scaled(self, coords: np.ndarray) -> np.ndarray:
        """Divide each column by its block's eps (unit-radius coordinates)."""
        return np.asarray(coords, np.float64) / self.radii

    def _slices(self):
        off = 0
        for a, b, e in self.blocks:
            w = b - a
            yield slice(off, off + w), e
            off += w

    def contains(self, coords, center, mult: float = 1.0) -> np.ndarray:
        """Conjunctive L2: every block within `mult * eps_block` of center."""
        coords = np.atleast_2d(np.asarray(coords, np.float64))
        center = np.asarray(center, np.float64).reshape(-1)
        out = np.ones(len(coords), bool)
        for sl, e in self._slices():
            d = np.linalg.norm(coords[:, sl] - center[sl], axis=1)
            out &= d <= mult * e
        return out

    def box_contains(self, coords, center, mult: float = 1.0) -> np.ndarray:
        """Conjunctive L-inf: every coordinate within `mult * eps_block`."""
        coords = np.atleast_2d(np.asarray(coords, np.float64))
        center = np.asarray(center, np.float64).reshape(-1)
        return np.all(np.abs(coords - center) <= mult * self.radii, axis=1)


@dataclass(frozen=True)
class EnvGeometry:
    env_name: str
    hit_spec: HitSpec
    probe_centers: str      # 'random' (archived protocol) | 'hindsight'
    dataset_name: str


GEOMETRY: dict[str, EnvGeometry] = {
    "cube-single-play-v0": EnvGeometry(
        "cube-single-play-v0", HitSpec.parse("19:22:0.4"),
        "random", "cube-single-play-v0"),
    "cube-double-play-v0": EnvGeometry(
        "cube-double-play-v0", HitSpec.parse("19:22:0.4,28:31:0.4"),
        "hindsight", "cube-double-play-v0"),
    "scene-play-v0": EnvGeometry(
        "scene-play-v0",
        HitSpec.parse("19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6"),
        "hindsight", "scene-play-v0"),
}

MULTS = (1.0, 2.0, 4.0)     # region size multipliers; on cube-single (eps=0.4)
                            # these are the archived half-widths 0.4/0.8/1.6.


def geometry_for(domain: str, hit_spec: str = "") -> EnvGeometry:
    """Registry entry for `domain`, with the CHECKPOINT's own hit_spec winning
    when supplied — the geometry travels with the weights (spec 14 §3.4)."""
    base = GEOMETRY[domain]
    if str(hit_spec).strip():
        return EnvGeometry(base.env_name, HitSpec.parse(hit_spec),
                           base.probe_centers, base.dataset_name)
    return base


def resolve_dataset(data_path: str, domain: str) -> str:
    """Recorded path first, then ~/.ogbench/data, else a loud error.

    The s42 multi-env runs recorded data_path=/dev/shm/factored-fb/datasets,
    which holds cube-single-play-v0.npz but for scene/cube-double only
    <env>/buffer/episode_*.npz — no top-level npz (spec 14 §3.4)."""
    candidates = [os.path.join(os.path.expanduser(data_path), f"{domain}.npz"),
                  os.path.expanduser(f"~/.ogbench/data/{domain}.npz")]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"no dataset for {domain!r}; looked in {candidates}")


def episode_bounds(terminals: np.ndarray) -> np.ndarray:
    """[n_ep, 2] half-open [start, stop) row ranges from a terminals mask."""
    ends = np.flatnonzero(np.asarray(terminals).astype(bool)) + 1
    starts = np.concatenate([[0], ends[:-1]])
    return np.stack([starts, ends], axis=1)


def hindsight_rows(bounds: np.ndarray, start_row: int,
                   rng: np.random.Generator, k: int) -> np.ndarray:
    """Up to k rows strictly AFTER start_row within its own episode.

    These are regions the behaviour policy actually reached from this state, so
    a Monte-Carlo occupancy estimate against them is not degenerate. On
    conjunctive envs (off-diagonal hit rate 0.046% on cube-double) random
    dataset centers give MC truth == 0 and every ratio becomes 0/0."""
    bounds = np.asarray(bounds, np.int64)
    ep = np.searchsorted(bounds[:, 1], start_row, side="right")
    if ep >= len(bounds):
        return np.zeros(0, np.int64)
    lo, hi = int(start_row) + 1, int(bounds[ep, 1])
    if hi <= lo:
        return np.zeros(0, np.int64)
    pool = np.arange(lo, hi)
    if len(pool) <= k:
        return pool
    return np.sort(rng.choice(pool, k, replace=False))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_env_geometry.py -v`
Expected: PASS (15 tests)

- [ ] **Step 5: Cross-check the registry against the live checkpoints**

Run:
```bash
cd /home/adityamohan/git/Factored-FB
for r in p1_scn_s42 p2_scn_s42 p1_cdbl_s42 p2_cdbl_s42; do
  grep -h "^hit_spec:" ~/exp_spec10/torchfb/$r/.hydra/config.yaml
done
```
Expected: scene rows print `19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6`, cube-double rows print `19:22:0.4,28:31:0.4` — identical to `GEOMETRY`. If any differs, the registry is wrong and Task 1 is not done.

- [ ] **Step 6: Commit**

```bash
git add scripts/rescuing_fb/env_geometry.py tests/test_env_geometry.py
git commit -m "feat(spec14): env_geometry registry — conjunctive HitSpec per env"
```

---

### Task 2: Generalize `FullMeasure` to conjunctive specs

**Files:**
- Modify: `scripts/rescuing_fb/full_measure.py:35-105` (module constant, `__init__`, `nhat`, `member_indices`)
- Test: `tests/test_full_measure_conjunctive.py` (new; `tests/test_full_measure.py` stays untouched and must still pass)

**Interfaces:**
- Consumes: `env_geometry.HitSpec` (Task 1).
- Produces: `FullMeasure(coords, spec=None, eps=EPS_OBS, floor_counts=0.5, max_members=64, seed=0, nhat_fn=None)`. When `spec is None` a single-block spec is built from `eps` so all existing callers are unchanged. `member_indices` gains `{'kind': 'ball'|'box', 'center': ..., 'mult': float}`; the legacy `'eps'` and `'half'` keys keep working.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_full_measure_conjunctive.py
"""Spec 14 §3.2 — FullMeasure over conjunctive multi-block regions.

The exactness claim under test: a Chebyshev KDTree on per-block-scaled
coordinates returns a strict SUPERSET of the conjunctive region (per-block
L2 <= 1 implies every coordinate <= 1 implies L-inf <= 1), so filtering those
candidates exactly reproduces a brute-force scan.
"""
import numpy as np
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "rescuing_fb"))

from full_measure import FullMeasure
from env_geometry import HitSpec


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    coords = rng.uniform(-2, 2, (4000, 5))
    spec = HitSpec(((0, 3, 0.4), (3, 5, 0.7)))   # block widths 3 and 2 = dim 5
    return FullMeasure(coords, spec=spec, seed=0), coords, spec


def test_nhat_matches_bruteforce_conjunction(world):
    fm, coords, spec = world
    q = coords[::400]
    brute = np.array([spec.contains(coords, c).mean() for c in q])
    assert np.allclose(fm.nhat(q), np.maximum(brute, fm.floor), atol=1e-12)


def test_member_indices_ball_matches_bruteforce(world):
    fm, coords, spec = world
    for mult in (1.0, 2.0, 4.0):
        c = coords[7]
        got = fm.member_indices({"kind": "ball", "center": c, "mult": mult})
        want = np.flatnonzero(spec.contains(coords, c, mult=mult))
        assert np.array_equal(got, want), f"mult={mult}"


def test_member_indices_box_matches_bruteforce(world):
    fm, coords, spec = world
    c = coords[11]
    got = fm.member_indices({"kind": "box", "center": c, "mult": 2.0})
    want = np.flatnonzero(spec.box_contains(coords, c, mult=2.0))
    assert np.array_equal(got, want)


def test_conjunction_is_not_a_joint_ball(world):
    """A point far in block 0 but near in block 1 must be excluded even when
    its joint L2 distance is small — the failure mode a naive port would have."""
    fm, coords, spec = world
    c = np.zeros(5)
    idx = fm.member_indices({"kind": "ball", "center": c, "mult": 1.0})
    assert np.all(np.linalg.norm(coords[idx][:, :3] - c[:3], axis=1) <= 0.4 + 1e-12)
    assert np.all(np.linalg.norm(coords[idx][:, 3:] - c[3:], axis=1) <= 0.7 + 1e-12)


def test_single_block_spec_equals_legacy_scalar_eps():
    """Backwards compatibility: the archived cube-single protocol is a
    one-block spec and must give bit-identical membership and N-hat."""
    rng = np.random.default_rng(1)
    cube = rng.normal(0, 0.5, (3000, 3))
    legacy = FullMeasure(cube, eps=0.4, seed=0)
    modern = FullMeasure(cube, spec=HitSpec.single(0, 3, 0.4), seed=0)
    q = cube[::250]
    assert np.allclose(legacy.nhat(q), modern.nhat(q), atol=0, rtol=0)
    for c in q[:5]:
        a = legacy.member_indices({"kind": "ball", "center": c, "eps": 0.4})
        b = modern.member_indices({"kind": "ball", "center": c, "mult": 1.0})
        assert np.array_equal(a, b)


def test_legacy_half_key_still_supported():
    rng = np.random.default_rng(2)
    cube = rng.normal(0, 0.5, (2000, 3))
    fm = FullMeasure(cube, eps=0.4, seed=0)
    c = cube[3]
    got = fm.member_indices({"kind": "box", "center": c, "half": np.full(3, 0.8)})
    want = np.flatnonzero(np.all(np.abs(cube - c) <= 0.8, axis=1))
    assert np.array_equal(got, want)


def test_union_still_works_over_conjunctive_parts(world):
    fm, coords, spec = world
    p1 = {"kind": "ball", "center": coords[1], "mult": 1.0}
    p2 = {"kind": "ball", "center": coords[2], "mult": 1.0}
    u = fm.member_indices({"kind": "union", "parts": [p1, p2]})
    want = np.union1d(fm.member_indices(p1), fm.member_indices(p2))
    assert np.array_equal(u, want)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_full_measure_conjunctive.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'spec'`

- [ ] **Step 3: Modify `FullMeasure`**

Replace the `__init__`, `nhat` and `member_indices` bodies in `scripts/rescuing_fb/full_measure.py` with:

```python
    def __init__(self, coords, spec=None, eps: float = EPS_OBS,
                 floor_counts: float = 0.5, max_members: int = 64,
                 seed: int = 0, nhat_fn=None):
        from env_geometry import HitSpec
        self.cube = np.atleast_2d(np.asarray(coords, np.float64))
        self.n = len(self.cube)
        self.spec = spec if spec is not None else \
            HitSpec.single(0, self.cube.shape[1], eps)
        assert self.spec.dim == self.cube.shape[1], \
            f"spec dim {self.spec.dim} != coords dim {self.cube.shape[1]}"
        self.eps = float(eps)
        # Chebyshev tree on per-block-scaled coords: a conjunctive hit implies
        # every block's scaled L2 <= mult, hence scaled L-inf <= mult. The query
        # is therefore a strict SUPERSET, refined by an exact filter below.
        self._scaled = self.spec.scaled(self.cube)
        self.tree = cKDTree(self._scaled)
        self.floor = float(floor_counts) / self.n
        self.rng = np.random.default_rng(seed)
        self.max_members = int(max_members)
        self.nhat_fn = nhat_fn

    def _candidates(self, center, mult: float) -> np.ndarray:
        c = self.spec.scaled(np.asarray(center, np.float64).reshape(1, -1))[0]
        idx = self.tree.query_ball_point(c, float(mult), p=np.inf)
        return np.asarray(sorted(idx), np.int64)

    # ------------------------------------------------------------- N-hat ----
    def nhat(self, centers) -> np.ndarray:
        """rho(region(center)) as a fraction of the dataset, floored."""
        centers = np.atleast_2d(np.asarray(centers, np.float64))
        if self.nhat_fn is not None:
            return np.maximum(np.asarray(self.nhat_fn(centers), np.float64),
                              self.floor)
        counts = np.empty(len(centers), np.float64)
        for i, c in enumerate(centers):
            cand = self._candidates(c, 1.0)
            counts[i] = self.spec.contains(self.cube[cand], c).sum() \
                if len(cand) else 0.0
        return np.maximum(counts / self.n, self.floor)

    # ----------------------------------------------------------- density ----
    # (unchanged)

    # -------------------------------------------------------- set queries ----
    def member_indices(self, query: dict) -> np.ndarray:
        """Dataset row indices inside a query region.
        query: {'kind':'ball','center':[dim],'mult':m} |
               {'kind':'box','center':[dim],'mult':m}  |
               {'kind':'union','parts':[subqueries]}
        Legacy keys 'eps' (ball) and 'half' (box) are still honoured for
        single-block specs so archived protocols reproduce exactly."""
        kind = query["kind"]
        if kind == "union":
            parts = [self.member_indices(p) for p in query["parts"]]
            return np.unique(np.concatenate(parts)) if parts else \
                np.zeros(0, np.int64)

        center = np.asarray(query["center"], np.float64).reshape(-1)
        if kind == "ball":
            if "eps" in query:                       # legacy single-block
                mult = float(query["eps"]) / self.spec.blocks[0][2]
            else:
                mult = float(query.get("mult", 1.0))
            cand = self._candidates(center, mult)
            if not len(cand):
                return np.zeros(0, np.int64)
            return cand[self.spec.contains(self.cube[cand], center, mult=mult)]

        if kind == "box":
            if "half" in query:                      # legacy absolute half-widths
                half = np.asarray(query["half"], np.float64)
                m = np.all(np.abs(self.cube - center) <= half, axis=1)
                return np.nonzero(m)[0]
            mult = float(query.get("mult", 1.0))
            cand = self._candidates(center, mult)
            if not len(cand):
                return np.zeros(0, np.int64)
            return cand[self.spec.box_contains(self.cube[cand], center, mult=mult)]

        raise ValueError(f"unknown query kind {kind!r}")
```

Note: `full_measure.py` must gain `import sys, pathlib` + the sibling-dir path insert only if it is imported from outside `scripts/rescuing_fb/`; it is always imported with that directory on `sys.path`, so a plain `from env_geometry import HitSpec` inside `__init__` is correct.

- [ ] **Step 4: Run both test files**

Run: `.venv/bin/python -m pytest tests/test_full_measure.py tests/test_full_measure_conjunctive.py -v`
Expected: PASS — all 8 pre-existing tests plus the 8 new ones. **If any pre-existing test changed behaviour, stop and fix the compatibility path; do not edit the old test.**

- [ ] **Step 5: Commit**

```bash
git add scripts/rescuing_fb/full_measure.py tests/test_full_measure_conjunctive.py
git commit -m "feat(spec14): FullMeasure over conjunctive multi-block regions"
```

---

### Task 3: Port `negmass_diag.py`

**Files:**
- Modify: `scripts/rescuing_fb/negmass_diag.py:27` (drop `CUBE_OBS`), `:57-64` (dataset load, coords, N̂)
- Test: `tests/test_negmass_port.py`

**Interfaces:**
- Consumes: `env_geometry.geometry_for`, `env_geometry.resolve_dataset`, `FullMeasure`.
- Produces: the JSON gains `"env"`, `"hit_spec"` (the spec string actually used) and keeps every existing key so archived cube-single JSONs stay comparable.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_negmass_port.py
"""Spec 14 §3.4 — negmass_diag reads geometry from the checkpoint, not a
hardcoded cube slice."""
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "rescuing_fb"))

import negmass_diag
from env_geometry import HitSpec


def test_no_hardcoded_cube_slice():
    assert not hasattr(negmass_diag, "CUBE_OBS"), \
        "geometry must come from env_geometry, not a module constant"


def test_density_quartiles_use_conjunctive_nhat():
    """The quartile split must use the conjunctive region mass, so on scene a
    'high density' goal means all five blocks are jointly common."""
    spec = HitSpec(((0, 2, 0.5), (2, 4, 0.5)))
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1.0, (2000, 4))
    goals = rng.normal(0, 1.0, (50, 4))
    got = negmass_diag.region_density(spec, ref, goals)
    want = np.array([spec.contains(ref, g).mean() for g in goals])
    assert np.allclose(got, want)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_negmass_port.py -v`
Expected: FAIL — `AttributeError: module 'negmass_diag' has no attribute 'region_density'` and the `CUBE_OBS` assertion fails.

- [ ] **Step 3: Edit the script**

Delete the `CUBE_OBS = slice(19, 22)` line. Add near the top, after the imports:

```python
def region_density(spec, ref_coords, centers) -> np.ndarray:
    """rho(region(center)) for each center — the conjunctive generalization of
    the old single-ball KDTree count."""
    from full_measure import FullMeasure
    fm = FullMeasure(ref_coords, spec=spec, floor_counts=0.0)
    return fm.nhat(centers)
```

Replace the dataset/coords/N̂ block (`npz = np.load(...)` through the `nhat = cKDTree(...)` line) with:

```python
    from env_geometry import geometry_for, resolve_dataset
    geo = geometry_for(cfg.domain, getattr(cfg, "hit_spec", "") or "")
    spec = geo.hit_spec
    npz = np.load(resolve_dataset(cfg.data_path, cfg.domain))
    obs_all = npz["observations"]
    gidx = rng.choice(len(obs_all), a.goals, replace=False)
    goals = torch.as_tensor(obs_all[gidx], device=dev)
    coords = spec.coords(obs_all[gidx])

    ref = spec.coords(obs_all[rng.choice(len(obs_all), a.rho_samples, replace=False)])
    nhat = region_density(spec, ref, coords)
```

and add to the `res` dict:

```python
        "env": cfg.domain, "hit_spec": ",".join(
            f"{a_}:{b_}:{e_}" for a_, b_, e_ in spec.blocks),
```

- [ ] **Step 4: Run the test**

Run: `.venv/bin/python -m pytest tests/test_negmass_port.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/rescuing_fb/negmass_diag.py tests/test_negmass_port.py
git commit -m "feat(spec14): negmass_diag reads geometry from the checkpoint"
```

---

### Task 4: Port `torchfb_full_measure.py` (L1) with hindsight probe centers

**Files:**
- Modify: `scripts/rescuing_fb/torchfb_full_measure.py:50` (drop `CUBE_OBS`/`EPS`), `:94-101` (dataset + coords), `:111-146` (N̂ + head), `:147-152` (queries), `:176-200` (MC membership), `:205-235` (report)
- Test: `tests/test_l1_probe_centers.py`

**Interfaces:**
- Consumes: `env_geometry.{geometry_for, resolve_dataset, episode_bounds, hindsight_rows, MULTS}`, generalized `FullMeasure`.
- Produces: JSON gains `"env"`, `"hit_spec"`, `"probe_centers"`, `"mults"`, and per-mult `"degenerate_frac"`. `half_widths` is retained for cube-single so archived comparisons still parse.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_l1_probe_centers.py
"""Spec 14 §3.3 — probe-center policy, and the degeneracy guard (G0)."""
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "rescuing_fb"))

import torchfb_full_measure as L1
from env_geometry import HitSpec, episode_bounds


def test_no_hardcoded_geometry():
    assert not hasattr(L1, "CUBE_OBS")
    assert not hasattr(L1, "EPS")


def test_random_policy_draws_anywhere():
    rng = np.random.default_rng(0)
    bounds = episode_bounds(np.array([0, 0, 1, 0, 0, 1], bool))
    rows = L1.probe_rows("random", bounds, start_row=0, n=4,
                         n_rows=6, rng=rng)
    assert len(rows) == 4
    assert set(rows.tolist()) <= set(range(6))


def test_hindsight_policy_stays_in_episode_after_start():
    rng = np.random.default_rng(0)
    bounds = episode_bounds(np.array([0, 0, 1, 0, 0, 1], bool))
    rows = L1.probe_rows("hindsight", bounds, start_row=3, n=2,
                         n_rows=6, rng=rng)
    assert set(rows.tolist()) <= {4, 5}


def test_hindsight_falls_back_when_episode_tail_is_empty():
    """A start at the last row of its episode has no future; the run must not
    crash — it reports zero probes for that start instead."""
    rng = np.random.default_rng(0)
    bounds = episode_bounds(np.array([0, 0, 1], bool))
    assert len(L1.probe_rows("hindsight", bounds, start_row=2, n=3,
                             n_rows=3, rng=rng)) == 0


def test_degenerate_frac_counts_zero_truth_probes():
    recs = [{"mult": 1.0, "mc": 0.0}, {"mult": 1.0, "mc": 0.5},
            {"mult": 2.0, "mc": 0.0}, {"mult": 2.0, "mc": 0.0}]
    assert L1.degenerate_frac(recs, 1.0) == 0.5
    assert L1.degenerate_frac(recs, 2.0) == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_l1_probe_centers.py -v`
Expected: FAIL — `AttributeError: module has no attribute 'probe_rows'`

- [ ] **Step 3: Edit the script**

Delete `CUBE_OBS = slice(19, 22)` and `EPS = 0.4`. Add module-level helpers:

```python
def probe_rows(policy: str, bounds, start_row: int, n: int,
               n_rows: int, rng) -> np.ndarray:
    """Rows whose coordinates become query centers for one start.

    'random'    — uniform over the dataset (the archived cube-single protocol).
    'hindsight' — strictly after start_row within its own episode, so the MC
                  occupancy has something to count (spec 14 §2.1/§3.3).
    """
    from env_geometry import hindsight_rows
    if policy == "random":
        return rng.choice(n_rows, min(n, n_rows), replace=False)
    if policy == "hindsight":
        return hindsight_rows(bounds, start_row, rng, n)
    raise ValueError(f"unknown probe-center policy {policy!r}")


def degenerate_frac(recs, mult) -> float:
    """G0: fraction of probes at this multiplier whose MC truth was exactly 0."""
    R = [r for r in recs if r["mult"] == mult]
    if not R:
        return float("nan")
    return float(np.mean([r["mc"] == 0.0 for r in R]))
```

Then, in `main()`:

1. After `cfg = load_cfg(...)`, add
   ```python
   from env_geometry import geometry_for, resolve_dataset, episode_bounds, MULTS
   geo = geometry_for(cfg.domain, getattr(cfg, "hit_spec", "") or "")
   spec = geo.hit_spec
   ```
2. Replace the dataset load with
   ```python
   npz = np.load(resolve_dataset(cfg.data_path, cfg.domain))
   obs_all, qpos_all, qvel_all = npz["observations"], npz["qpos"], npz["qvel"]
   bounds = episode_bounds(npz["terminals"])
   ```
   and `cube_ds = rho_obs[:, CUBE_OBS]` with `cube_ds = spec.coords(rho_obs)`.
3. Replace `fm_exact = FullMeasure(cube_ds, eps=EPS)` with
   `fm_exact = FullMeasure(cube_ds, spec=spec)` and likewise
   `fm_head = FullMeasure(cube_ds, spec=spec, nhat_fn=nhat_fn)`.
4. Replace the fixed query list with per-start, policy-driven centers. Move the
   query construction **inside** the `for si, (s_row, g_row)` loop, since
   hindsight centers depend on the start:
   ```python
       centers_rows = probe_rows(geo.probe_centers, bounds, int(s_row),
                                 a.boxes, len(obs_all), rng)
       queries = []
       for mult in MULTS:
           for c in spec.coords(obs_all[centers_rows]):
               queries.append({"kind": "box", "center": c, "mult": mult})
       if not queries:
           continue                      # start had no episode tail
       occ = np.zeros(len(queries))
   ```
5. In the MC loop replace the membership test with the spec's own rule:
   ```python
                   c = spec.coords(ob[None])[0]
                   for qi, q in enumerate(queries):
                       if spec.box_contains(c[None], q["center"],
                                            mult=q["mult"])[0]:
                           occ[qi] += g
   ```
6. In the record dict replace `"hw": q["hw"]` with `"mult": q["mult"]`, and in
   the report loop iterate `for mult in MULTS` keyed by `str(mult)`, adding
   ```python
           out["summary"][str(mult)]["degenerate_frac"] = degenerate_frac(recs, mult)
           out["summary"][str(mult)]["abs_half_widths"] = (
               (mult * spec.radii).round(3).tolist())
   ```
7. Add to `out`: `"env": cfg.domain`, `"probe_centers": geo.probe_centers`,
   `"mults": list(MULTS)`, `"hit_spec": ",".join(f"{a_}:{b_}:{e_}" for a_, b_, e_ in spec.blocks)`.

- [ ] **Step 4: Run the test**

Run: `.venv/bin/python -m pytest tests/test_l1_probe_centers.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/rescuing_fb/torchfb_full_measure.py tests/test_l1_probe_centers.py
git commit -m "feat(spec14): L1 over conjunctive regions with hindsight probe centers"
```

---

### Task 5: Port `torchfb_zeroshot_l2.py`

**Files:**
- Modify: `scripts/rescuing_fb/torchfb_zeroshot_l2.py:40` (drop `CUBE_OBS`/`TRAIN_EPS`), `:44-60` (`make_rewards`), `:127-145` (dataset + N̂ weight), `:168-205` (`cube_of` and the rollout)
- Test: `tests/test_l2_reward_families.py`

**Interfaces:**
- Consumes: `env_geometry.{geometry_for, resolve_dataset}`, generalized `FullMeasure`.
- Produces: `make_rewards(coords, rng, per_family, spec)` — families are now defined by **multipliers on the trained per-block radii**, so `ball1.6` means the same region size on every env. JSON gains `"env"`, `"hit_spec"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_l2_reward_families.py
"""Spec 14 §3.4 — reward families scale with the trained radii, not literal cm."""
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "rescuing_fb"))

import torchfb_zeroshot_l2 as L2
from env_geometry import HitSpec


def test_no_hardcoded_geometry():
    assert not hasattr(L2, "CUBE_OBS")
    assert not hasattr(L2, "TRAIN_EPS")


def test_family_names_are_multiplier_based():
    spec = HitSpec(((0, 2, 0.5),))
    rng = np.random.default_rng(0)
    fams = {r["family"] for r in L2.make_rewards(
        np.zeros((10, 2)), rng, per_family=1, spec=spec)}
    assert {"ball1.0", "ball2.0", "ball4.0", "twozone", "smooth2.0"} <= fams


def test_ball_family_uses_conjunctive_region():
    """On a two-block spec a reward-1 point must satisfy BOTH blocks."""
    spec = HitSpec(((0, 1, 1.0), (1, 2, 1.0)))
    rng = np.random.default_rng(0)
    coords = np.zeros((4, 2))
    r = [x for x in L2.make_rewards(coords, rng, 1, spec=spec)
         if x["family"] == "ball1.0"][0]
    pts = np.array([[0.0, 0.0], [0.5, 3.0]])
    got = r["fn"](pts + r["center"])
    assert got.tolist() == [1.0, 0.0]


def test_smooth_family_is_positive_and_decreasing():
    spec = HitSpec(((0, 2, 0.5),))
    rng = np.random.default_rng(0)
    r = [x for x in L2.make_rewards(np.zeros((4, 2)), rng, 1, spec=spec)
         if x["family"] == "smooth2.0"][0]
    near = r["fn"]((r["center"] + np.array([0.1, 0.0]))[None])[0]
    far = r["fn"]((r["center"] + np.array([2.0, 0.0]))[None])[0]
    assert near > far >= 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_l2_reward_families.py -v`
Expected: FAIL — `make_rewards() got an unexpected keyword argument 'spec'`

- [ ] **Step 3: Edit the script**

Delete `CUBE_OBS = slice(19, 22)` and `TRAIN_EPS = 0.4`. Rewrite `make_rewards`:

```python
def make_rewards(coords: np.ndarray, rng: np.random.Generator,
                 per_family: int, spec) -> list:
    """Held-out reward families over the measure coordinate.

    Radii are MULTIPLIERS on the trained per-block radii, so 'ball4.0' denotes
    the same relative region size on every environment (spec 14 §3.4)."""
    def ball(center, mult):
        return lambda c: spec.contains(c, center, mult=mult).astype(np.float64)

    def smooth(center, mult):
        s = mult * float(np.mean(spec.radii))
        return lambda c: np.exp(-0.5 * np.square(
            np.linalg.norm(np.asarray(c, np.float64) - center, axis=-1) / s))

    def twozone(a_, b_, mult):
        fa, fb = ball(a_, mult), ball(b_, mult)
        return lambda c: np.maximum(fa(c), fb(c))

    out = []
    centers = coords[rng.choice(len(coords), 5 * per_family, replace=False)]
    k = 0
    for mult in (1.0, 2.0, 4.0):
        for _ in range(per_family):
            c = centers[k % len(centers)]; k += 1
            out.append({"family": f"ball{mult}", "name": f"ball{mult}_{k}",
                        "center": c, "fn": ball(c, mult)})
    for _ in range(per_family):
        c = centers[k % len(centers)]; k += 1
        out.append({"family": "smooth2.0", "name": f"smooth_{k}",
                    "center": c, "fn": smooth(c, 2.0)})
    for _ in range(per_family):
        a_ = centers[k % len(centers)]; k += 1
        b_ = centers[k % len(centers)]; k += 1
        out.append({"family": "twozone", "name": f"twozone_{k}",
                    "center": a_, "fn": twozone(a_, b_, 1.0)})
    return out
```

In `main()`: add the same `geometry_for` / `resolve_dataset` preamble as Task 4;
replace `cube_all = obs_all[:, CUBE_OBS]` with `cube_all = spec.coords(obs_all)`;
`fm = FullMeasure(cube_all[ref_rows], eps=TRAIN_EPS)` with
`fm = FullMeasure(cube_all[ref_rows], spec=spec)`;
`rewards = make_rewards(cube_all, rng, a.per_family)` with
`rewards = make_rewards(cube_all, rng, a.per_family, spec=spec)`;
and `def cube_of(ob): return ob[CUBE_OBS]` with
`def cube_of(ob): return spec.coords(np.asarray(ob)[None])[0]`.
Add `"env": cfg.domain` and the `"hit_spec"` string to the output dict.

- [ ] **Step 4: Run the test**

Run: `.venv/bin/python -m pytest tests/test_l2_reward_families.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/rescuing_fb/torchfb_zeroshot_l2.py tests/test_l2_reward_families.py
git commit -m "feat(spec14): L2 reward families over the trained measure coordinate"
```

---

### Task 6: Cube-single regression — prove the port changes no archived number

**Files:**
- Create: `scripts/rescuing_fb/check_l1_regression.py`
- Read-only: `results/spec10_torchfb/L1fix_rzK50_s42.json`

**Interfaces:**
- Consumes: the ported scripts from Tasks 3–5.
- Produces: a pass/fail regression gate. **No later task may start until this passes.**

- [ ] **Step 1: Re-score the archived cube-single checkpoint**

Run:
```bash
cd /home/adityamohan/git/Factored-FB
CUDA_VISIBLE_DEVICES=5 MUJOCO_GL=egl .venv/bin/python \
  scripts/rescuing_fb/torchfb_full_measure.py \
  --run ~/exp_spec10/torchfb/p2_rzK50_s42 \
  --starts 32 --mc 3 --horizon 200 --boxes 12 --rho-samples 200000 \
  --seed 0 --out /tmp/L1fix_rzK50_s42_REPLAY.json
```
Expected: completes, and `probe_centers` in the JSON is `"random"` (cube-single keeps the archived policy).

- [ ] **Step 2: Write the comparison script**

```python
# scripts/rescuing_fb/check_l1_regression.py
"""Spec 14 §5 — the port must not move any archived cube-single number.

Archived JSONs key their summary by absolute half-width ('0.4'); the ported
code keys by multiplier ('1.0'). On cube-single eps=0.4 the map is exact:
0.4->1.0, 0.8->2.0, 1.6->4.0.
"""
import json
import sys

KEYMAP = {"0.4": "1.0", "0.8": "2.0", "1.6": "4.0"}
FIELDS = ["mc_mean", "model_exact_mean", "ratio_exact", "spearman_exact",
          "model_head_mean", "ratio_head", "spearman_head"]
TOL = 0.02          # MC rollouts are stochastic in the env, not the code


def main(archived: str, replay: str) -> int:
    A = json.load(open(archived))["summary"]
    B = json.load(open(replay))["summary"]
    bad = []
    for old_key, new_key in KEYMAP.items():
        a, b = A[old_key], B[new_key]
        for f in FIELDS:
            if f not in a:
                continue
            rel = abs(a[f] - b[f]) / max(abs(a[f]), 1e-9)
            print(f"hw={old_key:>4} {f:<18} archived={a[f]:>10.4f} "
                  f"replay={b[f]:>10.4f} rel={rel:.3%}")
            if rel > TOL:
                bad.append((old_key, f, a[f], b[f]))
    if bad:
        print(f"\nFAIL: {len(bad)} fields moved by more than {TOL:.0%}")
        for k, f, x, y in bad:
            print(f"  hw={k} {f}: {x} -> {y}")
        return 1
    print("\nPASS: the port reproduces the archived cube-single numbers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
```

- [ ] **Step 3: Run the regression gate**

Run:
```bash
.venv/bin/python scripts/rescuing_fb/check_l1_regression.py \
  results/spec10_torchfb/L1fix_rzK50_s42.json /tmp/L1fix_rzK50_s42_REPLAY.json
```
Expected: `PASS`.

If it fails on `mc_mean` only, the cause is MC sampling, not the port — confirm by re-running Step 1 with `--seed 1` and checking the archived-vs-replay spread is the same size as the seed-to-seed spread. If any `model_*` field moves, the port is wrong: the set-membership path has changed and must be fixed before proceeding.

- [ ] **Step 4: Commit**

```bash
git add scripts/rescuing_fb/check_l1_regression.py
git commit -m "test(spec14): cube-single regression gate for the geometry port"
```

---

### Task 7: Runner and the s42 campaign

**Files:**
- Create: `scripts/rescuing_fb/run_multienv_measure.sh`

**Interfaces:**
- Consumes: everything above.
- Produces: `results/spec10_torchfb/{negmass,L1fix,L2_B,L2_A}_{scn,cdbl}{p1,p2}_s42.json`

- [ ] **Step 1: Write the runner**

```bash
#!/bin/bash
# Spec 14 §3.5 — measure-axis chain over the torch multi-env checkpoints.
#
# Chain per checkpoint: negmass -> L1fix -> L2-B -> L2-A. Flags match the
# cube-single runs so the rows are directly comparable to the archived JSONs.
# G0 (degenerate_frac < 0.5) is checked after L1; a failing cell is re-run at
# wider multipliers rather than reported.
set -u
cd /home/adityamohan/git/Factored-FB
PY=.venv/bin/python
OUT=/home/adityamohan/exp_spec10/torchfb
RES=results/spec10_torchfb
LOG=/dev/shm/spec14_logs; mkdir -p "$LOG" "$RES"

SEED=${SEED:-42}
# tag|run dir|gpu
JOBS=(
  "scnp1|$OUT/p1_scn_s$SEED|5"
  "scnp2|$OUT/p2_scn_s$SEED|6"
  "cdblp1|$OUT/p1_cdbl_s$SEED|7"
  "cdblp2|$OUT/p2_cdbl_s$SEED|5"
)

run_one() { # tag run gpu
  local TAG=$1 RUN=$2 G=$3
  if [ ! -f "$RUN/checkpoints/final.pt" ]; then
    echo "[s14] $TAG: no ckpt at $RUN - skip"; return 1
  fi
  local T=${TAG}_s$SEED

  CUDA_VISIBLE_DEVICES=$G MUJOCO_GL=egl $PY scripts/rescuing_fb/negmass_diag.py \
    --run $RUN --starts 32 --goals 20000 --rho-samples 100000 --seed 0 \
    --out $RES/negmass_$T.json > $LOG/negmass_$T.log 2>&1
  echo "[s14] negmass $T exit=$? $(date +%H:%M:%S)"

  CUDA_VISIBLE_DEVICES=$G MUJOCO_GL=egl $PY scripts/rescuing_fb/torchfb_full_measure.py \
    --run $RUN --starts 32 --mc 3 --horizon 200 --boxes 12 --rho-samples 200000 \
    --seed 0 --out $RES/L1fix_$T.json > $LOG/L1fix_$T.log 2>&1
  echo "[s14] L1 $T exit=$? $(date +%H:%M:%S)"
  $PY - "$RES/L1fix_$T.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1])).get("summary", {})
for k, v in s.items():
    d = v.get("degenerate_frac")
    flag = "G0-FAIL" if (d is not None and d >= 0.5) else "ok"
    print(f"[G0] mult={k} degenerate_frac={d} {flag}")
EOF

  CUDA_VISIBLE_DEVICES=$G MUJOCO_GL=egl $PY scripts/rescuing_fb/torchfb_zeroshot_l2.py \
    --run $RUN --stage B --seed 0 --out $RES/L2_B_$T.json > $LOG/L2_B_$T.log 2>&1
  echo "[s14] L2-B $T exit=$? $(date +%H:%M:%S)"

  CUDA_VISIBLE_DEVICES=$G MUJOCO_GL=egl $PY scripts/rescuing_fb/torchfb_zeroshot_l2.py \
    --run $RUN --stage A --seed 0 --out $RES/L2_A_$T.json > $LOG/L2_A_$T.log 2>&1
  echo "[s14] L2-A $T exit=$? $(date +%H:%M:%S)"
}

for j in "${JOBS[@]}"; do
  IFS='|' read -r tag run gpu <<<"$j"
  echo "[s14] $(date +%H:%M:%S) start $tag on GPU$gpu"
  run_one "$tag" "$run" "$gpu" &
  sleep 15
done
wait
echo "[s14] MULTI-ENV MEASURE CHAIN DONE (seed $SEED) $(date +%H:%M:%S)"
```

- [ ] **Step 2: Dry-run one cell before committing the box**

Run:
```bash
chmod +x scripts/rescuing_fb/run_multienv_measure.sh
CUDA_VISIBLE_DEVICES=5 MUJOCO_GL=egl .venv/bin/python \
  scripts/rescuing_fb/torchfb_full_measure.py \
  --run ~/exp_spec10/torchfb/p2_cdbl_s42 --starts 4 --mc 1 --horizon 50 \
  --boxes 4 --rho-samples 20000 --seed 0 --out /tmp/L1_cdbl_SMOKE.json
```
Expected: completes; `probe_centers == "hindsight"`; at least one multiplier has `degenerate_frac < 1.0`. **If every multiplier is 1.0, stop** — hindsight centers are not rescuing cube-double's 0.046% region and the multipliers must be widened before spending the full campaign.

- [ ] **Step 3: Launch the campaign in tmux**

Run:
```bash
tmux new -d -s s14eval 'scripts/rescuing_fb/run_multienv_measure.sh 2>&1 | tee -a /dev/shm/spec14_logs/dispatch.log'
```
Expected: four cells start on GPUs 5/6/7.

- [ ] **Step 4: Commit**

```bash
git add scripts/rescuing_fb/run_multienv_measure.sh
git commit -m "feat(spec14): multi-env measure-eval runner"
```

---

### Task 8: Collect, gate, and write up

**Files:**
- Modify: `PAPER/FB-Manipulation-ICLR/MULTIENV-RESULTS.md` (§5), `PAPER/rescuing-fb/PHASE2-LEDGER.md` (new §9.17), `PAPER/rescuing-fb/DECISIONS.md`, `scripts/results_map.py`

**Interfaces:**
- Consumes: the JSONs from Task 7 (and the s43 repeat once tmux `menv43` finishes).
- Produces: the measure-axis table and a written verdict against the pre-registered gates.

- [ ] **Step 1: Check G0 first, for every cell**

Run:
```bash
.venv/bin/python - <<'EOF'
import glob, json
for f in sorted(glob.glob("results/spec10_torchfb/L1fix_*p[12]_s4*.json")):
    d = json.load(open(f))
    print(f.split("/")[-1], d.get("env"), d.get("probe_centers"),
          {k: v.get("degenerate_frac") for k, v in d.get("summary", {}).items()})
EOF
```
Any cell with `degenerate_frac >= 0.5` at every multiplier is **uninformative** — re-run it with wider multipliers (edit `MULTS` in `env_geometry.py`, re-run that cell only) before reading any other gate.

- [ ] **Step 2: Tabulate against the pre-registered gates**

Build a table with one row per (env, phase, seed) and columns: `neg_frac`, `neg_mass_share`, `ratio_head` and `spearman_head` at each multiplier, `degenerate_frac`, and L2 `gap_frac` per family. Compare to the cube-single **no-penalty** control (`negmass_rzK50_s42.json`, `L1fix_rzK50_s42.json`, `L2_B_rzK50_s42.json`) — not the mcA/mcB arms, because multi-env phase 2 ran with validity penalties OFF (spec 14 §6).

- [ ] **Step 3: Write the verdict into the ledger**

Add `## 9.17` to `PAPER/rescuing-fb/PHASE2-LEDGER.md` stating, for each gate G0/G1/G2/G4, pass or fail with the numbers, and explicitly whether the NULL fired (both envs passing G2/G4 at cube-single levels ⇒ the measure axis transfers and §9.16's regression is actor-side only). State the s42/s43 spread for every quoted number; if s43 is not yet finished, say so and mark the section provisional.

- [ ] **Step 4: Note the protocol difference wherever the envs share a table**

Any table containing both cube-single and cube-double/scene L1 rows must carry the footnote from spec 14 §3.3: cube-single uses random probe centers, the conjunctive envs use hindsight centers, so the L1 protocols are not identical.

- [ ] **Step 5: Refresh the results CSVs (standing repo rule)**

Run:
```bash
.venv/bin/python scripts/compile_results_csvs.py --refresh
```
after adding any new run groups to `scripts/results_map.py`.

- [ ] **Step 6: Commit the code-side changes only**

```bash
git add scripts/results_map.py
git commit -m "chore(spec14): register multi-env measure runs in results map"
```
`PAPER/`, `results/` and `analysis/` are gitignored — do not attempt to add them.

---

## Self-Review

**Spec coverage.** §3.1 registry → Task 1. §3.2 generalized `FullMeasure` → Task 2. §3.3 probe centers and `degenerate_frac` (all three scripts) → Tasks 3, 4 (L1 centers), 5 (L2 reward centers, which draw from the same coordinate). §3.4 the three ports plus dataset-path and reward-radius fixes → Tasks 3–5. §3.5 runner and output naming → Task 7. §3.6 phase 1 scored with a footnote → Task 7 (`p1` cells) and Task 8 Step 4. §4 seeds → already launched (tmux `menv43`); consumed in Task 8. §5 tests → Tasks 1–5, with the cube-single regression as its own gate in Task 6. §6 gates → Task 7 Step 2 (G0 inline) and Task 8 Steps 1–3. §7 JAX port → explicitly deferred, no task. §8 deliverable → Task 8.

**Placeholder scan.** No TBD/TODO. Every code step carries runnable content; every test step names the exact command and expected outcome.

**Type consistency.** `HitSpec.contains/box_contains(coords, center, mult)` is used with that signature in Tasks 2–5. `probe_rows(policy, bounds, start_row, n, n_rows, rng)` matches its test. `make_rewards(coords, rng, per_family, spec)` matches its test and its call site. `FullMeasure(coords, spec=...)` is used consistently after Task 2. `MULTS` is defined once in `env_geometry.py` and imported by Task 4 and Task 7's G0 check.

**One risk left deliberately open:** Task 7 Step 2 is a hard stop. If hindsight centers still leave cube-double fully degenerate, the multipliers must be widened before the campaign runs — that decision needs the smoke number in hand, so the plan asks for it rather than guessing now.
