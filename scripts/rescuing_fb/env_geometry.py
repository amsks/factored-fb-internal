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
