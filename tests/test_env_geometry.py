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
