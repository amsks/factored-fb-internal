"""Tests for scripts/rescuing_fb/smb_tasks.py + the MC/dataset-side pieces of
smb_common.py (spec 11 item 5.1). Structure + math only -- no env, no
checkpoint, no jax: synthetic trajectory arrays drive the region-selection
logic; determinism of the frozen artifact is checked by rebuilding. Dataset-
dependent checks live behind the `slow` marker (skipped when the ogbench npz
is absent)."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts" / "rescuing_fb"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # numpy-only at module level (jax is lazy)
    return mod


@pytest.fixture(scope="module")
def smc():
    return _load("smb_common")


@pytest.fixture(scope="module")
def smt():
    return _load("smb_tasks")


# ============================ synthetic play data ================================
# A miniature "play dataset": episodes where the cube dwells in a sticky bowl at
# STABLE (on table, z=0), is carried through a TRANSIENT waypoint (z=0.8) for
# exactly one step per crossing, and rests at a second table spot. Enough
# spatial spread for FPS/candidate machinery to operate.

STABLE = np.array([0.0, 0.0, 0.0])
TRANSIENT = np.array([1.0, 1.0, 0.8])
REST2 = np.array([2.0, 2.0, 0.0])


def synth_play(n_episodes=40, seed=0):
    """Every on-table visit is a DWELL (as in real play data: cube on table =
    settled), the elevated waypoint is a 1-step pass-through; the long bowl
    dwell at STABLE dominates the occupancy ranking."""
    rng = np.random.default_rng(seed)
    cube, term = [], []
    for _ in range(n_episodes):
        ep = []
        # long dwell at STABLE (the sticky bowl -> high post-arrival occupancy)
        ep += [STABLE + rng.normal(0, 0.05, 3) * [1, 1, 0] for _ in range(25)]
        # transport arc: ONE step at the elevated waypoint
        ep.append(TRANSIENT + rng.normal(0, 0.05, 3) * [1, 1, 0])
        # rest at the second spot
        ep += [REST2 + rng.normal(0, 0.05, 3) * [1, 1, 0] for _ in range(10)]
        # short dwells at random on-table positions (spatial diversity)
        for _ in range(4):
            p = rng.uniform([-2, -2, 0], [4, 4, 0])
            ep += [p + rng.normal(0, 0.03, 3) * [1, 1, 0] for _ in range(4)]
        cube += ep
        t = np.zeros(len(ep))
        t[-1] = 1.0
        term.append(t)
    return np.asarray(cube, np.float64), np.concatenate(term)


@pytest.fixture(scope="module")
def play(smc):
    cube, term = synth_play()
    spans = smc.split_episodes(term)
    return cube, spans


# ============================ table z + geometry =================================

def test_table_z_mode(smt):
    z = np.concatenate([np.full(900, 0.021), np.full(50, 0.82), np.full(50, 0.4)])
    tz = smt.table_z_mode(z)
    assert abs(tz - 0.021) < 0.02


def test_point_seg_dist(smt):
    a, b = np.zeros(3), np.array([2.0, 0.0, 0.0])
    p = np.array([[1.0, 1.0, 0.0],      # perpendicular off midpoint -> 1
                  [3.0, 0.0, 0.0],      # beyond b -> 1 (clamped)
                  [-1.0, 0.0, 0.0],     # before a -> 1 (clamped)
                  [1.0, 0.0, 0.0]])     # on segment -> 0
    np.testing.assert_allclose(smt.point_seg_dist(p, a, b), [1, 1, 1, 0], atol=1e-12)
    # degenerate segment = point distance
    np.testing.assert_allclose(smt.point_seg_dist(p, a, a),
                               np.linalg.norm(p, axis=-1))


def test_support_counts(smt):
    cube = np.array([[0, 0, 0], [0.1, 0, 0], [1, 1, 1], [0.39, 0, 0]], float)
    centers = np.array([[0, 0, 0], [1, 1, 1]], float)
    np.testing.assert_array_equal(smt.support_counts(cube, centers, eps=0.4), [3, 1])


def test_fps_select_deterministic_and_spread(smt):
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    pts = np.random.default_rng(1).uniform(0, 1, size=(100, 3))
    s1 = smt.fps_select(pts, 10, rng1)
    s2 = smt.fps_select(pts, 10, rng2)
    np.testing.assert_array_equal(s1, s2)               # deterministic
    assert len(set(s1.tolist())) == 10                  # no duplicates
    assert smt.fps_select(pts, 0, np.random.default_rng(0)).size == 0


# ============================ MC region stats ====================================

def test_mc_region_stats_basic(smc):
    """Hand-built track: ep0 enters the ball at t=3 and stays 3 steps (one
    sojourn = one entry, value 1+g+g^2); ep1 passes through for 1 step at t=2
    (value 1). Boundaries respected; first arrivals 3 and 2."""
    g = 0.99
    far = np.array([9.0, 9.0, 9.0])
    ctr = np.zeros(3)
    ep0 = [far, far, far, ctr, ctr, ctr, far, far, far, far]
    ep1 = [far, far, ctr, far, far, far, far, far]
    cube = np.asarray(ep0 + ep1, float)
    spans = [(0, 10), (10, 18)]
    st = smc.mc_region_stats(cube, spans, ctr[None], gamma=g, eps=0.4)
    assert st["n_entries"][0] == 2
    np.testing.assert_allclose(st["occ_mean"][0], ((1 + g + g * g) + 1) / 2)
    np.testing.assert_allclose(st["first_med"][0], 2.5)   # median(3, 2)
    np.testing.assert_allclose(st["hit_frac"][0], 1.0)
    assert st["support"][0] == 4


def test_mc_region_stats_boundary_and_start(smc):
    """In-ball steps at an episode END and the next episode's START are two
    separate entries (no cross-boundary sojourn), and an episode that starts in
    the ball counts as an entry at t=0."""
    g = 0.5
    far = np.array([9.0, 9.0, 9.0])
    ctr = np.zeros(3)
    cube = np.asarray([far, ctr, ctr, ctr, far], float)   # ep0 ends in-ball x1? no:
    cube = np.asarray([far, far, ctr,          # ep0: entry at t=2, runs to end
                       ctr, far, far], float)             # ep1: STARTS in-ball
    spans = [(0, 3), (3, 6)]
    st = smc.mc_region_stats(cube, spans, ctr[None], gamma=g, eps=0.4)
    assert st["n_entries"][0] == 2                        # NOT one merged sojourn
    np.testing.assert_allclose(st["occ_mean"][0], 1.0)    # both entries: single step
    np.testing.assert_allclose(st["first_med"][0], 1.0)   # median(2, 0)


def test_mc_region_stats_never_entered(smc):
    cube = np.zeros((10, 3))
    st = smc.mc_region_stats(cube, [(0, 10)], np.array([[5.0, 5.0, 5.0]]), eps=0.4)
    assert np.isnan(st["occ_mean"][0]) and st["n_entries"][0] == 0
    assert st["hit_frac"][0] == 0.0 and st["support"][0] == 0


# ============================ T3 pair selection ==================================

def test_select_t3_pairs_synthetic(smt, play):
    cube, spans = play
    rng = np.random.default_rng(0)
    tz = smt.table_z_mode(cube[:, 2])
    pairs, meta = smt.select_t3_pairs(
        cube, spans, tz, rng, n_pairs=2, n_cand=40, min_support=5,
        min_entries=3, pool_n=1000)
    assert len(pairs) >= 1
    for p in pairs:
        s, t = p["stable"], p["transient"]
        # stable on table, transient pinned at table_z + 0.8
        assert abs(s["center"][2] - tz) <= 0.1
        np.testing.assert_allclose(t["center"][2], tz + smt.ELEV_DZ)
        # the discriminator structure: stable dwells, transient passes through
        assert s["R_mc"] > t["R_mc"]
        assert p["R_ratio"] > 1.0
        assert p["w_S"] == 1.0
        assert 2.0 <= p["w_T"] <= 8.0
        assert p["validation"] in ("ok", "failed")
        assert p["correct_region"] in ("stable", "transient")
        # the synthetic bowl dwell (25 steps) vs 1-step pass-through: gap >= 5
        assert p["R_ratio"] >= 5.0 and p["validation"] == "ok"


# ============================ T4 / T5 / battery ==================================

def test_select_union3(smt, play):
    cube, spans = play
    rng = np.random.default_rng(0)
    tz = smt.table_z_mode(cube[:, 2])
    tasks = smt.select_union3(cube, tz, rng, n_tasks=2, min_support=3, pool_n=500)
    assert len(tasks) == 2
    for t in tasks:
        assert len(t["centers"]) == 3
        assert t["pairwise_dist_min"] >= 3 * smt.EPS_OBS
        for c in t["centers"]:
            assert abs(c[2] - tz) <= 0.1


def test_select_corridor(smt):
    """Dense blob near the segment midpoint must win the density argmax; the
    picked center must lie within halfwidth of the segment and outside the
    goal/start exclusion."""
    rng = np.random.default_rng(0)
    start = np.zeros(3)
    goal = np.array([4.0, 0.0, 0.0])
    blob = np.array([2.0, 0.1, 0.0])
    cube = np.concatenate([
        blob + np.random.default_rng(1).normal(0, 0.05, (200, 3)),
        np.random.default_rng(2).uniform([-1, -3, 0], [5, 3, 0], (200, 3)),
    ])
    rec = smt.select_corridor(cube, start, goal, rng, samp_n=len(cube))
    assert np.linalg.norm(np.asarray(rec["avoid_center"]) - blob) < 0.3
    assert rec["seg_dist"] <= rec["halfwidth_used"] + 1e-9
    assert np.linalg.norm(np.asarray(rec["avoid_center"]) - goal) > \
        rec["avoid_radius"] + smt.EPS_OBS


def test_select_corridor_excludes_goal_neighbourhood(smt):
    """A blob INSIDE the goal exclusion zone must not be picked even if densest."""
    rng = np.random.default_rng(0)
    start = np.zeros(3)
    goal = np.array([4.0, 0.0, 0.0])
    near_goal = goal - [0.5, 0, 0]          # within avoid_r + eps = 0.9 of goal
    cube = np.concatenate([
        near_goal + np.random.default_rng(1).normal(0, 0.03, (500, 3)),
        np.array([2.0, 0.05, 0.0]) + np.random.default_rng(2).normal(0, 0.05, (50, 3)),
    ])
    rec = smt.select_corridor(cube, start, goal, rng, samp_n=len(cube))
    assert np.linalg.norm(np.asarray(rec["avoid_center"]) - goal) > 0.9


def test_build_battery(smt, play):
    cube, spans = play
    # add a supported elevated cluster so the >=8-elevated requirement is testable
    rng_e = np.random.default_rng(5)
    elev = TRANSIENT + rng_e.normal(0, 0.15, (400, 3))
    cube2 = np.concatenate([cube, elev])
    tz = smt.table_z_mode(cube2[:, 2])
    rng = np.random.default_rng(0)
    bat = smt.build_battery(cube2, tz, rng, n_balls=16, n_elev=4, n_boxes=6,
                            n_unions=3, min_support=3, pool_n=800, samp_n=len(cube2))
    assert len(bat["balls"]) == 16
    assert bat["n_elevated"] >= 4
    assert len(bat["boxes"]) == 6
    assert len(bat["unions"]) == 3
    for b in bat["boxes"]:
        assert 0.2 <= min(b["half"]) and max(b["half"]) <= 0.6
        assert len(b["rep_rows"]) >= 1
        for r in b["rep_rows"]:            # rep rows really are inside the box
            assert (np.abs(cube2[r] - np.asarray(b["center"]))
                    <= np.asarray(b["half"]) + 1e-9).all()
    for u in bat["unions"]:
        assert u["sep"] >= 3 * smt.EPS_OBS
        for k in (0, 1):
            for r in u["rep_rows"][k]:
                assert np.linalg.norm(cube2[r] - np.asarray(u["centers"][k])) \
                    <= u["eps"] + 1e-9 or r == u["rows"][k]  # fallback = center row


# ============================ artifact determinism ===============================

def _fake_goals():
    return {i: np.array([0.5 * i, 0.5 * i, 0.0]) for i in range(1, 6)}


_SMALL = dict(
    t3_kwargs=dict(n_pairs=2, n_cand=40, min_support=4, min_entries=3),
    t4_kwargs=dict(min_support=3, pool_n=500),
    battery_kwargs=dict(n_balls=16, n_elev=2, n_boxes=6, n_unions=3,
                        min_support=3, pool_n=800, samp_n=5000),
)


def test_build_artifact_deterministic(smt, play):
    cube, spans = play
    a1 = smt.build_artifact(cube, spans, _fake_goals(), seed=0, **_SMALL)
    a2 = smt.build_artifact(cube, spans, _fake_goals(), seed=0, **_SMALL)
    assert smt.artifact_text(a1) == smt.artifact_text(a2)   # --check contract
    a3 = smt.build_artifact(cube, spans, _fake_goals(), seed=1, **_SMALL)
    assert smt.artifact_text(a1) != smt.artifact_text(a3)   # seed actually drives it
    # round-trips through json (to_native leaves no numpy scalars behind)
    rt = json.loads(smt.artifact_text(a1))
    assert rt["version"] == smt.TASK_JSON_VERSION
    assert len(rt["t3_occhoice"]["pairs"]) >= 1
    assert len(rt["t4_union3"]["tasks"]) == 3
    assert len(rt["t5_corridor"]["tasks"]) == 5
    assert len(rt["wingb_battery"]["balls"]) == 16          # _SMALL sizes
    assert len(rt["wingb_battery"]["boxes"]) == 6
    assert len(rt["wingb_battery"]["unions"]) == 3


def test_to_native_types(smc):
    out = smc.to_native({"a": np.float32(1.5), "b": np.int64(2),
                         "c": np.bool_(True), "d": np.arange(3),
                         "e": [np.float64(0.25)]})
    assert out == {"a": 1.5, "b": 2, "c": True, "d": [0, 1, 2], "e": [0.25]}
    assert json.dumps(out)                                  # serializable


# ============================ CLI ================================================

def test_smb_tasks_help_runs():
    proc = subprocess.run([sys.executable, str(SCRIPTS / "smb_tasks.py"), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    for flag in ("--dataset-path", "--seed", "--out", "--check"):
        assert flag in proc.stdout


# ============================ slow (real dataset) ================================

_DS = Path.home() / ".ogbench" / "data" / "cube-single-play-v0.npz"


@pytest.mark.slow
@pytest.mark.skipif(not _DS.exists(), reason="ogbench dataset not present")
def test_real_dataset_table_and_elevation(smt):
    import numpy as np
    with np.load(_DS) as data:
        obs = np.asarray(data["observations"][:200_000], np.float32)
    cube = obs[:, 19:22].astype(np.float64)
    tz = smt.table_z_mode(cube[:, 2])
    # the cube rests on the table: the mode must be low and heavily populated
    assert tz < 0.5
    frac_on_table = float((np.abs(cube[:, 2] - tz) <= 0.1).mean())
    assert frac_on_table > 0.5
    # elevated rows (transport) exist near table_z + 0.8
    n_elev = int((np.abs(cube[:, 2] - (tz + 0.8)) <= 0.25).sum())
    assert n_elev > 100
