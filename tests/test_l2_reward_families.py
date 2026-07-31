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
    # centers must be spread out or no pair clears twozone's separation rule
    coords = rng.uniform(-5, 5, (10, 2))
    fams = {r["family"] for r in L2.make_rewards(
        coords, rng, per_family=1, spec=spec)}
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
