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
