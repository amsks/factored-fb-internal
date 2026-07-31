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
