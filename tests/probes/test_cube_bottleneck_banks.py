"""Bank selection and restore-gate logic (no simulator; see test_cube_bottleneck_restore.py)."""

import numpy as np
import pytest

from scripts.probes.cube_bottleneck import banks as BK


def _raw(n_ep=3, ep_len=10, seed=0):
    """A synthetic dataset in the layout `load_raw` produces."""
    rng = np.random.default_rng(seed)
    n = n_ep * ep_len
    obs = rng.uniform(-1, 1, size=(n, 28)).astype(np.float32)
    qpos = rng.uniform(-1, 1, size=(n, 21)).astype(np.float32)
    episode = np.repeat(np.arange(n_ep), ep_len).astype(np.int32)
    step = np.tile(np.arange(ep_len), n_ep).astype(np.int32)
    return dict(
        observations=obs,
        actions=rng.uniform(-1, 1, (n, 5)).astype(np.float32),
        qpos=qpos,
        qvel=rng.uniform(-1, 1, (n, 20)).astype(np.float32),
        episode=episode,
        step=step,
    )


def test_persists_requires_k_consecutive_within_one_episode():
    episode = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    mask = np.array([1, 1, 1, 0, 1, 1, 1, 1], bool)
    out = BK._persists(mask, episode, 3)
    # ep0: only index 0 starts a run of 3. index 1 would need 1,2,3 and 3 is False.
    assert out.tolist() == [True, False, False, False, True, True, False, False]


def test_persists_does_not_bridge_an_episode_boundary():
    """A persistence run does not extend into the next episode."""
    episode = np.array([0, 0, 1, 1])
    mask = np.array([1, 1, 1, 1], bool)
    assert BK._persists(mask, episode, 3).tolist() == [False, False, False, False]


def test_phase_masks_are_mutually_exclusive():
    raw = _raw()
    # Force a spread of contact / lift / distance so every phase is populated.
    raw["observations"][:, BK.CONTACT_IDX] = np.tile(
        [0.9, 0.1], len(raw["observations"]) // 2
    )
    raw["qpos"][:, BK.CUBE_QPOS_Z] = np.tile(
        [0.10, 0.02], len(raw["observations"]) // 2
    )
    raw["observations"][:, BK.EFF_OBS] = 0.0
    raw["observations"][:, BK.CUBE_OBS] = 0.0
    m = BK.phase_masks(raw)
    assert not (m["P2"] & m["P3"]).any(), "a state cannot be both at-contact and held"
    assert not (m["P1"] & m["P2"]).any()
    assert not (m["P1"] & m["P3"]).any()


def test_p1_requires_no_contact_and_p2_p3_require_contact():
    raw = _raw()
    m = BK.phase_masks(raw)
    contact = raw["observations"][:, BK.CONTACT_IDX] > BK.CONTACT_THRESH
    assert not contact[m["P1"]].any()
    assert contact[m["P2"]].all()
    assert contact[m["P3"]].all()


def test_p3_is_lifted_and_p2_is_not():
    raw = _raw()
    m = BK.phase_masks(raw)
    z = raw["qpos"][:, BK.CUBE_QPOS_Z]
    assert (z[m["P3"]] > BK.LIFT_THRESH).all()
    assert (z[m["P2"]] <= BK.LIFT_THRESH).all()


def test_sample_bank_is_deterministic_for_a_fixed_seed():
    raw = _raw()
    mask = np.ones(len(raw["observations"]), bool)
    a = BK.sample_bank(raw, mask, 5, np.random.default_rng(BK.RNG_SEED))
    b = BK.sample_bank(raw, mask, 5, np.random.default_rng(BK.RNG_SEED))
    assert a["row"].tolist() == b["row"].tolist()
    np.testing.assert_array_equal(a["qpos"], b["qpos"])


def test_sample_bank_raises_rather_than_shrinking_a_short_bank():
    """sample_bank raises on a short bank instead of shrinking it."""
    raw = _raw()
    mask = np.zeros(len(raw["observations"]), bool)
    mask[:3] = True
    with pytest.raises(ValueError, match="only 3 candidate rows"):
        BK.sample_bank(raw, mask, 10, np.random.default_rng(0))


def test_lift_threshold_matches_the_delta_lift_the_phase_code_uses():
    """LIFT_THRESH equals table_z + delta_lift."""
    from scripts.probes.cube_bottleneck.phases import PhaseThresholds

    assert BK.LIFT_THRESH == pytest.approx(BK.TABLE_Z + PhaseThresholds().delta_lift)
