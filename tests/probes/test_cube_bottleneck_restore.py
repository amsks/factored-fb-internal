"""Simulator-backed restore tests: they pin the facts the probe's validity rests on.

These are the regression form of the two determinations the spec demanded be made
empirically rather than guessed (spec section 2.2): which gripper command actually closes,
and whether a restored state is the state that was recorded. They need MuJoCo and the
OGBench play dataset, and skip cleanly without them.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest

from scripts.probes.cube_bottleneck import banks as BK
from scripts.probes.cube_bottleneck import restore as R

ENV_NAME = "cube-single-play-v0"
SCAN_ROWS = 50_000     # enough to find held rows without loading 1M


@pytest.fixture(scope="module")
def held_rows():
    try:
        path = BK.dataset_path(ENV_NAME)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    d = np.load(path, mmap_mode="r")
    obs = np.asarray(d["observations"][:SCAN_ROWS])
    qpos = np.asarray(d["qpos"][:SCAN_ROWS])
    qvel = np.asarray(d["qvel"][:SCAN_ROWS])
    # The SAME criterion the real bank uses, persistence included. Without the k_steps
    # filter this fixture picks up transients and marginal lifts (a cube at lift 0.0301
    # settles to 0.0288 and is then correctly not `held`), which would make these tests
    # assert something the probe never relies on.
    held = ((obs[:, BK.CONTACT_IDX] > BK.CONTACT_THRESH)
            & (qpos[:, BK.CUBE_QPOS_Z] > BK.LIFT_THRESH))
    episode = (np.arange(len(obs)) // 1001).astype(np.int32)
    m = np.flatnonzero(BK._persists(held, episode, BK.K_STEPS))
    if len(m) < 4:
        pytest.skip("no persistently-held rows in the scanned prefix")
    # Sample the way `banks.sample_bank` does rather than taking a deterministic slice: an
    # evenly-spaced pick lands on the first candidate row, which sits exactly at the lift
    # boundary (0.0301 against delta_lift 0.030) and settles just below it. G2 rejecting
    # such a row is CORRECT -- it is a marginal grasp, not a restore failure -- but it is
    # not the behaviour these tests are about, and the real 64-row bank (lift p5 = 0.084)
    # does not contain one.
    sel = np.random.default_rng(BK.RNG_SEED).choice(m, size=4, replace=False)
    return [dict(obs=obs[i], qpos=qpos[i], qvel=qvel[i]) for i in sel]


@pytest.fixture(scope="module")
def env():
    pytest.importorskip("mujoco")
    pytest.importorskip("ogbench")
    e = R.make_env(ENV_NAME, seed=0)
    R.ensure_cube_env(e)
    return e


def test_set_state_reproduces_every_obs_channel_except_contact(env, held_rows):
    """The 28-dim state observation is a pure function of (qpos, qvel) -- except obs[18],
    whose contact force also depends on `data.ctrl`, which no dataset records."""
    for r in held_rows:
        env.reset(options=dict(task_id=1, render_goal=False))
        ob = R.set_state(env, r["qpos"], r["qvel"])
        err = np.abs(ob - r["obs"])
        err_no_contact = np.delete(err, R.CONTACT_IDX)
        assert err_no_contact.max() < 0.01, (
            f"non-contact channels drifted by {err_no_contact.max():.4f}")


def test_settling_rebuilds_the_contact_channel(env, held_rows):
    """N_SETTLE steps of a commanded close restore contact to the recorded magnitude.

    Without it the restored state claims a lifted cube with no grip force on it -- a
    combination the dataset essentially never contains.
    """
    for r in held_rows:
        env.reset(options=dict(task_id=1, render_goal=False))
        R.set_state(env, r["qpos"], r["qvel"])
        before = R.observation(env)[R.CONTACT_IDX]
        R.settle(env)
        after = R.observation(env)[R.CONTACT_IDX]
        assert after > 0.5, f"contact only reached {after:.3f} after settling"
        # Monotonicity only where there was something to rebuild: a few restored rows
        # already read full contact from `mj_rnePostConstraint` alone.
        if before < 0.5:
            assert after > before


def test_positive_gripper_action_closes_and_negative_drops(env, held_rows):
    """THE gripper-convention fact, measured rather than assumed.

    `ctrl = 255 * target_gripper_opening` over a driver joint that rises as the Robotiq
    CLOSES, so +1 is the closing command. `[torch] counterfactual_value_probe.py:165`
    commands 0.0 to "establish the grasp", which is the open extreme.
    """
    held_pos, held_neg = [], []
    for r in held_rows:
        for a_grip, sink in ((+1.0, held_pos), (-1.0, held_neg)):
            R.restore(env, 1, r["qpos"], r["qvel"], a_grip=a_grip)
            sink.append(R.g2_check(env, r["qpos"]).held)
    assert all(held_pos), "a_grip=+1 must keep the cube held"
    assert not any(held_neg), "a_grip=-1 must drop the cube"


def test_g2_rejects_a_latent_drop(env, held_rows):
    """Raw `ctrl = 0` leaves the cube momentarily in place with the fingers slack.

    It passes the purely geometric `held` test, so G2's contact conjunct is what keeps such
    a row out of the bank.
    """
    import mujoco

    slack_passes, slack_held = [], []
    for r in held_rows:
        env.reset(options=dict(task_id=1, render_goal=False))
        R.set_state(env, r["qpos"], r["qvel"])
        u = env.unwrapped
        u._data.ctrl[u._gripper_actuator_ids] = 0.0
        for _ in range(R.N_SETTLE):
            mujoco.mj_step(u._model, u._data)
        R.forward(env)
        g = R.g2_check(env, r["qpos"])
        slack_passes.append(g.passes)
        slack_held.append(g.held)
    assert any(slack_held), "the geometric test alone does NOT catch this -- that is the point"
    assert not any(slack_passes), "G2 must reject a slack-gripper restore"


def test_g2_accepts_the_shipped_recipe(env, held_rows):
    for r in held_rows:
        R.restore(env, 1, r["qpos"], r["qvel"])
        g = R.g2_check(env, r["qpos"])
        assert g.passes, f"G2 rejected a correctly restored row: {g.as_row()}"
        assert g.drift_m < R.G2_DRIFT_M
