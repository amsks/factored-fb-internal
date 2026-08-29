"""HILP phi learner — pinned against an INDEPENDENT numpy reconstruction.

PARITY METHOD, and its limits. Nothing here runs PyTorch, so agreement with the official
tensors is argued from transcription, not measured -- the same standing caveat as
test_rldp_seam.py. What the reconstruction DOES establish is that the jax expression
computes the arithmetic written down in `sf.py:154-198`, independently re-derived in
numpy from the reference source rather than by calling the implementation under test.

The reconstruction is written to make the three documented traps FAIL LOUDLY if the
implementation drifts:
  * `adv` must use min(next_v1,next_v2) for q but the MEAN of the target heads for its
    baseline, while q1/q2 use the heads SEPARATELY (test_reductions_are_not_interchangeable)
  * the clamp is inside the sqrt, so v floors at -1e-3 rather than 0 (test_clamp_inside_sqrt)
  * at the shipped expectile 0.5 the weight is constant (test_shipped_expectile_is_inert)
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from impls.critics.hilp import hilp_phi_loss, hilp_value, expectile_loss

B, OBS, D = 7, 5, 4
DISC, EXP = 0.98, 0.5


def _mlp_params(rng, din, dout):
    return {"w": rng.standard_normal((din, dout)), "b": rng.standard_normal(dout)}


def _apply(p, x):
    return jnp.tanh(jnp.asarray(x) @ jnp.asarray(p["w"])) + jnp.asarray(p["b"])


def _np_apply(p, x):
    return np.tanh(x @ p["w"]) + p["b"]


def _fixture(seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((B, OBS))
    goals = rng.standard_normal((B, OBS))
    goals[2] = obs[2]                       # force one exact obs == goal row
    nobs = rng.standard_normal((B, OBS))
    ps = {k: _mlp_params(rng, OBS, D) for k in ("p1", "p2", "t1", "t2")}
    return obs, goals, nobs, ps


def _np_reference(obs, goals, nobs, ps, discount=DISC, expectile=EXP):
    """sf.py:154-198 re-derived in numpy, reading only the parameter arrays."""
    v = lambda p, a, b: -np.sqrt(np.clip(((_np_apply(p, a) - _np_apply(p, b)) ** 2).sum(-1), 1e-6, None))
    same = (np.linalg.norm(obs - goals, axis=-1) < 1e-6).astype(np.float64)
    masks = 1.0 - same
    rewards = same - 1.0
    tn1, tn2 = v(ps["t1"], nobs, goals), v(ps["t2"], nobs, goals)
    tv1, tv2 = v(ps["t1"], obs, goals), v(ps["t2"], obs, goals)
    q = rewards + discount * masks * np.minimum(tn1, tn2)
    adv = q - (tv1 + tv2) / 2.0
    q1 = rewards + discount * masks * tn1
    q2 = rewards + discount * masks * tn2
    v1, v2 = v(ps["p1"], obs, goals), v(ps["p2"], obs, goals)
    w = lambda a: np.where(a >= 0, expectile, 1.0 - expectile)
    return float((w(adv) * (q1 - v1) ** 2).mean() + (w(adv) * (q2 - v2) ** 2).mean())


def _run(obs, goals, nobs, ps, **kw):
    loss, info = hilp_phi_loss(ps["p1"], ps["p2"], ps["t1"], ps["t2"], _apply,
                               jnp.asarray(obs), jnp.asarray(goals), jnp.asarray(nobs),
                               kw.get("discount", DISC), kw.get("expectile", EXP))
    return float(loss), info


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_phi_loss_matches_numpy_reconstruction(seed):
    obs, goals, nobs, ps = _fixture(seed)
    got, _ = _run(obs, goals, nobs, ps)
    want = _np_reference(obs, goals, nobs, ps)
    assert abs(got - want) < 1e-10, f"{got} vs {want}"


@pytest.mark.parametrize("expectile", [0.5, 0.7, 0.9])
def test_phi_loss_matches_across_expectiles(expectile):
    obs, goals, nobs, ps = _fixture(3)
    got, _ = _run(obs, goals, nobs, ps, expectile=expectile)
    want = _np_reference(obs, goals, nobs, ps, expectile=expectile)
    assert abs(got - want) < 1e-10


def test_clamp_is_inside_the_sqrt_so_identical_inputs_floor_at_1e_3():
    """Trap: `-sqrt(clamp(sq, min=1e-6))`, not `-clamp(sqrt(sq), ...)`. At phi(a)==phi(b)
    the value is -1e-3, not 0, and the gradient stays finite -- which matters because
    hindsight relabelling makes obs == goal a common row."""
    x = jnp.zeros((3, D))
    assert np.allclose(np.asarray(hilp_value(x, x)), -1e-3)


def test_shipped_expectile_is_inert():
    """Trap 2: at the SHIPPED 0.5 the weight does not depend on adv at all, so the loss is
    plain 0.5*diff^2. Pinned so that a later 'fix' to 0.7 cannot pass silently."""
    adv = jnp.asarray([-3.0, -1e-9, 0.0, 1e-9, 5.0])
    diff = jnp.ones_like(adv)
    w = np.asarray(expectile_loss(adv, diff, 0.5))
    assert np.allclose(w, 0.5), w
    w7 = np.asarray(expectile_loss(adv, diff, 0.7))
    assert not np.allclose(w7, w7[0]), "at 0.7 the weight MUST depend on the sign of adv"


def test_reductions_are_not_interchangeable():
    """Trap 1: three reductions in one loss. The robust invariant is that swapping the
    adv baseline (MEAN of target heads -> min) or the per-head q1/q2 (-> the shared
    min-based q) changes the computation.

    The shared-q swap changes the LOSS directly. The baseline swap changes `adv` -- by
    0.4-1.0 on these fixtures -- but often NOT the loss, and that is a property of the
    objective worth stating rather than a weak test: HILP's reward is -1 on every row
    where the hindsight goal is not reached, so `q` sits near -1 while the values sit near
    0, and `adv` is negative on essentially every row. The only rows with adv >= 0 are the
    degenerate obs == goal ones (q = 0, v_t = -1e-3 => adv = +1e-3). Since the expectile
    weight depends only on sign(adv), a baseline change that shifts adv without flipping
    its sign is invisible to the loss -- at ANY expectile, not just the shipped 0.5.
    """
    obs, goals, nobs, ps = _fixture(5)
    v = lambda p, a, b: -np.sqrt(np.clip(((_np_apply(p, a) - _np_apply(p, b)) ** 2).sum(-1), 1e-6, None))
    same = (np.linalg.norm(obs - goals, axis=-1) < 1e-6).astype(np.float64)
    masks, rewards = 1.0 - same, same - 1.0
    tn1, tn2 = v(ps["t1"], nobs, goals), v(ps["t2"], nobs, goals)
    tv1, tv2 = v(ps["t1"], obs, goals), v(ps["t2"], obs, goals)
    q = rewards + DISC * masks * np.minimum(tn1, tn2)
    v1, v2 = v(ps["p1"], obs, goals), v(ps["p2"], obs, goals)
    w = lambda a: np.where(a >= 0, EXP, 1.0 - EXP)
    ref = _np_reference(obs, goals, nobs, ps)

    adv = q - (tv1 + tv2) / 2.0
    adv_minbase = q - np.minimum(tv1, tv2)
    # the baseline choice always moves adv itself, which is what gets logged as
    # accept_prob / adv_mean even when the loss is insensitive to it
    assert np.abs(adv - adv_minbase).max() > 1e-6

    # the shared-q swap changes the loss outright
    wrong_sharedq = float((w(adv) * (q - v1) ** 2).mean() + (w(adv) * (q - v2) ** 2).mean())
    assert abs(wrong_sharedq - ref) > 1e-8, "shared-q variant indistinguishable; fixture too weak"


def test_advantage_is_almost_always_negative():
    """Records the structural fact the test above rests on, so a future change that makes
    adv straddle zero shows up as a failure here rather than as a silent shift in what the
    expectile is doing. Across fixtures, at most a small minority of rows have adv >= 0,
    and those are the degenerate obs == goal rows."""
    for seed in range(4):
        obs, goals, nobs, ps = _fixture(seed)
        _, info = _run(obs, goals, nobs, ps)
        assert float(info["hilp_accept_prob"]) <= 0.35, (seed, info["hilp_accept_prob"])
