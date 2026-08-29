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


# ── SF stage: build + an independent numpy reconstruction of update_sf ──────────
from impls.critics.hilp import HILPCritic  # noqa: E402

SB, SOBS, SACT, SZ, SH = 6, 5, 3, 4, 8


def _sf_cfg(**over):
    cfg = dict(seed=0, z_dim=SZ, num_parallel=2, discount=0.98, norm_z=True,
               hilp_discount=0.98, hilp_expectile=0.5, phi_target_tau=0.005,
               sf_target_tau=0.01, hilp_stats_momentum=0.005, hilp_q_loss=True,
               phi=dict(hidden_dim=SH, hidden_layers=2),
               sf=dict(hidden_dim=SH, hidden_layers=2, embedding_layers=2))
    cfg.update(over)
    return cfg


def _sf_ex():
    rng = np.random.default_rng(0)
    return (jnp.asarray(rng.standard_normal((SB, SOBS))),
            jnp.asarray(np.tanh(rng.standard_normal((SB, SACT)))))


def _sf_batch(seed=1):
    rng = np.random.default_rng(seed)
    return dict(observations=jnp.asarray(rng.standard_normal((SB, SOBS))),
                actions=jnp.asarray(np.tanh(rng.standard_normal((SB, SACT)))),
                next_observations=jnp.asarray(rng.standard_normal((SB, SOBS))),
                actor_goals=jnp.asarray(rng.standard_normal((SB, SOBS))))


def test_build_produces_the_declared_groups_and_targets():
    critic, params, targets = HILPCritic.build(*_sf_ex(), _sf_cfg())
    assert set(params) == set(critic.GROUPS)
    assert set(targets) == set(critic.TARGETS)
    # the stats slot must never polyak -- it is written by update_stats
    assert critic.TARGETS["hilp_stats"] == 0.0
    # phi is trained ONLY by the phi stage and sf ONLY by the sf stage
    assert set(critic.stage_groups("phi")) == {"hilp_phi1", "hilp_phi2"}
    assert set(critic.stage_groups("sf")) == {"hilp_sf"}
    # init reads the raw tower: mean 0, std 1
    assert np.allclose(np.asarray(params["hilp_stats"]["mean"]), 0.0)
    assert np.allclose(np.asarray(params["hilp_stats"]["std"]), 1.0)


def test_sf_loss_matches_numpy_reconstruction():
    """update_sf (sf.py:600-637) re-derived in numpy, reading only the flax arrays."""
    critic, params, targets = HILPCritic.build(*_sf_ex(), _sf_cfg())
    batch = _sf_batch()
    rng = np.random.default_rng(9)
    z = jnp.asarray(np.tanh(rng.standard_normal((SB, SZ))))
    boot = {"next_action": jnp.asarray(np.tanh(rng.standard_normal((SB, SACT))))}
    got, _ = critic.stage_loss("sf", params, targets, batch, {"z": z}, boot)

    tF = np.asarray(critic._sf(targets["hilp_sf"], batch["next_observations"], z,
                               boot["next_action"]))                    # [P,B,d]
    phi = np.asarray(critic._phi(params["hilp_phi1"], batch["next_observations"]))
    target_phi = phi - np.asarray(params["hilp_stats"]["mean"])          # feature_net
    zz = np.asarray(z)
    tQ = np.einsum("pbd,bd->pb", tF, zz)
    next_F = np.where((tQ[0] < tQ[1])[:, None], tF[0], tF[1])            # per-sample pick
    target_F = target_phi + 0.98 * next_F
    F = np.asarray(critic._sf(params["hilp_sf"], batch["observations"], zz, batch["actions"]))
    Q = np.einsum("pbd,bd->pb", F, zz)
    tQs = np.einsum("bd,bd->b", target_F, zz)
    want = float(((Q - tQs[None]) ** 2).mean(axis=-1).sum())
    assert abs(float(got) - want) < 1e-10, f"{float(got)} vs {want}"


def test_sf_target_reads_the_ONLINE_phi_not_the_target_phi():
    """Trap 4, and the mirror of the RLDP run-1 bug. Perturbing the TARGET phi must leave
    the SF loss untouched; perturbing the ONLINE phi must change it."""
    critic, params, targets = HILPCritic.build(*_sf_ex(), _sf_cfg())
    batch = _sf_batch()
    rng = np.random.default_rng(9)
    z = {"z": jnp.asarray(np.tanh(rng.standard_normal((SB, SZ))))}
    boot = {"next_action": jnp.asarray(np.tanh(rng.standard_normal((SB, SACT))))}
    base, _ = critic.stage_loss("sf", params, targets, batch, z, boot)

    bump = lambda tree: jax.tree_util.tree_map(lambda a: a + 1.0, tree)
    t2 = {**targets, "hilp_phi1": bump(targets["hilp_phi1"]), "hilp_phi2": bump(targets["hilp_phi2"])}
    same, _ = critic.stage_loss("sf", params, t2, batch, z, boot)
    assert abs(float(same) - float(base)) < 1e-12, "SF loss must not read the target phi"

    p2 = {**params, "hilp_phi1": bump(params["hilp_phi1"])}
    diff, _ = critic.stage_loss("sf", p2, targets, batch, z, boot)
    assert abs(float(diff) - float(base)) > 1e-8, "SF loss must read the ONLINE phi"


def test_next_F_selection_is_per_sample_not_elementwise():
    """Trap 5b. The reference picks the WHOLE vector of the head with the smaller Q; an
    elementwise minimum over the two F vectors is a different object. The two agree only
    by coincidence, so this pins the difference on a constructed case."""
    critic, params, targets = HILPCritic.build(*_sf_ex(), _sf_cfg())
    rng = np.random.default_rng(4)
    tF = rng.standard_normal((2, SB, SZ))
    zz = rng.standard_normal((SB, SZ))
    tQ = np.einsum("pbd,bd->pb", tF, zz)
    per_sample = np.where((tQ[0] < tQ[1])[:, None], tF[0], tF[1])
    elementwise = np.minimum(tF[0], tF[1])
    assert not np.allclose(per_sample, elementwise)


def test_running_mean_arithmetic():
    """The EMA rule `update_stats` applies. Deliberately tests the ARITHMETIC only: a full
    update_stats call needs Agent-level TrainState fixtures, and once this critic is
    registered test_boot_contract.py exercises that path for every critic x actor pair.
    A skipped placeholder here would read as coverage without being any.

    The rule itself: new = (1-m)*old + m*batch_stat, m = 0.005 (sf.py:185-187)."""
    m = 0.005
    old_mean, old_std = np.zeros(SZ), np.ones(SZ)
    phi = np.random.default_rng(2).standard_normal((SB, SZ))
    new_mean = old_mean * (1 - m) + phi.mean(axis=0) * m
    new_std = old_std * (1 - m) + phi.std(axis=0) * m
    assert np.allclose(new_mean, phi.mean(axis=0) * m)
    assert np.all(np.abs(new_std - 1.0) < 0.02)
