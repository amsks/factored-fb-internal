"""The persistence sampler, extended for REBALANCING (lam < 1 / a target transient mass) and for
an ONSET pool (rows within +-window of a motion onset) -- the mitigate-the-imbalance sanity
check: keep everything else the same, draw the rare bottleneck rows often enough to undo their
rarity, compare FB vs Density-FB."""
import numpy as np
import pytest
from impls.utils.datasets import FBDataset, make_dataset


def _base():
    # 2 trajectories x 100 steps; cube block = cols 3:6. Cube at rest, moves for steps 40-59
    # (onset at t=39 -> 40), at rest again. Trajectory 2 identical.
    n, T = 200, 100
    obs = np.zeros((n, 8), np.float32)
    for tr in range(2):
        for t in range(T):
            i = tr * T + t
            x = 0.0 if t < 40 else (0.01 * (t - 39) if t < 60 else 0.2)
            obs[i, 3:6] = x
    act = np.zeros((n, 2), np.float32)
    term = np.zeros((n,), np.float32); term[T - 1] = 1.0; term[-1] = 1.0
    return FBDataset.create(observations=obs, actions=act, terminals=term)


def test_motion_pool_lam_below_one_upweights_moving_rows():
    ds = _base()
    st = ds.enable_persist_oversample((3, 6), 0.005, lam=0.25)
    assert st["enabled"] and st["pool"] == "motion"
    # moving rows: steps 39..59 (obs[t+1]-obs[t] >= 0.005) x 2 trajectories, minus the
    # trajectory-end row which compares to the next trajectory's first row (a jump) -> count it
    assert st["ntransient"] > 0 and st["npersist"] > 0
    idx = ds.get_random_idxs(20000)
    frac_tr = np.isin(idx, ds._transient_pool).mean()
    assert frac_tr > st["ntransient"] / (st["ntransient"] + st["npersist"]) + 0.1
    assert abs(frac_tr - (1 - st["persist_mass"])) < 0.02


def test_transient_mass_target_sets_lam():
    ds = _base()
    st = ds.enable_persist_oversample((3, 6), 0.005, lam=1.0, transient_mass=0.5)
    assert abs(st["persist_mass"] - 0.5) < 1e-9
    idx = ds.get_random_idxs(40000)
    assert abs(np.isin(idx, ds._transient_pool).mean() - 0.5) < 0.02


def test_onset_pool_is_window_around_motion_onset():
    ds = _base()
    st = ds.enable_persist_oversample((3, 6), 0.005, lam=1.0, pool="onset", window=5,
                                      transient_mass=0.5)
    assert st["pool"] == "onset" and st["window"] == 5
    # onset row = 39 (at rest 38->39, moving 39->40); window +-5 -> rows 34..44, both trajectories
    exp = sorted([tr * 100 + t for tr in range(2) for t in range(34, 45)])
    assert sorted(ds._transient_pool.tolist()) == exp
    idx = ds.get_random_idxs(40000)
    assert abs(np.isin(idx, ds._transient_pool).mean() - 0.5) < 0.02
    # support preserved: every row still reachable
    assert len(np.union1d(ds._persist_pool, ds._transient_pool)) == 200


def test_make_dataset_wires_rebalance_keys():
    cfg = dict(z_dim=50, persist_oversample_lambda=1.0, persist_transient_mass=0.5,
               persist_pool="onset", persist_window=5, persist_block="3:6",
               persist_vel_thresh=0.005)
    ds = make_dataset(_base(), cfg)
    assert getattr(ds, "_oversample", False)
    assert abs(ds._persist_mass - 0.5) < 1e-9
    # lam=1 with no target is still the uniform sampler (existing lever semantics)
    ds2 = make_dataset(_base(), dict(z_dim=50, persist_oversample_lambda=1.0))
    assert not getattr(ds2, "_oversample", False)


def test_make_dataset_rejects_unquoted_block():
    # --override persist_block=19:22 (unquoted) is read by YAML 1.1 as the sexagesimal int
    # 1162; the loader must name the trap instead of dying on a tuple unpack.
    cfg = dict(z_dim=50, persist_transient_mass=0.5, persist_pool="onset", persist_window=5,
               persist_block=1162, persist_vel_thresh=0.005)
    with pytest.raises(ValueError, match="persist_block.*1162.*quote"):
        make_dataset(_base(), cfg)


# ==========================================================================================
# persist_mode="loss": same split, but a per-row CRITIC loss weight instead of resampling
# ==========================================================================================
def _gc_cfg(**over):
    from impls.utils.config_utils import load_config
    cfg = dict(load_config("density_fb", "flowbc", "cube_single"))
    cfg.update(over)
    return cfg


def test_row_weight_present_and_all_ones_when_off():
    ds = _base()
    b = ds.sample(32)
    assert b["row_weight"].shape == (32,) and b["row_weight"].dtype == np.float32
    assert np.all(b["row_weight"] == 1.0)
    gds = make_dataset(_base(), _gc_cfg())
    assert not getattr(gds.dataset, "_oversample", False)
    gb = gds.sample(32)
    assert gb["row_weight"].shape == (32,) and np.all(gb["row_weight"] == 1.0)
    # sample mode: the batch key is still there and still all ones (weights live in sampling)
    sds = make_dataset(_base(), dict(z_dim=50, persist_transient_mass=0.5, persist_pool="onset",
                                     persist_window=5, persist_block="3:6"))
    assert np.all(sds.sample(32)["row_weight"] == 1.0)


def test_loss_mode_weights_have_mean_one_and_target_transient_share():
    ds = _base()
    st = ds.enable_persist_oversample((3, 6), 0.005, pool="onset", window=5,
                                      transient_mass=0.5, mode="loss")
    assert st["enabled"] and st["mode"] == "loss" and st["W"] > 1.0
    assert not ds._oversample                        # sampling untouched
    w = ds._row_weight
    assert abs(w.mean() - 1.0) < 1e-6
    tr = ds._transient_pool
    assert abs(w[tr].sum() / w.sum() - 0.5) < 1e-6   # transient share of total weight
    assert np.all(w > 0)                             # support preserved
    assert abs(st["w_transient"] / st["w_persist"] - st["W"]) < 1e-9
    # W solves W*ntr/(W*ntr+npst) = transient_mass
    W, ntr, npst = st["W"], st["ntransient"], st["npersist"]
    assert abs(W * ntr / (W * ntr + npst) - 0.5) < 1e-9
    b = ds.sample(64)
    assert np.allclose(b["row_weight"], w[b["index"]])


def test_loss_mode_sampling_frequencies_unchanged():
    ds = _base()
    st = ds.enable_persist_oversample((3, 6), 0.005, pool="onset", window=5,
                                      transient_mass=0.5, mode="loss")
    idx = ds.get_random_idxs(40000)
    frac_tr = np.isin(idx, ds._transient_pool).mean()
    uniform = st["ntransient"] / (st["ntransient"] + st["npersist"])
    assert abs(frac_tr - uniform) < 0.02             # NOT 0.5: uniform draw


def test_make_dataset_wires_loss_mode():
    ds = make_dataset(_base(), dict(z_dim=50, persist_mode="loss", persist_transient_mass=0.5,
                                    persist_pool="onset", persist_window=5, persist_block="3:6"))
    assert not getattr(ds, "_oversample", False) and ds._row_weight is not None
    gds = make_dataset(_base(), _gc_cfg(persist_mode="loss", persist_transient_mass=0.5,
                                        persist_pool="onset", persist_window=5,
                                        persist_block="3:6"))
    gb = gds.sample(64)
    assert not np.all(gb["row_weight"] == 1.0)
    assert np.allclose(gb["row_weight"], gds.dataset._row_weight[gb["index"]])
    with pytest.raises(ValueError, match="persist mode"):
        make_dataset(_base(), dict(z_dim=50, persist_mode="bogus", persist_transient_mass=0.5,
                                   persist_block="3:6"))


# ---- critic-level: the critic loss reads row_weight, the actor loss does not --------------
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

_B, _OBS, _ACT = 16, 40, 5


class _TS:
    def __init__(self, params):
        self.params = params


def _net(params):
    return {k: _TS(v) for k, v in params.items()}


def _ex():
    rng = np.random.default_rng(0)
    return (jnp.asarray(rng.standard_normal((_B, _OBS))),
            jnp.asarray(np.tanh(rng.standard_normal((_B, _ACT)))))


def _crit_batch(seed=1):
    rng = np.random.default_rng(seed)
    return dict(
        observations=jnp.asarray(rng.standard_normal((_B, _OBS))),
        actions=jnp.asarray(np.tanh(rng.standard_normal((_B, _ACT)))),
        next_observations=jnp.asarray(rng.standard_normal((_B, _OBS))),
        value_goals=jnp.asarray(rng.standard_normal((_B, _OBS))),
        actor_goals=jnp.asarray(rng.standard_normal((_B, _OBS))),
        rewards=jnp.asarray(rng.integers(0, 2, size=_B) - 1.0),
        masks=jnp.asarray(rng.integers(0, 2, size=_B).astype(float)),
        row_weight=jnp.ones((_B,)),
    )


def _weights():
    w = np.ones(_B); w[::3] = 4.0
    return jnp.asarray(w / w.mean())


_ACTOR = dict(hidden_dim=32, hidden_layers=2, embedding_layers=2, flow_vf_hidden_dim=32,
              flow_vf_hidden_layers=2, flow_steps=2, bc_coeff=3.0)


def _actor_for(critic, ex_obs, ex_act):
    from impls.actors.flowbc import FlowBCActor
    return FlowBCActor.build(ex_obs, ex_act, dict(seed=0, actor=_ACTOR, cvec_dim=critic.cvec_dim))


def _check_weight_sensitivity(critic, stage, params, targets, actor, aparams, batch, cond):
    boot = actor.sample_bootstrap({**_net(params), **_net(aparams)}, batch, cond,
                                  jax.random.PRNGKey(2))
    allp = {**params, **aparams}
    b_ones = dict(batch)
    b_none = {k: v for k, v in batch.items() if k != "row_weight"}
    b_w = {**batch, "row_weight": _weights()}
    l_ones, _ = critic.stage_loss(stage, params, targets, b_ones, cond, boot)
    l_none, _ = critic.stage_loss(stage, params, targets, b_none, cond, boot)
    l_w, _ = critic.stage_loss(stage, params, targets, b_w, cond, boot)
    # all-ones == absent (the mode-off path is exactly the old loss); non-uniform moves it
    assert np.allclose(float(l_ones), float(l_none), rtol=0, atol=1e-12)
    assert abs(float(l_w) - float(l_ones)) > 1e-6 * max(1.0, abs(float(l_ones)))
    a_ones, _ = actor.loss(allp, targets, b_ones, critic, cond, boot)
    a_w, _ = actor.loss(allp, targets, b_w, critic, cond, boot)
    assert float(a_ones) == float(a_w)               # the actor never reads row_weight


def test_fb_critic_loss_reads_row_weight_actor_does_not():
    from impls.critics.fb import FBCritic
    ex_obs, ex_act = _ex()
    cfg = dict(seed=0, z_dim=8, L_dim=8, num_parallel=2, discount=0.99, ortho_coef=1.0,
               fb_pessimism_penalty=0.0, train_goal_ratio=0.5, norm_z=True,
               f_target_tau=0.005, b_target_tau=0.005,
               forward=dict(hidden_dim=32, hidden_layers=2, embedding_layers=2),
               backward=dict(hidden_dim=32, hidden_layers=2, norm=True),
               left_encoder=dict(hidden_dim=32, hidden_layers=2, norm=True))
    critic, params, targets = FBCritic.build(ex_obs, ex_act, cfg)
    actor, aparams = _actor_for(critic, ex_obs, ex_act)
    batch = _crit_batch()
    cond = critic.sample_cond(_net(params), batch, jax.random.PRNGKey(1))
    _check_weight_sensitivity(critic, "fb", params, targets, actor, aparams, batch, cond)


def test_density_fb_z_critic_loss_reads_row_weight_actor_does_not():
    from impls.critics.density_fb import DensityFBCritic
    ex_obs, ex_act = _ex()
    cfg = dict(lr=3e-4, lr_actor=3e-4, batch_size=_B, discount=0.99, tau=0.005, seed=0,
               value_hidden_dims=[32, 32], layer_norm=True, measure_hit_spec="19:22:0.4",
               measure_density_dim=8, measure_density_temp=1.0, actor_q_center=True,
               measure_density_z=True, measure_density_z_eval_pool_m=8,
               measure_density_hard=True)
    critic, params, targets = DensityFBCritic.build(ex_obs, ex_act, cfg)
    actor, aparams = _actor_for(critic, ex_obs, ex_act)
    batch = _crit_batch()
    cond = critic.sample_cond(_net(params), batch, jax.random.PRNGKey(1), target=targets)
    _check_weight_sensitivity(critic, "density", params, targets, actor, aparams, batch, cond)
