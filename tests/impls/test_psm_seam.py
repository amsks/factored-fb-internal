"""PSM on the seam: numerical parity with the PyTorch reference + all four actor families.

Parity
------
`tests/impls/fixtures/psm_reference.npz` was exported from the PyTorch reference in this
repo's torch tree (`agents/psm/`) under the torch venv, in float64 on CPU, by
`tests/impls/gen_psm_reference_fixture.py` (which runs THERE, not here). It builds a real
`PSMAgent`, dumps every weight, then calls `PSMAgent._update_psm` / `_update_sf` through
their `next_action=` INJECTION HOOKS and records the loss scalars the reference itself
returns (which are computed before its optimizer step).

Why the injection hooks matter: the reference's proto behaviour sampler is deterministic
(a fixed `seed_to_action` table built with `torch.random.manual_seed(i)`), but it is
deterministic *in torch's MT19937* and our JAX port draws with threefry instead -- see
deviation 1 in impls/critics/psm.py. Bypassing the sampler on BOTH sides removes it from
the comparison entirely, so what is pinned here is the loss math and the network
transcription, not a PRNG coincidence. The proto sampler is separately tested below for
the properties that actually matter (purity, key sensitivity, range, seed arithmetic).

The SF stage reads the phi that the PSM stage just stepped, so the fixture also carries
the reference's POST-step phi weights (`w1__phi.*`) and this test loads them for the SF
comparison -- that isolates the SF loss math from Adam-implementation differences.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from impls.agents import Agent, CRITICS, _LR_KEY
from impls.critics.psm import PSMCritic, PsmConfigError, proto_actions
from impls.utils import torch_to_flax as t2f
from impls.utils.config_utils import REQUIRED_KEYS, ConfigError, load_config

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "psm_reference.npz")
ATOL = 1e-10
OBS, ACT = 8, 5


@pytest.fixture(scope="module")
def fx():
    assert os.path.exists(FIXTURE), f"missing {FIXTURE}"
    return dict(np.load(FIXTURE))


def _f(fx, key):
    return jnp.asarray(fx[key], jnp.float64)


# ───────────────────────── seam config / registration ───────────────────────── #

def test_registered():
    assert CRITICS["psm"] is PSMCritic
    assert _LR_KEY["phi"] == "lr_phi"
    assert _LR_KEY["sf_psi"] == _LR_KEY["psm_psi"] == "lr_sf"
    assert "psm" in REQUIRED_KEYS


def test_class_attrs():
    assert PSMCritic.GROUPS == ("phi", "sf_psi", "psm_psi")
    assert PSMCritic.STAGES == ("psm", "sf")
    assert PSMCritic.PROVIDES_V is False
    assert PSMCritic.PROVIDES_COND == frozenset({"z"})
    assert PSMCritic.BOOT_COND == "actor_goal"


def test_shipped_config_composes_and_builds():
    cfg = load_config("psm", "flowbc", "cube_single")
    cfg["batch_size"] = 4
    agent = Agent.create("psm", "flowbc", 0, jnp.zeros((4, OBS)), jnp.zeros((4, ACT)), cfg)
    assert set(agent.net) >= set(PSMCritic.GROUPS)
    assert set(agent.target) == set(PSMCritic.GROUPS)
    assert agent.critic.cvec_dim == cfg["z_dim"]
    # every group is polyak'd at the single target_tau, as the reference does
    assert set(agent.critic.TARGETS) == set(PSMCritic.GROUPS)
    assert set(agent.critic.TARGETS.values()) == {cfg["target_tau"]}


def test_missing_required_key_is_a_named_error():
    with pytest.raises(ConfigError, match="proto_seed"):
        load_config("psm", "flowbc", "cube_single", overrides=["~proto_seed"])


@pytest.mark.parametrize("bad", [dict(num_parallel=1), dict(max_log_seed=0),
                                 dict(max_log_seed=31)])
def test_structurally_invalid_config_raises_at_build(bad):
    cfg = {**_cfg("flowbc"), **bad}
    with pytest.raises(PsmConfigError):
        Agent.create("psm", "flowbc", 0, jnp.zeros((4, OBS)), jnp.zeros((4, ACT)), cfg)


# ───────────────────────── numerical parity vs the torch reference ──────────── #

def _critic_from_fixture(fx):
    """A PSMCritic at the fixture's dims, with the reference's weights loaded."""
    d = lambda k: int(fx[k])
    cfg = dict(
        z_dim=d("dim_z"), max_log_seed=d("dim_max_log_seed"),
        num_parallel=d("dim_num_parallel"), discount=float(fx["cfg_discount"]),
        ortho_coef=float(fx["cfg_ortho_coef"]),
        pessimism_penalty=float(fx["cfg_pessimism"]),
        actor_pessimism_penalty=float(fx["cfg_actor_pessimism"]),
        mix_ratio=0.5, norm_z=True, target_tau=0.01, proto_seed=0, seed=0,
        phi=dict(hidden_dim=d("dim_phi_hidden"), hidden_layers=2, norm=True),
        sf=dict(hidden_dim=d("dim_psi_hidden"), hidden_layers=1, embedding_layers=2),
    )
    B = d("dim_batch")
    critic, _, _ = PSMCritic.build(jnp.zeros((B, d("dim_obs"))), jnp.zeros((B, d("dim_act"))), cfg)
    params = {
        "phi": t2f.load_phi_params(fx),
        "sf_psi": t2f.load_psi_params(fx, "sf_psi"),
        "psm_psi": t2f.load_psi_params(fx, "psm_psi"),
    }
    return critic, params


def _batch_from_fixture(fx):
    return {
        "observations": _f(fx, "in__observations"),
        "actions": _f(fx, "in__actions"),
        "next_observations": _f(fx, "in__next_observations"),
        "terminals": _f(fx, "in__terminated"),
        # The torch generator did not pass batch["index"], so the reference took its
        # arange(B) fallback; reproduce that exactly or the parity comparison is invalid.
        "index": np.arange(_f(fx, "in__observations").shape[0]),
    }


def test_phi_forward_matches_reference(fx):
    critic, params = _critic_from_fixture(fx)
    out = critic._phi(params["phi"], _f(fx, "in__goal"))
    np.testing.assert_allclose(np.asarray(out), fx["out__phi_goal"], atol=ATOL)


def test_sf_psi_forward_matches_reference(fx):
    critic, params = _critic_from_fixture(fx)
    out = critic._sf(params["sf_psi"], _f(fx, "in__observations"),
                     _f(fx, "in__z_sf"), _f(fx, "in__actions"))
    np.testing.assert_allclose(np.asarray(out), fx["out__sf_psi"], atol=ATOL)


def test_psm_psi_forward_matches_reference(fx):
    """The proto tower's z leg is max_log_seed wide, not z_dim wide."""
    critic, params = _critic_from_fixture(fx)
    out = critic._psm(params["psm_psi"], _f(fx, "in__observations"),
                      _f(fx, "in__z_psm"), _f(fx, "in__actions"))
    np.testing.assert_allclose(np.asarray(out), fx["out__psm_psi"], atol=ATOL)


def test_q_readout_matches_reference(fx):
    """q_fn is <sf_psi, z> reduced by mean - actor_pessimism_penalty * uncertainty."""
    critic, params = _critic_from_fixture(fx)
    q, qs = critic.q_fn(params, _f(fx, "in__observations"),
                        {"z": _f(fx, "in__z_sf")}, _f(fx, "in__actions"))
    np.testing.assert_allclose(np.asarray(qs), fx["out__qs"], atol=ATOL)
    np.testing.assert_allclose(np.asarray(q), fx["out__q"], atol=ATOL)


def test_infer_cond_matches_reference_reward_inference(fx):
    critic, params = _critic_from_fixture(fx)

    class _S:  # minimal stand-in for a TrainState
        def __init__(self, p):
            self.params = p

    net = {"phi": _S(params["phi"])}
    z = critic.infer_cond(net, _f(fx, "in__next_observations"), _f(fx, "in__rewards"))["z"]
    np.testing.assert_allclose(np.asarray(z), fx["out__infer_z"].reshape(-1), atol=ATOL)


def test_psm_stage_loss_matches_reference(fx):
    """The proto branch: contrastive measure loss + ortho_coef * phi orthonormality."""
    critic, params = _critic_from_fixture(fx)
    batch = _batch_from_fixture(fx)
    cond = {"z_psm": _f(fx, "in__z_psm"), "psm_next_action": _f(fx, "in__next_action_psm")}
    loss, info = critic.stage_loss("psm", params, params, batch, cond, {})
    for ours, theirs in (("psm_loss", "psm_loss"), ("psm_diag", "psm_diag"),
                         ("psm_offdiag", "psm_offdiag"), ("orth_loss", "orth_loss"),
                         ("orth_loss_diag", "orth_loss_diag"),
                         ("orth_loss_offdiag", "orth_loss_offdiag"),
                         ("phi_norm", "phi_norm"), ("psm_M_diag", "psm_M_diag"),
                         ("psm_targetM_diag", "psm_targetM_diag")):
        np.testing.assert_allclose(float(info[ours]), float(fx[f"loss__{theirs}"]),
                                   atol=ATOL, rtol=0, err_msg=ours)
    np.testing.assert_allclose(float(loss), float(fx["loss__psm_loss"]), atol=ATOL, rtol=0)


def test_sf_stage_loss_matches_reference(fx):
    """The SF branch, at the reference's POST-psm-step phi (see module docstring).

    Also pins the two easy-to-get-wrong details: the SF target measure uses the ONLINE
    phi (not target_phi), and the SF loss carries NO orthonormality term.
    """
    critic, params = _critic_from_fixture(fx)
    params = {**params, "phi": t2f.load_phi_params(fx, ns="w1__")}
    batch = _batch_from_fixture(fx)
    cond = {"z": _f(fx, "in__z_sf")}
    boot = {"next_action": _f(fx, "in__next_action_sf")}
    loss, info = critic.stage_loss("sf", params, params, batch, cond, boot)
    for k in ("sf_loss", "sf_diag", "sf_offdiag", "sf_phi_norm", "sf_M_diag",
              "sf_targetM_diag"):
        np.testing.assert_allclose(float(info[k]), float(fx[f"loss__{k}"]),
                                   atol=ATOL, rtol=0, err_msg=k)
    np.testing.assert_allclose(float(loss), float(fx["loss__sf_loss"]), atol=ATOL, rtol=0)


def test_sf_stage_target_uses_online_phi_not_target_phi(fx):
    """Guards reference _update_sf line 499 against a "fix" to target_phi: perturbing the
    TARGET phi must not move the SF loss at all, while perturbing the ONLINE phi must."""
    critic, params = _critic_from_fixture(fx)
    batch = _batch_from_fixture(fx)
    cond, boot = {"z": _f(fx, "in__z_sf")}, {"next_action": _f(fx, "in__next_action_sf")}
    base = float(critic.stage_loss("sf", params, params, batch, cond, boot)[0])

    bumped = jax.tree_util.tree_map(lambda a: a + 0.1, params["phi"])
    tgt_only = float(critic.stage_loss("sf", params, {**params, "phi": bumped},
                                       batch, cond, boot)[0])
    online = float(critic.stage_loss("sf", {**params, "phi": bumped}, params,
                                     batch, cond, boot)[0])
    assert tgt_only == base, "SF stage must not read target_phi"
    assert online != base, "SF stage must read the online phi"


def test_psm_stage_target_uses_target_phi(fx):
    """The mirror of the above for the proto branch, which DOES use target_phi."""
    critic, params = _critic_from_fixture(fx)
    batch = _batch_from_fixture(fx)
    cond = {"z_psm": _f(fx, "in__z_psm"), "psm_next_action": _f(fx, "in__next_action_psm")}
    base = float(critic.stage_loss("psm", params, params, batch, cond, {})[0])
    bumped = jax.tree_util.tree_map(lambda a: a + 0.1, params["phi"])
    assert float(critic.stage_loss("psm", params, {**params, "phi": bumped},
                                   batch, cond, {})[0]) != base


def test_row_discount_follows_terminals(fx):
    """gamma * (1 - terminated) per ROW (reference agent.py update()): flipping one
    row's terminal flag must move the loss."""
    critic, params = _critic_from_fixture(fx)
    batch = _batch_from_fixture(fx)
    cond = {"z_psm": _f(fx, "in__z_psm"), "psm_next_action": _f(fx, "in__next_action_psm")}
    base = float(critic.stage_loss("psm", params, params, batch, cond, {})[0])
    flipped = {**batch, "terminals": 1.0 - batch["terminals"]}
    assert float(critic.stage_loss("psm", params, params, flipped, cond, {})[0]) != base
    # a GCDataset batch carries `masks` (= 1 - terminated) instead; same discount
    gc = {k: v for k, v in batch.items() if k != "terminals"}
    gc["masks"] = 1.0 - batch["terminals"]
    np.testing.assert_allclose(
        float(critic.stage_loss("psm", params, params, gc, cond, {})[0]), base, atol=ATOL)


# ───────────────────────── proto behaviour sampler ──────────────────────────── #

def test_proto_sampler_is_pure_and_key_sensitive():
    seeds = jnp.arange(16)
    a = proto_actions(seeds, 5, jax.random.PRNGKey(0))
    assert np.array_equal(np.asarray(a), np.asarray(proto_actions(seeds, 5, jax.random.PRNGKey(0))))
    assert not np.allclose(np.asarray(a), np.asarray(proto_actions(seeds, 5, jax.random.PRNGKey(1))))
    # distinct seeds -> distinct actions
    assert len({tuple(np.round(r, 12)) for r in np.asarray(a)}) == 16
    # the reference draws (rand - 1) * 2, i.e. the [-2, 0] range -- reproduced verbatim
    assert float(a.min()) >= -2.0 and float(a.max()) <= 0.0


def test_proto_action_is_a_function_of_the_binary_code_and_the_row():
    """Reference seed arithmetic: ((z_bin . reversed_powers) + row_hash) mod max_seed.
    Two rows with the same code but different indices must get different actions, and the
    same (code, row) must always get the same one."""
    cfg = _cfg("flowbc", max_log_seed=4, z_dim=6)
    agent = Agent.create("psm", "flowbc", 0, jnp.zeros((4, OBS)), jnp.zeros((4, ACT)), cfg)
    c = agent.critic
    base = jax.random.PRNGKey(c.proto_seed)
    code = jnp.array([[1.0, 0.0, 1.0, 1.0]])
    powers = 2 ** jnp.arange(c.max_log_seed)[::-1]
    s = (code.astype(jnp.int32) * powers).sum(1)
    a0 = proto_actions((s + 0) % c.max_seed, ACT, base)
    a1 = proto_actions((s + 1) % c.max_seed, ACT, base)
    assert not np.allclose(np.asarray(a0), np.asarray(a1))
    np.testing.assert_array_equal(np.asarray(a0),
                                  np.asarray(proto_actions((s + 0) % c.max_seed, ACT, base)))


def test_sample_cond_shapes_and_binary_code():
    cfg = _cfg("flowbc")
    agent = Agent.create("psm", "flowbc", 0, jnp.zeros((4, OBS)), jnp.zeros((4, ACT)), cfg)
    cond = agent.critic.sample_cond(agent.net, _batch(), jax.random.PRNGKey(0))
    B = _batch()["observations"].shape[0]
    assert cond["z"].shape == (B, cfg["z_dim"])
    assert cond["z_psm"].shape == (B, cfg["max_log_seed"])
    assert cond["psm_next_action"].shape == (B, ACT)
    assert set(np.unique(np.asarray(cond["z_psm"]))) <= {0.0, 1.0}
    # norm_z=True => every z row has norm sqrt(z_dim)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(cond["z"]), axis=-1),
                               np.sqrt(cfg["z_dim"]), rtol=1e-6)
    # cond_vec is z, so cvec_dim must match what actors were built for
    assert agent.critic.cond_vec(cond).shape[-1] == agent.critic.cvec_dim


# ───────────────────────── the four actor families ──────────────────────────── #

def _cfg(actor_kind, **over):
    base = dict(
        z_dim=6, max_log_seed=4, num_parallel=2, discount=0.99, ortho_coef=1.0,
        pessimism_penalty=0.0, actor_pessimism_penalty=0.5, mix_ratio=0.5, norm_z=True,
        target_tau=0.01, proto_seed=0, batch_size=16,
        lr_phi=1e-4, lr_sf=1e-4, lr_actor=1e-4,
        phi=dict(hidden_dim=16, hidden_layers=2, norm=True),
        sf=dict(hidden_dim=32, hidden_layers=1, embedding_layers=2),
    )
    if actor_kind == "flowbc":
        base["actor"] = dict(type="flow", hidden_dim=32, hidden_layers=2, embedding_layers=2,
                             bc_coeff=3.0, flow_steps=5, lr_actor_vf=3e-4,
                             flow_vf_hidden_dim=32, flow_vf_hidden_layers=2)
    elif actor_kind == "fql":
        base["actor"] = dict(hidden_dims=[32, 32], alpha=3.0, flow_steps=5,
                             const_std=True, layer_norm=False, q_agg="mean",
                             lr_actor_vf=3e-4)
    else:
        base["actor"] = dict(hidden_dims=[32, 32], alpha=3.0, const_std=True, layer_norm=False)
    base.update(over)
    return base


def _batch(B=16, seed=0):
    rng = np.random.default_rng(seed)
    return {"observations": rng.standard_normal((B, OBS)).astype(np.float32),
            "actions": np.tanh(rng.standard_normal((B, ACT))).astype(np.float32),
            "next_observations": rng.standard_normal((B, OBS)).astype(np.float32),
            "terminals": rng.integers(0, 2, size=B).astype(np.float32),
            "index": np.arange(B)}


ACTOR_FAMILIES = ("flowbc", "ddpgbc", "awr", "fql")


@pytest.mark.parametrize("actor_kind", ACTOR_FAMILIES)
def test_update_runs_and_moves_every_group(actor_kind):
    """A real `Agent.update()` per actor family: finite losses, and EVERY critic group's
    parameters (and target) actually move -- a stage that silently receives no gradient
    would otherwise pass a smoke test."""
    agent = Agent.create("psm", actor_kind, 0, jnp.zeros((4, OBS)), jnp.zeros((4, ACT)),
                         _cfg(actor_kind))
    before = jax.tree_util.tree_map(np.asarray, {g: agent.net[g].params for g in PSMCritic.GROUPS})
    tbefore = jax.tree_util.tree_map(np.asarray, dict(agent.target))

    info = None
    for s in range(3):
        agent, info = agent.update(_batch(seed=s))

    for k in ("psm_loss", "psm_diag", "psm_offdiag", "orth_loss",
              "sf_loss", "sf_diag", "sf_offdiag", "actor_loss"):
        assert jnp.isfinite(jnp.asarray(info[k])).all(), (actor_kind, k)

    for g in PSMCritic.GROUPS:
        after = jax.tree_util.tree_map(np.asarray, agent.net[g].params)
        moved = any(not np.array_equal(a, b) for a, b in
                    zip(jax.tree_util.tree_leaves(before[g]), jax.tree_util.tree_leaves(after)))
        assert moved, f"{actor_kind}: critic group {g!r} did not move"
        tafter = jax.tree_util.tree_map(np.asarray, agent.target[g])
        tmoved = any(not np.array_equal(a, b) for a, b in
                     zip(jax.tree_util.tree_leaves(tbefore[g]), jax.tree_util.tree_leaves(tafter)))
        assert tmoved, f"{actor_kind}: target group {g!r} was not polyak'd"

    for g in agent.actor.GROUPS:
        assert jnp.isfinite(jnp.concatenate(
            [jnp.ravel(x) for x in jax.tree_util.tree_leaves(agent.net[g].params)])).all()


@pytest.mark.parametrize("actor_kind", ACTOR_FAMILIES)
def test_sample_actions_is_in_range_and_goal_sensitive(actor_kind):
    """map_eval_cond embeds the env goal as z = project_z(phi(goal)); without it the eval
    policy runs at eval_cond0's zero z, for which Q = <psi, 0> == 0 and the policy is
    goal-blind (the FB-zero bug). Two different goals must give two different actions."""
    agent = Agent.create("psm", actor_kind, 0, jnp.zeros((4, OBS)), jnp.zeros((4, ACT)),
                         _cfg(actor_kind))
    obs = jnp.asarray(_batch(B=3)["observations"])
    g1 = jnp.asarray(np.random.default_rng(1).standard_normal((3, OBS)))
    g2 = jnp.asarray(np.random.default_rng(2).standard_normal((3, OBS)))
    a1 = agent.sample_actions(obs, goals=g1, seed=jax.random.PRNGKey(0))
    a2 = agent.sample_actions(obs, goals=g2, seed=jax.random.PRNGKey(0))
    assert a1.shape == (3, ACT)
    assert np.all(np.abs(np.asarray(a1)) <= 1.0 + 1e-6)
    assert not np.allclose(np.asarray(a1), np.asarray(a2))


def test_stage_order_is_psm_then_sf_and_sf_sees_the_stepped_phi():
    """The reference steps phi in _update_psm and then _update_sf reads that stepped phi.
    The seam reproduces it only because STAGES is ordered and the generic update re-reads
    `online()` per stage -- assert the observable consequence rather than the ordering."""
    agent = Agent.create("psm", "flowbc", 0, jnp.zeros((4, OBS)), jnp.zeros((4, ACT)),
                         _cfg("flowbc"))
    assert agent.critic.STAGES == ("psm", "sf")
    batch = _batch()
    cond = agent.critic.sample_cond(agent.net, batch, jax.random.PRNGKey(0))
    boot = agent.actor.sample_bootstrap(agent.net, batch, cond, jax.random.PRNGKey(1))

    # sf_loss at the PRE-update phi (compute_static; no stage ever steps)
    static = agent.compute_static(batch, cond, boot)
    stepped, live = agent.apply_update(batch, cond, boot)
    phi0 = jax.tree_util.tree_leaves(agent.net["phi"].params)[0]
    phi1 = jax.tree_util.tree_leaves(stepped.net["phi"].params)[0]
    assert not np.array_equal(np.asarray(phi0), np.asarray(phi1)), "psm stage must step phi"
    assert float(live["sf_loss"]) != float(static["sf_loss"]), (
        "the sf stage must see the phi the psm stage just stepped")


def test_psm_cond_keys_never_reach_an_actor():
    """z_psm / psm_next_action are critic-private: PROVIDES_COND advertises only "z", and
    no actor emits a value-goal draw for psm (BOOT_COND is "actor_goal")."""
    for actor_kind in ACTOR_FAMILIES:
        agent = Agent.create("psm", actor_kind, 0, jnp.zeros((4, OBS)), jnp.zeros((4, ACT)),
                             _cfg(actor_kind))
        batch = _batch()
        cond = agent.critic.sample_cond(agent.net, batch, jax.random.PRNGKey(0))
        boot = agent.actor.sample_bootstrap(agent.net, batch, cond, jax.random.PRNGKey(1))
        assert "_cvec_vgoal" not in cond
        assert "next_action_vgoal" not in boot
        assert agent.critic.cond_vec(cond).shape == cond["z"].shape


def test_proto_policy_is_keyed_on_the_dataset_row_id(): 
    """PSM assigns a proto behaviour policy per transition via
    `seed = (z_bits + row_hash) % max_seed` (reference ProtoBehaviorSampler). `row_hash`
    must be the DATASET row id -- batch["index"], which train.py enables with
    with_index=True for psm -- and NOT the batch position.

    Note what is NOT asserted: the seed also carries `z_bits` from a per-row random z, so
    the proto action legitimately differs between draws even for a fixed row. The property
    that distinguishes correct from broken is narrower: the row id must actually be
    CONSUMED. Keyed on position, arange(B) is the same vector for every batch of size B,
    so the output would be invariant to the ids below.
    """
    import numpy as np
    import jax, jax.numpy as jnp
    from impls.agents import Agent
    from impls.utils.config_utils import load_config

    cfg = load_config("psm", "flowbc", "cube_single")
    B, OBS, ACT = 8, 40, 5
    cfg["batch_size"] = B
    agent = Agent.create("psm", "flowbc", 0, jnp.zeros((B, OBS)), jnp.zeros((B, ACT)), cfg)

    r = np.random.default_rng(0)
    base = dict(observations=jnp.asarray(r.standard_normal((B, OBS))),
                actions=jnp.asarray(np.tanh(r.standard_normal((B, ACT)))),
                next_observations=jnp.asarray(r.standard_normal((B, OBS))),
                rewards=jnp.zeros((B, 1)), terminals=jnp.zeros((B,)))

    def act(index):
        cond = agent.critic.sample_cond(agent.net, {**base, "index": jnp.asarray(index)},
                                        jax.random.PRNGKey(0))
        return np.asarray(cond["psm_next_action"])

    positional = act(np.arange(B))                 # what the broken fallback produced
    real_rows = act(np.array([5, 11, 23, 4, 9, 1, 6, 2]))
    other_rows = act(np.array([101, 7, 42, 3, 88, 12, 60, 9]))

    assert not np.allclose(positional, real_rows), \
        "row ids are ignored -- the proto policy is still keyed on batch position"
    assert not np.allclose(real_rows, other_rows), \
        "different dataset rows produced identical proto actions"
    # and it is deterministic given the same rows
    assert np.allclose(real_rows, act(np.array([5, 11, 23, 4, 9, 1, 6, 2])))


def test_psm_refuses_a_batch_without_the_row_index():
    """Silently falling back to arange(B) is the defect; it must raise instead."""
    import numpy as np
    import jax, jax.numpy as jnp
    import pytest as _pytest
    from impls.agents import Agent
    from impls.utils.config_utils import load_config

    cfg = load_config("psm", "flowbc", "cube_single")
    B, OBS, ACT = 8, 40, 5
    cfg["batch_size"] = B
    agent = Agent.create("psm", "flowbc", 0, jnp.zeros((B, OBS)), jnp.zeros((B, ACT)), cfg)
    r = np.random.default_rng(0)
    no_index = dict(observations=jnp.asarray(r.standard_normal((B, OBS))),
                    actions=jnp.asarray(np.tanh(r.standard_normal((B, ACT)))),
                    next_observations=jnp.asarray(r.standard_normal((B, OBS))),
                    rewards=jnp.zeros((B, 1)), terminals=jnp.zeros((B,)))
    with _pytest.raises(KeyError, match="index"):
        agent.critic.sample_cond(agent.net, no_index, jax.random.PRNGKey(0))
