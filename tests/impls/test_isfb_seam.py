"""ISFB on the seam: construction, staging, PROVIDES_V/COND, and smoke training.

ISFB is the first FB-family critic with PROVIDES_V=True, so (isfb, awr) is a genuinely new
capability — awr x fb still raises ComboError, awr x isfb must construct and train.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from impls.agents import Agent, CRITICS, _LR_KEY
from impls.critics.base import ComboError, check_combo
from impls.critics.isfb import ISFBCritic, IsfbConfigError
from impls.utils.config_utils import REQUIRED_KEYS, load_config

OBS, ACT = 8, 5


def _cfg(actor_kind, **over):
    base = dict(
        z_dim=6, L_dim=4, num_parallel=3, discount=0.99, ortho_coef=1.0,
        fb_pessimism_penalty=0.0, train_goal_ratio=0.5, norm_z=True,
        f_target_tau=0.005, b_target_tau=0.005, lr_f=1e-4, lr_b=1e-4, lr_actor=1e-4,
        batch_size=16,
        forward=dict(hidden_dim=32, hidden_layers=2, embedding_layers=2),
        backward=dict(hidden_dim=32, hidden_layers=4, norm=True),
        left_encoder=dict(hidden_dim=32, hidden_layers=4, norm=True),
        measure_insample="scalar_gate_hard", iql_expectile=0.9,
        alpha_n=1.0, alpha_m=1.0, alpha_v=1.0, awr_beta=3.0, awr_wmax=100.0,
        awr_normalize_adv=True, goal_mask=True,
    )
    if actor_kind == "flowbc":
        base["actor"] = dict(type="flow", hidden_dim=32, hidden_layers=2, embedding_layers=2,
                             bc_coeff=3.0, flow_steps=5, lr_actor_vf=3e-4,
                             flow_vf_hidden_dim=32, flow_vf_hidden_layers=2)
    else:
        base["actor"] = dict(hidden_dims=[32, 32], alpha=3.0, const_std=True, layer_norm=False)
    base.update(over)
    return base


def _batch(B=16, seed=0):
    rng = np.random.default_rng(seed)
    return {"observations": rng.standard_normal((B, OBS)).astype(np.float32),
            "actions": np.tanh(rng.standard_normal((B, ACT))).astype(np.float32),
            "next_observations": rng.standard_normal((B, OBS)).astype(np.float32),
            "masks": rng.integers(0, 2, size=B).astype(np.float32)}


def _agent(actor_kind, **over):
    return Agent.create("isfb", actor_kind, 0, jnp.zeros((2, OBS)), jnp.zeros((2, ACT)),
                        _cfg(actor_kind, **over))


def test_registered():
    assert CRITICS["isfb"] is ISFBCritic
    assert _LR_KEY["value_forward"] == "lr_f"   # F_V rides the forward group's lr


def test_bad_measure_insample_raises_named_error_at_build():
    """FIX 3: a bad measure_insample enum must raise the named IsfbConfigError at BUILD time
    (Agent.create), matching every other build-validated critic — not a bare ValueError deep
    inside stage_loss at trace time. Before the fix this only died on `raise ValueError` when
    the update actually ran."""
    with pytest.raises(IsfbConfigError, match="measure_insample must be one of"):
        _agent("ddpgbc", measure_insample="soft_gate")


def test_class_attrs():
    assert ISFBCritic.STAGES == ("isfb",)
    assert ISFBCritic.GROUPS == ("forward", "value_forward", "backward", "left_encoder")
    assert ISFBCritic.PROVIDES_V is True
    assert ISFBCritic.PROVIDES_COND == frozenset({"z"})


def test_targets_cover_all_groups_with_fb_taus():
    critic, params, targets = ISFBCritic.build(jnp.zeros((2, OBS)), jnp.zeros((2, ACT)),
                                               {**_cfg("flowbc"), "seed": 0})
    assert set(params) == set(ISFBCritic.GROUPS) == set(targets)
    t = critic.TARGETS
    assert t["value_forward"] == t["forward"] == critic.f_tau
    assert t["backward"] == critic.b_tau
    assert critic.cvec_dim == critic.z_dim


def test_coord_mode_forces_alpha_v_zero():
    critic, _, _ = ISFBCritic.build(jnp.zeros((2, OBS)), jnp.zeros((2, ACT)),
                                    {**_cfg("flowbc", measure_insample="coord", alpha_v=1.0),
                                     "seed": 0})
    assert critic.alpha_v == 0.0


def test_awr_constructs_against_isfb_but_not_fb():
    """The new capability: an FB-family critic that satisfies AWRActor.NEEDS_V."""
    from impls.actors.awr import AWRActor
    from impls.critics.fb import FBCritic
    check_combo(ISFBCritic, AWRActor)                    # must not raise
    with pytest.raises(ComboError):
        check_combo(FBCritic, AWRActor)
    assert _agent("awr") is not None


@pytest.mark.parametrize("actor_kind", ["flowbc", "awr"])
@pytest.mark.parametrize("mode", ["coord", "scalar_gate_hard", "scalar_gate_awr"])
def test_smoke_trains(actor_kind, mode):
    agent = _agent(actor_kind, measure_insample=mode)
    info = None
    for s in range(3):
        agent, info = agent.update(_batch(seed=s))
    for k in ("isfb_loss", "isfb_L_N", "isfb_L_M", "ortho_loss", "actor_loss"):
        assert jnp.isfinite(jnp.asarray(info[k])).all(), (actor_kind, mode, k)
    if mode == "coord":
        assert float(info["isfb_L_V"]) == 0.0


def test_v_fn_is_action_independent_and_q_shaped():
    critic, params, targets = ISFBCritic.build(jnp.zeros((2, OBS)), jnp.zeros((2, ACT)),
                                               {**_cfg("flowbc"), "seed": 0})
    b = _batch()
    # a NON-zero z: q = <F, z> is identically 0 at z = 0, which would make the
    # action-dependence check vacuous.
    cond = {"z": jnp.asarray(np.random.default_rng(1).standard_normal(
        (b["observations"].shape[0], critic.z_dim)))}
    v = critic.v_fn(params, b["observations"], cond)
    q, qs = critic.q_fn(params, b["observations"], cond, b["actions"])
    assert v.shape == q.shape == (b["observations"].shape[0],)
    assert qs.shape == (critic.num_parallel, b["observations"].shape[0])
    # F_V takes a zeroed action by construction => v cannot move with the action.
    q2, _ = critic.q_fn(params, b["observations"], cond, -b["actions"])
    v2 = critic.v_fn(params, b["observations"], cond)
    assert np.allclose(np.asarray(v), np.asarray(v2))
    assert not np.allclose(np.asarray(q), np.asarray(q2))


def test_stage_loss_never_reads_boot():
    critic, params, targets = ISFBCritic.build(jnp.zeros((2, OBS)), jnp.zeros((2, ACT)),
                                               {**_cfg("flowbc"), "seed": 0})
    b = _batch()
    cond = {"z": jnp.zeros((b["observations"].shape[0], critic.z_dim))}
    loss, _ = critic.stage_loss("isfb", params, targets, b, cond, boot=None)
    assert jnp.isfinite(loss)


def test_config_composes_and_required_keys_are_complete():
    cfg = load_config("isfb", "flowbc", "cube_single")
    for key in REQUIRED_KEYS["isfb"]:
        node, present = cfg, True
        for part in key.split("."):
            assert isinstance(node, dict) and part in node, key
            node = node[part]
        assert node is not None, key
    assert cfg["dataset_class"] == "GCDataset"       # supplies batch['masks'] for goal_mask
