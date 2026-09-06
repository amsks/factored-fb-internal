"""`backward_input: hit_spec` on plain FB — the privileged-object-coordinate control.

This exists so the "with / without privileged coordinates" table has an FB row that is a
SINGLE-KNOB comparison. It is deliberately NOT feat/goal-conditioned-fb's V2
`fixed_b=cube_xyz`, which replaced B with a fixed non-learned map AND collapsed z_dim to 3
(cube-single 0.06 against plain FB's ~0.50). Two things moved there, and latent width alone
is known to be fatal (the GCIQL morph's -d5 arm floors at 9.6 against -d50's 90.7). Here B
stays a learned PhiMap at the full z_dim and only its input narrows.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from env import make_config
from impls.agents import Agent

OBS_DIM, ACT_DIM, B = 37, 5, 6      # cube-double's width, so both specs fit


def _built(critic, overrides=()):
    cfg = make_config(critic, "flowbc", "cube_single", list(overrides))
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.standard_normal((B, OBS_DIM)))
    act = jnp.asarray(rng.standard_normal((B, ACT_DIM)))
    return Agent.create(critic, "flowbc", 0, obs, act, cfg), obs, act, cfg


def test_b_consumes_only_the_hit_spec_columns():
    ag, _, _, cfg = _built("fb_oc")
    assert cfg["backward_input"] == "hit_spec"
    assert ag.net["backward"].params["Dense_0"]["kernel"].shape[0] == 3


def test_latent_width_is_untouched():
    """THE distinction from V2: z_dim stays 50, only B's INPUT narrows."""
    ag, _, _, cfg = _built("fb_oc")
    assert ag.critic.z_dim == 50 == int(cfg["z_dim"])
    # B still EMITS the full latent -- checked on the OUTPUT, not a guessed layer index
    rng = np.random.default_rng(3)
    out = ag.critic._B(ag.net["backward"].params,
                       jnp.asarray(rng.standard_normal((B, OBS_DIM))))
    assert out.shape == (B, 50)


def test_state_mode_is_untouched():
    ag, _, _, cfg = _built("fb")
    assert cfg.get("backward_input", "state") == "state"
    assert ag.net["backward"].params["Dense_0"]["kernel"].shape[0] == OBS_DIM


def test_forward_tower_still_sees_the_full_state():
    """The restriction is ONE-SIDED: F / the left encoder keep the whole observation."""
    ag, _, _, _ = _built("fb_oc")
    assert ag.net["left_encoder"].params["Dense_0"]["kernel"].shape[0] == OBS_DIM


def test_eval_goal_mapping_routes_through_the_choke_point():
    """map_eval_cond does z = project_z(B(goal)) on a RAW goal row; if it bypassed `_b_in`
    the eval z would be computed from a differently-shaped input than training used."""
    ag, obs, _, _ = _built("fb_oc")
    cr = ag.critic
    z = cr.map_eval_cond(ag.net, {"goal": obs})["z"]
    assert z.shape == (B, cr.z_dim)
    # columns outside the block must not move the eval z
    z2 = cr.map_eval_cond(ag.net, {"goal": obs.at[:, :19].add(5.0).at[:, 22:].add(-5.0)})["z"]
    assert np.allclose(np.asarray(z), np.asarray(z2)), "map_eval_cond bypassed `_b_in`"


def test_narrow_observations_are_refused():
    cfg = make_config("fb_oc", "flowbc", "cube_single", [])
    rng = np.random.default_rng(0)
    with pytest.raises(AssertionError, match="wide enough"):
        Agent.create("fb_oc", "flowbc", 0,
                     jnp.asarray(rng.standard_normal((B, 10))),
                     jnp.asarray(rng.standard_normal((B, ACT_DIM))), cfg)


def test_cube_double_spec_widens_b():
    ag, _, _, _ = _built("fb_oc", ["measure_hit_spec=19:22:0.4,28:31:0.4"])
    assert ag.net["backward"].params["Dense_0"]["kernel"].shape[0] == 6
