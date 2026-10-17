"""`backward_input: state` on goal-conditioned Density-FB: B reads the full observation
instead of the hit-spec columns. Flipping it must change B's input and leave the hit-kernel
reward's operand (the raw hit-spec columns) alone.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from env import make_config
from impls.agents import Agent

OBS_DIM, ACT_DIM, B = 28, 5, 6      # cube-single's real width


def _built(critic, overrides=()):
    cfg = make_config(critic, "flowbc", "cube_single", list(overrides))
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.standard_normal((B, OBS_DIM)))
    act = jnp.asarray(rng.standard_normal((B, ACT_DIM)))
    return Agent.create(critic, "flowbc", 0, obs, act, cfg), obs, act, cfg


def test_state_mode_widens_b_to_the_full_observation():
    ag, _, _, cfg = _built("density_fb_state")
    assert cfg["backward_input"] == "state"
    assert ag.net["density_b"].params["Dense_0"]["kernel"].shape[0] == OBS_DIM


def test_default_is_unchanged():
    """The parent must stay on object coordinates -- 3 columns, not 28."""
    ag, _, _, cfg = _built("density_fb")
    assert cfg.get("backward_input", "hit_spec") == "hit_spec"
    assert ag.net["density_b"].params["Dense_0"]["kernel"].shape[0] == 3


def test_the_knob_moves_only_b_not_the_kernel_reward():
    """THE test. `u_raw` feeds the hit kernel and must be identical under both settings;
    only `u_sc` (B's input) may change."""
    from impls.critics.density_fb import _raw_block_cols
    rng = np.random.default_rng(1)
    nobs = jnp.asarray(rng.standard_normal((B, OBS_DIM)))
    out = {}
    for critic in ("density_fb", "density_fb_state"):
        ag, _, _, _ = _built(critic)
        cr = ag.critic
        out[critic] = (
            np.asarray(_raw_block_cols(nobs, cr.blocks)),          # kernel operand
            np.asarray(cr._b_query(nobs)),                         # B's input
        )
    # kernel operand identical
    assert np.array_equal(out["density_fb"][0], out["density_fb_state"][0])
    # B's input differs, and in the expected direction
    assert out["density_fb"][1].shape[1] == 3
    assert out["density_fb_state"][1].shape[1] == OBS_DIM
    assert np.allclose(out["density_fb_state"][1], np.asarray(nobs))


def test_state_plus_reward_index_is_refused():
    """Reward inference stores its evaluation pool as hit-spec columns, so it runs only with
    B on the hit-spec columns."""
    cfg = make_config("density_fb_state", "flowbc", "cube_single",
                      ["infer_z=true", "eval_pool_size=4"])
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="backward_input"):
        Agent.create("density_fb_state", "flowbc", 0,
                     jnp.asarray(rng.standard_normal((B, OBS_DIM))),
                     jnp.asarray(rng.standard_normal((B, ACT_DIM))), cfg)
