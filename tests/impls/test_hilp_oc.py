"""`backward_input: hit_spec` on the HILP family — the object-coordinate phi input.

This is the ingredient the cube HP study identified: on the 500k seed-0 oc screen it is
what separates rldp_density_oc (1.00 on cube-single) from rldp_density (0.15-0.25). HILP
had no such hook, so `_phi_in` was added to impls/critics/hilp.py mirroring rldp.py's
`_b_in`.

The trap this file exists to catch: HILP applies phi from FOUR places -- `_phi`, `_phi2`,
`_feature_net`, and the `apply_phi` closure handed to the Hilbert value loss. That last
one used to call `self.hilp_phi1.apply(...)` directly. If it bypasses the choke point, the
value loss fits the metric on FULL STATES while the SF branch reads object coordinates:
two different representations under one name, and nothing fails.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from env import make_config
from impls.agents import Agent

# 37 = cube-double's real width (19 proprio + 2 x 9 block columns), so both the
# cube-single and the cube-double hit spec fit.
OBS_DIM, ACT_DIM, B = 37, 5, 6
SPEC = "19:22:0.4"          # the cube-single block: columns 19,20,21


def _built(critic, overrides=()):
    cfg = make_config(critic, "flowbc", "cube_single", list(overrides))
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.standard_normal((B, OBS_DIM)))
    act = jnp.asarray(rng.standard_normal((B, ACT_DIM)))
    return Agent.create(critic, "flowbc", 0, obs, act, cfg), obs, act, cfg


def _batch(obs, act):
    return {"observations": obs, "next_observations": obs[::-1], "actions": act,
            "hilp_future_observations": obs[::-1],
            "terminals": jnp.zeros(obs.shape[0])}


@pytest.mark.parametrize("critic", ["hilp_oc", "hilp_density_oc"])
def test_phi_consumes_only_the_hit_spec_columns(critic):
    """phi's first layer must be as wide as the BLOCK, not as the observation."""
    ag, _, _, cfg = _built(critic)
    assert cfg["backward_input"] == "hit_spec" and cfg["measure_hit_spec"] == SPEC
    kernel = ag.net["hilp_phi1"].params["Dense_0"]["kernel"]
    assert kernel.shape[0] == 3, (
        f"phi's input is {kernel.shape[0]} wide; the hit spec {SPEC} slices 3 columns")


@pytest.mark.parametrize("critic", ["hilp_oc", "hilp_density_oc"])
def test_value_loss_ignores_columns_outside_the_blocks(critic):
    """THE test: the Hilbert value loss must route through `_phi_in` too.

    Perturbing observation columns OUTSIDE 19:22 changes the phi loss if and only if the
    `apply_phi` closure bypasses the choke point.
    """
    ag, obs, act, _ = _built(critic)
    cr = ag.critic
    params = {g: ag.net[g].params for g in cr.GROUPS}
    target = {g: ag.target[g] for g in cr.TARGETS}
    z = jnp.asarray(np.random.default_rng(1).standard_normal((B, cr.z_dim)))
    cond, boot = {"z": z, "actor_goals": obs[::-1]}, {"next_action": act}

    def phi_loss(o):
        return cr.stage_loss("phi", params, target, _batch(o, act), cond, boot)[0]

    perturbed = obs.at[:, :19].add(7.0).at[:, 22:].add(-7.0)   # everything BUT the block
    assert float(phi_loss(obs)) == float(phi_loss(perturbed)), (
        "the Hilbert value loss saw columns outside the hit spec -- `apply_phi` is not "
        "routed through `_phi_in`")
    # ... and it is not simply constant in its input
    assert float(phi_loss(obs)) != float(phi_loss(obs.at[:, 19:22].add(1.0)))


def test_state_mode_is_untouched():
    """The default must reproduce critic/hilp.yaml byte-for-byte: full-width phi."""
    ag, _, _, cfg = _built("hilp", ["dataset_class=HILPFutureDataset"])
    assert cfg.get("backward_input", "state") == "state"
    assert ag.net["hilp_phi1"].params["Dense_0"]["kernel"].shape[0] == OBS_DIM


def test_narrow_observations_are_refused():
    """The spec reaches column 22; a 10-wide observation would be silently clipped."""
    cfg = make_config("hilp_density_oc", "flowbc", "cube_single", [])
    rng = np.random.default_rng(0)
    with pytest.raises(AssertionError, match="wide enough"):
        Agent.create("hilp_density_oc", "flowbc", 0,
                     jnp.asarray(rng.standard_normal((B, 10))),
                     jnp.asarray(rng.standard_normal((B, ACT_DIM))), cfg)


def test_cube_double_spec_widens_phi():
    """Two blocks, so six columns — the conjunction the domain yaml documents."""
    ag, _, _, _ = _built("hilp_density_oc",
                         ["measure_hit_spec=19:22:0.4,28:31:0.4"])
    assert ag.net["hilp_phi1"].params["Dense_0"]["kernel"].shape[0] == 6
