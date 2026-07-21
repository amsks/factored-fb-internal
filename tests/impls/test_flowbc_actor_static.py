import jax.numpy as jnp, numpy as np
from impls.actors.flowbc import FlowBCActor
from impls.utils import torch_to_flax as t2f
from tests.impls.test_fb_critic_static import _critic, _params_targets  # reuse critic wiring

ATOL = 1e-10
A, H = 5, 32


def _actor():
    from impls.utils.networks import NoiseConditionedActor, FlowVectorField
    return FlowBCActor(
        actor=NoiseConditionedActor(action_dim=A, hidden_dim=H, hidden_layers=2, embedding_layers=2),
        actor_vf=FlowVectorField(action_dim=A, hidden_dim=H, hidden_layers=4),
        flow_steps=10, bc_coeff=3.0, actor_pessimism_penalty=0.0,
    )


def test_flowbc_static_loss_matches_fixture(fb_fixture):
    fx = fb_fixture
    critic = _critic()
    online, _ = _params_targets(fx)
    online["actor"] = t2f.load_flow_actor(fx)
    online["actor_vf"] = t2f.load_flow_vf(fx)
    actor = _actor()
    batch = {k: jnp.asarray(fx["in__" + k], jnp.float64)
             for k in ("observations", "actions", "next_observations")}
    cond = {"z": jnp.asarray(fx["in__z"], jnp.float64)}
    boot = {"flow_x0": jnp.asarray(fx["in__flow_x0"], jnp.float64),
            "flow_t": jnp.asarray(fx["in__flow_t"], jnp.float64),
            "actor_noise": jnp.asarray(fx["in__actor_noise"], jnp.float64)}
    loss, info = actor.loss(online, None, batch, critic, cond, boot)
    np.testing.assert_allclose(float(info["bc_flow_loss"]), float(fx["out__bc_flow_loss"]), atol=ATOL)
    np.testing.assert_allclose(float(info["bc_error"]), float(fx["out__bc_error"]), atol=ATOL)
    np.testing.assert_allclose(float(info["q"]), float(fx["out__q"]), atol=ATOL)
    np.testing.assert_allclose(float(info["actor_loss"]), float(fx["out__actor_loss"]), atol=ATOL)
