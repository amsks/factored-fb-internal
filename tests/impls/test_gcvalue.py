import jax, jax.numpy as jnp
from impls.utils.networks import GCValue


def test_value_shape_and_goal_conditioning():
    v = GCValue(hidden_dims=(32, 32), layer_norm=True, ensemble=False)
    obs = jnp.ones((4, 8)); g = jnp.ones((4, 8))
    p = v.init(jax.random.PRNGKey(0), obs, g)["params"]
    out = v.apply({"params": p}, obs, g)
    assert out.shape == (4,)
    # different goal -> different value
    out2 = v.apply({"params": p}, obs, jnp.zeros((4, 8)))
    assert not jnp.allclose(out, out2)


def test_critic_twin_q_shape():
    c = GCValue(hidden_dims=(32, 32), layer_norm=True, ensemble=True)
    obs = jnp.ones((4, 8)); g = jnp.ones((4, 8)); a = jnp.ones((4, 5))
    p = c.init(jax.random.PRNGKey(0), obs, g, a)["params"]
    out = c.apply({"params": p}, obs, g, a)
    assert out.shape == (2, 4)  # [ensemble, batch]
    assert "value_net" in p
