import jax, jax.numpy as jnp
from impls.utils.networks import Actor, Value, ActorVectorField


def test_gaussian_actor_dist():
    net = Actor(hidden_dims=(32, 32), action_dim=5, const_std=True, tanh_squash=True)
    p = net.init(jax.random.PRNGKey(0), jnp.ones((2, 12)))["params"]
    dist = net.apply({"params": p}, jnp.ones((2, 12)))
    assert dist.mode().shape == (2, 5)
    assert dist.log_prob(jnp.zeros((2, 5))).shape == (2,)


def test_value_ensemble():
    v = Value(hidden_dims=(32, 32), num_ensembles=2)
    p = v.init(jax.random.PRNGKey(0), jnp.ones((2, 12)), jnp.ones((2, 5)))["params"]
    out = v.apply({"params": p}, jnp.ones((2, 12)), jnp.ones((2, 5)))
    assert out.shape == (2, 2)  # [ensemble, batch]


def test_actor_vector_field():
    vf = ActorVectorField(hidden_dims=(32, 32), action_dim=5)
    p = vf.init(jax.random.PRNGKey(0), jnp.ones((2, 12)), jnp.ones((2, 5)), jnp.ones((2, 1)))["params"]
    out = vf.apply({"params": p}, jnp.ones((2, 12)), jnp.ones((2, 5)), jnp.ones((2, 1)))
    assert out.shape == (2, 5)
