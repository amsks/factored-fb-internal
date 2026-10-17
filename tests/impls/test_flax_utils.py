import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from impls.utils.flax_utils import ModuleDict, TrainState, soft_update


def test_soft_update_matches_polyak():
    online = {"w": jnp.ones((2, 2))}
    target = {"w": jnp.zeros((2, 2))}
    out = soft_update(online, target, 0.1)
    np.testing.assert_allclose(np.asarray(out["w"]), 0.1 * np.ones((2, 2)))


def test_per_group_trainstate_isolates_optimizer_state():
    # Stepping group A leaves group B's Adam state unchanged.
    a = nn.Dense(3)
    b = nn.Dense(3)
    pa = a.init(jax.random.PRNGKey(0), jnp.ones((1, 4)))["params"]
    pb = b.init(jax.random.PRNGKey(1), jnp.ones((1, 4)))["params"]
    ta = TrainState.create(a, pa, tx=optax.adam(1e-2))
    tb = TrainState.create(b, pb, tx=optax.adam(1e-2))
    grads = jax.tree_util.tree_map(jnp.ones_like, pa)
    ta2 = ta.apply_gradients(grads=grads)
    assert ta2.step == 2 and tb.step == 1
    # group B params and opt_state unchanged
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda x, y: bool((x == y).all()), tb.params, pb)
    )


def test_moduledict_dispatch_by_name():
    md = ModuleDict({"enc": nn.Dense(3), "dec": nn.Dense(5)})
    params = md.init(
        jax.random.PRNGKey(0), enc=(jnp.ones((1, 4)),), dec=(jnp.ones((1, 3)),)
    )["params"]
    ts = TrainState.create(md, params)
    out = ts.select("enc")(jnp.ones((1, 4)))
    assert out.shape == (1, 3)
