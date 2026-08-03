import os
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from flax.traverse_util import unflatten_dict
from impls.critics.gciql import GCIQLCritic
from impls.utils.networks import GCValue

ATOL = 1e-10
_FX = np.load(os.path.join(os.path.dirname(__file__), "fixtures", "gciql_reference.npz"))


def _params(group):
    pref = "param/" + group + "/"
    flat = {tuple(k[len(pref):].split("/")): jnp.asarray(_FX[k], jnp.float64)
            for k in _FX.files if k.startswith(pref)}
    return unflatten_dict(flat)


def _batch():
    return {k[len("batch/"):]: jnp.asarray(_FX[k], jnp.float64) for k in _FX.files if k.startswith("batch/")}


def _critic():
    hd = tuple(int(x) for x in _FX["meta/hidden_dims"])
    return GCIQLCritic(value=GCValue(hidden_dims=hd, layer_norm=True, ensemble=False),
                       critic=GCValue(hidden_dims=hd, layer_norm=True, ensemble=True),
                       discount=float(_FX["meta/discount"]), expectile=float(_FX["meta/expectile"]),
                       tau=0.005, obs_dim=8, q_bound=0.0)


def test_value_and_critic_loss_match():
    critic = _critic()
    params = {"value": _params("value"), "critic": _params("critic")}
    target = {"critic": _params("critic")}   # fixture target_critic == critic at init
    b = _batch()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    _, info = critic.stage_loss("iql", params, target, b, cond, boot=None)
    np.testing.assert_allclose(np.asarray(info["value_loss"]), _FX["exp/value_loss"], atol=ATOL)
    np.testing.assert_allclose(np.asarray(info["critic_loss"]), _FX["exp/critic_loss"], atol=ATOL)


def test_v_and_q_fn_match():
    critic = _critic()
    params = {"value": _params("value"), "critic": _params("critic")}
    b = _batch()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    v = critic.v_fn(params, b["observations"], cond)
    qmin, qs = critic.q_fn(params, b["observations"], cond, b["actions"])
    np.testing.assert_allclose(np.asarray(v), _FX["exp/v"], atol=ATOL)
    np.testing.assert_allclose(np.asarray(qs[0]), _FX["exp/q1"], atol=ATOL)
    np.testing.assert_allclose(np.asarray(qs[1]), _FX["exp/q2"], atol=ATOL)
