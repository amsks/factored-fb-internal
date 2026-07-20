import os
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from flax.traverse_util import unflatten_dict
from impls.critics.gciql import GCIQLCritic
from impls.utils.networks import GCValue, Actor
from impls.actors.ddpgbc import DDPGBCActor
from impls.actors.awr import AWRActor

ATOL = 1e-10
_FX = np.load(os.path.join(os.path.dirname(__file__), "fixtures", "actor_reference.npz"))


def _params(group):
    pref = "param/" + group + "/"
    flat = {tuple(k[len(pref):].split("/")): jnp.asarray(_FX[k], jnp.float64)
            for k in _FX.files if k.startswith(pref)}
    return unflatten_dict(flat)


def _batch():
    return {k[len("batch/"):]: jnp.asarray(_FX[k], jnp.float64)
            for k in _FX.files if k.startswith("batch/")}


def _critic():
    hd = tuple(int(x) for x in _FX["meta/hidden_dims"])
    return GCIQLCritic(value=GCValue(hidden_dims=hd, layer_norm=True, ensemble=False),
                       critic=GCValue(hidden_dims=hd, layer_norm=True, ensemble=True),
                       discount=float(_FX["meta/discount"]),
                       expectile=float(_FX["meta/expectile"]), tau=0.005, obs_dim=8)


def _actor_net():
    hd = tuple(int(x) for x in _FX["meta/actor_hidden_dims"])
    # Must mirror vendored GCActor: state_dependent_std=False, const_std=True,
    # NO tanh squash, no layer norm.
    return Actor(hidden_dims=hd, action_dim=5, layer_norm=False,
                 tanh_squash=False, const_std=True, state_dependent_std=False)


def _all_params():
    return {"value": _params("value"), "critic": _params("critic"), "actor": _params("actor")}


def _cond(b):
    return {"value_goal": b["value_goals"], "goal": b["actor_goals"]}


def test_ddpgbc_actor_loss_matches_vendored():
    b, critic = _batch(), _critic()
    actor = DDPGBCActor(actor=_actor_net(), alpha=float(_FX["meta/alpha"]), const_std=True)
    _, info = actor.loss(_all_params(), None, b, critic, _cond(b), boot=None)
    np.testing.assert_allclose(np.asarray(info["actor_loss"]), _FX["exp/ddpgbc_actor_loss"], atol=ATOL)
    np.testing.assert_allclose(np.asarray(info["q_loss"]), _FX["exp/ddpgbc_q_loss"], atol=ATOL)
    np.testing.assert_allclose(np.asarray(info["bc_loss"]), _FX["exp/ddpgbc_bc_loss"], atol=ATOL)


def test_awr_actor_loss_matches_vendored():
    b, critic = _batch(), _critic()
    actor = AWRActor(actor=_actor_net(), alpha=float(_FX["meta/alpha"]))
    _, info = actor.loss(_all_params(), None, b, critic, _cond(b), boot=None)
    np.testing.assert_allclose(np.asarray(info["actor_loss"]), _FX["exp/awr_actor_loss"], atol=ATOL)
    np.testing.assert_allclose(np.asarray(info["adv"]), _FX["exp/awr_adv"], atol=ATOL)


def test_built_actors_do_not_tanh_squash():
    """Regression lock: build() must not re-introduce the tanh squash (see Task 1 Step 5)."""
    cfg = dict(seed=0, cvec_dim=8,
               actor=dict(hidden_dims=[64, 64], alpha=0.3, const_std=True, layer_norm=False))
    ddpg, _ = DDPGBCActor.build(jnp.zeros((2, 8)), jnp.zeros((2, 5)), cfg)
    awr, _ = AWRActor.build(jnp.zeros((2, 8)), jnp.zeros((2, 5)), cfg)
    assert ddpg.actor.tanh_squash is False
    assert awr.actor.tanh_squash is False
