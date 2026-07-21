"""THROWAWAY fixture generator — imports the vendored OGBench GCIQL to bake float64
reference actor losses for ddpgbc and awr. NOT run in CI; deleted at Phase 6. Run with:
  JAX_PLATFORMS=cpu PYTHONPATH=third_party/ogbench/impls \
    /tmp/factored-fb-venv-jax/bin/python tests/impls/fixtures/_gen_actor_fixture.py
"""
import os, sys
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from flax.traverse_util import flatten_dict

sys.path.insert(0, os.path.join("third_party", "ogbench", "impls"))
from agents.gciql import GCIQLAgent, get_config  # vendored

B, OBS, ACT = 16, 8, 5
HID = (64, 64)
ALPHA = 0.3
rng = np.random.default_rng(0)


def _batch():
    g = rng.standard_normal((B, OBS))
    return dict(
        observations=rng.standard_normal((B, OBS)),
        actions=np.tanh(rng.standard_normal((B, ACT))) * 0.9,  # keep off the tanh boundary
        next_observations=rng.standard_normal((B, OBS)),
        value_goals=g, actor_goals=g,
        rewards=(rng.integers(0, 2, size=B) - 1.0),
        masks=rng.integers(0, 2, size=B).astype(float),
    )


def _agent(actor_loss, ex_obs, ex_act):
    config = dict(get_config())
    config.update(discrete=False, encoder=None, layer_norm=True,
                  value_hidden_dims=HID, actor_hidden_dims=HID,
                  expectile=0.9, discount=0.99, alpha=ALPHA,
                  const_std=True, actor_loss=actor_loss)
    return GCIQLAgent.create(0, ex_obs, ex_act, config)


def main():
    ex_obs = jnp.asarray(rng.standard_normal((B, OBS)))
    ex_act = jnp.asarray(np.tanh(rng.standard_normal((B, ACT))))
    batch = {k: jnp.asarray(v) for k, v in _batch().items()}

    out = {}
    for k, arr in batch.items():
        out["batch/" + k] = np.asarray(arr)

    # ddpgbc agent supplies the shared params (value/critic/actor trees are identical
    # across the two actor_loss settings — only the loss function differs).
    ag_d = _agent("ddpgbc", ex_obs, ex_act)
    params = ag_d.network.params
    for our, vend in [("value", "modules_value"), ("critic", "modules_critic"),
                      ("actor", "modules_actor")]:
        for k, arr in flatten_dict(params[vend]).items():
            out["param/" + our + "/" + "/".join(k)] = np.asarray(arr)

    d_loss, d_info = ag_d.actor_loss(batch, params, rng=jax.random.PRNGKey(0))
    out["exp/ddpgbc_actor_loss"] = np.asarray(d_loss)
    out["exp/ddpgbc_q_loss"] = np.asarray(d_info["q_loss"])
    out["exp/ddpgbc_bc_loss"] = np.asarray(d_info["bc_loss"])

    ag_a = _agent("awr", ex_obs, ex_act)
    a_loss, a_info = ag_a.actor_loss(batch, params, rng=jax.random.PRNGKey(0))
    out["exp/awr_actor_loss"] = np.asarray(a_loss)
    out["exp/awr_adv"] = np.asarray(a_info["adv"])

    out["meta/hidden_dims"] = np.asarray(HID)
    out["meta/actor_hidden_dims"] = np.asarray(HID)
    out["meta/alpha"] = np.asarray(ALPHA)
    out["meta/expectile"] = np.asarray(0.9)
    out["meta/discount"] = np.asarray(0.99)

    path = os.path.join("tests", "impls", "fixtures", "actor_reference.npz")
    np.savez(path, **out)
    print("wrote", path, "with", len(out), "arrays")


if __name__ == "__main__":
    main()
