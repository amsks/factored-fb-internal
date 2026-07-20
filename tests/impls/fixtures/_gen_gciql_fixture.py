"""THROWAWAY fixture generator — imports the vendored OGBench GCIQL to bake a float64
reference. NOT run in CI; deleted at PR8. Run with:
  JAX_PLATFORMS=cpu PYTHONPATH=third_party/ogbench/impls \
    /tmp/factored-fb-venv-jax/bin/python tests/impls/fixtures/_gen_gciql_fixture.py
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
rng = np.random.default_rng(0)


def _batch():
    g = rng.standard_normal((B, OBS))
    return dict(
        observations=rng.standard_normal((B, OBS)),
        actions=np.tanh(rng.standard_normal((B, ACT))),
        next_observations=rng.standard_normal((B, OBS)),
        value_goals=g, actor_goals=g,
        rewards=(rng.integers(0, 2, size=B) - 1.0),
        masks=rng.integers(0, 2, size=B).astype(float),
    )


def main():
    config = get_config()
    config = dict(config)
    config.update(discrete=False, encoder=None, layer_norm=True,
                  value_hidden_dims=(64, 64), expectile=0.9, discount=0.99, actor_loss="ddpgbc")
    ex_obs = jnp.asarray(rng.standard_normal((B, OBS)))
    ex_act = jnp.asarray(np.tanh(rng.standard_normal((B, ACT))))
    agent = GCIQLAgent.create(0, ex_obs, ex_act, config)

    batch = {k: jnp.asarray(v) for k, v in _batch().items()}
    params = agent.network.params
    value_loss, _ = agent.value_loss(batch, params)
    critic_loss, _ = agent.critic_loss(batch, params)
    v = agent.network.select("value")(batch["observations"], batch["actor_goals"])
    q1, q2 = agent.network.select("critic")(batch["observations"], batch["actor_goals"], batch["actions"])

    out = {}
    # our GCIQLCritic param layout: "value" <- modules_value, "critic" <- modules_critic
    for our, vend in [("value", "modules_value"), ("critic", "modules_critic")]:
        for k, arr in flatten_dict(params[vend]).items():
            out["param/" + our + "/" + "/".join(k)] = np.asarray(arr)
    for k, arr in batch.items():
        out["batch/" + k] = np.asarray(arr)
    out["exp/value_loss"] = np.asarray(value_loss)
    out["exp/critic_loss"] = np.asarray(critic_loss)
    out["exp/v"] = np.asarray(v)
    out["exp/q1"] = np.asarray(q1)
    out["exp/q2"] = np.asarray(q2)
    out["meta/hidden_dims"] = np.asarray([64, 64])
    out["meta/expectile"] = np.asarray(0.9)
    out["meta/discount"] = np.asarray(0.99)
    path = os.path.join("tests", "impls", "fixtures", "gciql_reference.npz")
    np.savez(path, **out)
    print("wrote", path, "with", len(out), "arrays")


if __name__ == "__main__":
    main()
