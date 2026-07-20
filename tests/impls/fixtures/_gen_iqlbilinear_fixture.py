"""THROWAWAY fixture generator — imports the FIRST-PARTY bridge research code
tools/wandb_mode_shim/gciql_flow.py to bake a float64 reference for IqlBilinearCritic's
objective. NOT run in CI.

VERIFICATION HONESTY: gciql_flow.py is first-party research code, NOT upstream OGBench.
This fixture is therefore a weaker bar than the vendored 1e-10 lock on the bilinear head
itself (tests/impls/test_gcbilinear.py). See the parity test's docstring.

Run with:
  JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    /tmp/factored-fb-venv-jax/bin/python tests/impls/fixtures/_gen_iqlbilinear_fixture.py

NOTE: sys.path.insert (not PYTHONPATH) — the repo-root utils.py shadows vendored utils/.
"""
import os, sys
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from flax.traverse_util import flatten_dict

sys.path.insert(0, os.path.join("third_party", "ogbench", "impls"))
sys.path.insert(0, os.path.join("tools", "wandb_mode_shim"))
from gciql_flow import GCIQLFlowAgent, get_config  # first-party bridge research code

B, OBS, ACT, D = 16, 8, 5, 8
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
    with config.unlocked():
        config.discrete = False
        config.encoder = None
        config.layer_norm = True
        config.value_hidden_dims = (64, 64)
        config.expectile = 0.9
        config.discount = 0.99
        config.actor_loss = "ddpgbc"
        # --- the cell under test: plain bilinear head, learned psi, ortho on, no sphere ---
        config.value_arch = "bilinear"
        config.value_latent_dim = D
        config.psi_base = "learned"
        config.project_z = False      # see the plan's Known Risks (1): our project_z is
                                      # SYMMETRIC (audit S3 fix), the reference's is not, so
                                      # only project_z=False is numerically comparable.
        config.double_q = True
        config.actor_cond = "goal"
        config.actor_kind = "ddpgbc"
        config.iql_ortho_coef = 1.0
        config.action_dim = ACT

    ex_obs = jnp.asarray(rng.standard_normal((B, OBS)))
    ex_act = jnp.asarray(np.tanh(rng.standard_normal((B, ACT))))
    agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, config)

    batch = {k: jnp.asarray(v) for k, v in _batch().items()}
    params = agent.network.params
    value_loss, _ = agent.value_loss(batch, params)
    critic_loss, _ = agent.critic_loss(batch, params)
    ortho_loss, _ = agent.ortho_loss(batch, params)
    v = agent.network.select("value")(batch["observations"], batch["actor_goals"])
    q1, q2 = agent.network.select("critic")(
        batch["observations"], batch["actor_goals"], batch["actions"])
    _, _, psi = agent.network.select("critic")(
        batch["observations"], batch["value_goals"], batch["actions"], info=True)

    out = {}
    # our IqlBilinearCritic param layout: "value" <- modules_value, "critic" <- modules_critic
    for our, vend in [("value", "modules_value"), ("critic", "modules_critic")]:
        for k, arr in flatten_dict(params[vend]).items():
            out["param/" + our + "/" + "/".join(k)] = np.asarray(arr)
    for k, arr in batch.items():
        out["batch/" + k] = np.asarray(arr)
    # PER-TERM dumps: a mis-scaled term must not be maskable by a compensating one.
    out["exp/value_loss"] = np.asarray(value_loss)
    out["exp/critic_loss"] = np.asarray(critic_loss)
    out["exp/ortho_loss"] = np.asarray(ortho_loss)
    out["exp/v"] = np.asarray(v)
    out["exp/q1"] = np.asarray(q1)
    out["exp/q2"] = np.asarray(q2)
    out["exp/psi"] = np.asarray(psi)
    out["meta/hidden_dims"] = np.asarray([64, 64])
    out["meta/latent_dim"] = np.asarray(D)
    out["meta/expectile"] = np.asarray(0.9)
    out["meta/discount"] = np.asarray(0.99)
    out["meta/iql_ortho_coef"] = np.asarray(1.0)
    path = os.path.join("tests", "impls", "fixtures", "iql_bilinear_reference.npz")
    np.savez(path, **out)
    print("wrote", path, "with", len(out), "arrays")


if __name__ == "__main__":
    main()
