"""THROWAWAY fixture generator — imports the FIRST-PARTY bridge research code
tools/wandb_mode_shim/gciql_flow.py to bake a float64 reference for FbForwardBilinearCritic's
z-INDEXED objective (flat phi(s,a,z)). NOT run in CI.

VERIFICATION HONESTY: gciql_flow.py is first-party research code, NOT upstream OGBench, and
the z task-index (`fb_forward_z`) it implements has NO external ground truth (not vendored,
not torch) — it is the Phase 1 analogue of Phase 0's FQL actor. This fixture is therefore a
TRANSCRIPTION lock: it proves our GCFbForwardValue + objective reproduce the bridge's own
numbers, NOT that the z-indexing is "correct" against any upstream reference. What it does
buy: the flat forward math (GCFBValue), the z-indexed IQL value/critic backup (B1), and the
measure critic-leg DEFAULT path (C1) — the last being FB's own successor-measure TD, whose
math traces to torch agents/fb/agent.py.

The measure loss bootstraps a+ ~ pi(s+); to keep the lock on the measure MATH (not the
actor's sampling) we use a ddpgbc const_std actor (dist.mode() is DETERMINISTIC), dump the
exact bootstrap action a_next, and the parity test injects it via boot['next_action'].

Run with:
  JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    /tmp/factored-fb-venv-jax/bin/python tests/impls/fixtures/_gen_fbforward_fixture.py

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

B, OBS, ACT, D = 16, 8, 5, 8      # ensemble=2 critic, D != 1 -> scale factors are exercised
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
        config.actor_kind = "ddpgbc"
        config.const_std = True            # deterministic dist.mode() bootstrap
        # --- the cell under test: z-indexed flat forward, learned psi, no sphere ---
        config.value_arch = "bilinear"
        config.value_latent_dim = D
        config.psi_base = "learned"
        config.fb_forward_z = True         # phi(s,a,z), z=sg(psi(g)) -> the z-indexed family
        config.project_z = False
        config.double_q = True
        config.actor_cond = "goal"
        # measure knobs: DEFAULT path (critic legs, rho/full, no scale-norm), ortho on.
        config.measure_coef = 1.0
        config.measure_legs = "critic"
        config.measure_scale_norm = False
        config.measure_stopgrad_phi = False
        config.measure_ortho_coef = 1.0
        config.iql_ortho_coef = 1.0
        config.action_dim = ACT

    ex_obs = jnp.asarray(rng.standard_normal((B, OBS)))
    ex_act = jnp.asarray(np.tanh(rng.standard_normal((B, ACT))))
    agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, config)

    batch = {k: jnp.asarray(v) for k, v in _batch().items()}
    params = agent.network.params

    value_loss, _ = agent.value_loss(batch, params)
    critic_loss, _ = agent.critic_loss(batch, params)
    m_loss, m_info = agent.measure_loss(batch, params, jax.random.PRNGKey(0))

    # The reference ortho_loss() (gciql_flow.py L1156) does NOT thread z_goals, so it crashes
    # under fb_forward_z (GCFBValue requires z_goals). Reproduce its EXACT formula here with
    # z_goals supplied — the z arg does not affect psi(vgoals), which is all ortho reads.
    _, _, psi_o = agent.network.select("critic")(
        batch["observations"], batch["value_goals"], batch["actions"], batch["value_goals"],
        info=True)
    _off = 1.0 - jnp.eye(B)
    _cov = jnp.einsum("hid,hjd->hij", psi_o, psi_o)
    _o_off = 0.5 * jnp.sum((_cov * _off[None]) ** 2, axis=(1, 2)) / jnp.sum(_off)
    _o_diag = -jnp.mean(jnp.diagonal(_cov, axis1=1, axis2=2), axis=1)
    ortho_loss = jnp.mean(_o_off + _o_diag)

    # the deterministic bootstrap the measure loss used internally (actor_cond='goal' -> vgoals)
    a_next = jnp.clip(agent.network.select("actor")(
        batch["next_observations"], batch["value_goals"]).mode(), -1, 1)

    # scalar readouts (z-indexed): Q(s,a,g)=<F(s,a,psi(g)),psi(g)>/sqrt(d), V(s,g) z-blind.
    v = agent.network.select("value")(batch["observations"], batch["actor_goals"])
    q1, q2 = agent.network.select("critic")(
        batch["observations"], batch["actor_goals"], batch["actions"], batch["actor_goals"])

    out = {}
    for our, vend in [("value", "modules_value"), ("critic", "modules_critic")]:
        for k, arr in flatten_dict(params[vend]).items():
            out["param/" + our + "/" + "/".join(k)] = np.asarray(arr)
    for k, arr in batch.items():
        out["batch/" + k] = np.asarray(arr)
    out["batch/a_next"] = np.asarray(a_next)
    # PER-TERM dumps: a mis-scaled term must not be maskable by a compensating one.
    out["exp/value_loss"] = np.asarray(value_loss)
    out["exp/critic_loss"] = np.asarray(critic_loss)
    out["exp/ortho_loss"] = np.asarray(ortho_loss)
    out["exp/measure_loss"] = np.asarray(m_loss)
    out["exp/measure_offdiag"] = np.asarray(m_info["measure_offdiag"])
    out["exp/measure_diag"] = np.asarray(m_info["measure_diag"])
    out["exp/measure_orth"] = np.asarray(m_info["measure_orth"])
    out["exp/v"] = np.asarray(v)
    out["exp/q1"] = np.asarray(q1)
    out["exp/q2"] = np.asarray(q2)
    out["meta/hidden_dims"] = np.asarray([64, 64])
    out["meta/latent_dim"] = np.asarray(D)
    out["meta/expectile"] = np.asarray(0.9)
    out["meta/discount"] = np.asarray(0.99)
    out["meta/measure_ortho_coef"] = np.asarray(1.0)
    out["meta/iql_ortho_coef"] = np.asarray(1.0)
    path = os.path.join("tests", "impls", "fixtures", "fb_forward_bilinear_reference.npz")
    np.savez(path, **out)
    print("wrote", path, "with", len(out), "arrays")


if __name__ == "__main__":
    main()
