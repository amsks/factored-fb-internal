"""THROWAWAY fixture generator — imports the vendored OGBench GCBilinearValue to bake a
float64 reference for the in-house port. NOT run in CI. Run with:
  JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    /tmp/factored-fb-venv-jax/bin/python tests/impls/fixtures/_gen_bilinear_fixture.py

NOTE: sys.path.insert (not PYTHONPATH) — the repo-root utils.py shadows vendored utils/.
Vendored OGBench is jax, so this runs under the jax venv.
"""
import os, sys
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from flax.traverse_util import flatten_dict

sys.path.insert(0, os.path.join("third_party", "ogbench", "impls"))
from utils.networks import GCBilinearValue  # vendored

B, OBS, ACT, GOAL = 16, 8, 5, 8
HD = (64, 64)
# (name, latent_dim, ensemble) — two configs so /sqrt(d) is exercised, not hidden.
CONFIGS = [("c0", 8, True), ("c1", 32, False)]
rng = np.random.default_rng(0)


def main():
    obs = jnp.asarray(rng.standard_normal((B, OBS)))
    goals = jnp.asarray(rng.standard_normal((B, GOAL)))
    actions = jnp.asarray(np.tanh(rng.standard_normal((B, ACT))))

    out = {
        "batch/observations": np.asarray(obs),
        "batch/goals": np.asarray(goals),
        "batch/actions": np.asarray(actions),
        "meta/hidden_dims": np.asarray(HD),
    }

    for name, d, ens in CONFIGS:
        net = GCBilinearValue(hidden_dims=HD, latent_dim=d, layer_norm=True, ensemble=ens)
        out["meta/" + name + "/latent_dim"] = np.asarray(d)
        out["meta/" + name + "/ensemble"] = np.asarray(ens)

        # The two call paths have different phi input widths (obs vs obs+act), so each
        # needs its own init. tag="value" -> actions=None; tag="critic" -> actions given.
        for tag, act in [("value", None), ("critic", actions)]:
            params = net.init(jr.PRNGKey(1234), obs, goals, act)["params"]
            for k, arr in flatten_dict(params).items():
                out[f"param/{name}/{tag}/" + "/".join(k)] = np.asarray(arr)
            v, phi, psi = net.apply({"params": params}, obs, goals, act, info=True)
            out[f"exp/{name}/{tag}/v"] = np.asarray(v)
            out[f"exp/{name}/{tag}/phi"] = np.asarray(phi)
            out[f"exp/{name}/{tag}/psi"] = np.asarray(psi)
            # info=False return path must equal the info=True v.
            out[f"exp/{name}/{tag}/v_noinfo"] = np.asarray(net.apply({"params": params}, obs, goals, act))

    path = os.path.join("tests", "impls", "fixtures", "bilinear_reference.npz")
    np.savez(path, **out)
    print("wrote", path, "with", len(out), "arrays")


if __name__ == "__main__":
    main()
