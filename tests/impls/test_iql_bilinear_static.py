"""FIRST-PARTY reference parity for IqlBilinearCritic's objective.

VERIFICATION HONESTY: the bilinear HEAD is locked at 1e-10 against VENDORED OGBench
(test_gcbilinear.py, Axis 2). This test is different and WEAKER: the bilinear-head + IQL
objective + FB-orthonormality combination exists only in the first-party bridge research
code tools/wandb_mode_shim/gciql_flow.py, NOT in upstream OGBench. So this fixture is a
FIRST-PARTY reference. It does NOT establish upstream parity; it establishes that our
transcription reproduces the bridge's own numbers.

Baked at d=8, ensemble critic, psi_base='learned', project_z=False, double_q=True,
iql_ortho_coef=1.0, expectile=0.9. project_z=False on purpose: our project_z is symmetric
across both nets (audit S3 fix) while the reference's is critic-only, so only the
unsphered path is numerically comparable (see the plan's Known Risks).

Per-term (value_loss, critic_loss, ortho_loss dumped separately) so a mis-scaled term
cannot be masked by a compensating one.
"""
import os
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from flax.traverse_util import unflatten_dict

from impls.critics.iql_bilinear import IqlBilinearCritic
from impls.utils.networks import GCBilinearPsiValue

ATOL = 1e-10
_FX = np.load(os.path.join(os.path.dirname(__file__), "fixtures", "iql_bilinear_reference.npz"))


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
    d = int(_FX["meta/latent_dim"])
    value = GCBilinearPsiValue(hidden_dims=hd, latent_dim=d, layer_norm=True, ensemble=False,
                               psi_base="learned", project_z=False)
    critic = GCBilinearPsiValue(hidden_dims=hd, latent_dim=d, layer_norm=True, ensemble=True,
                                psi_base="learned", project_z=False)
    return IqlBilinearCritic(
        value=value, critic=critic, discount=float(_FX["meta/discount"]),
        expectile=float(_FX["meta/expectile"]), tau=0.005, obs_dim=8, latent_dim=d,
        psi_base="learned", actor_cond="goal", actor_z_from_critic=False,
        actor_z_infer=False, iql_ortho_coef=float(_FX["meta/iql_ortho_coef"]),
        project_z=False, double_q=True)


def test_per_term_objective_matches_first_party_reference():
    critic = _critic()
    params = {"value": _params("value"), "critic": _params("critic")}
    target = {"critic": _params("critic")}   # fixture target_critic == critic at init
    b = _batch()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    _, info = critic.stage_loss("iql", params, target, b, cond, boot=None)
    np.testing.assert_allclose(np.asarray(info["value_loss"]), _FX["exp/value_loss"], atol=ATOL)
    np.testing.assert_allclose(np.asarray(info["critic_loss"]), _FX["exp/critic_loss"], atol=ATOL)
    np.testing.assert_allclose(np.asarray(info["ortho_loss"]), _FX["exp/ortho_loss"], atol=ATOL)


def test_v_and_q_fn_match_first_party_reference():
    critic = _critic()
    params = {"value": _params("value"), "critic": _params("critic")}
    b = _batch()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    v = critic.v_fn(params, b["observations"], cond)
    _, qs = critic.q_fn(params, b["observations"], cond, b["actions"])
    np.testing.assert_allclose(np.asarray(v), _FX["exp/v"], atol=ATOL)
    np.testing.assert_allclose(np.asarray(qs[0]), _FX["exp/q1"], atol=ATOL)
    np.testing.assert_allclose(np.asarray(qs[1]), _FX["exp/q2"], atol=ATOL)
