"""FIRST-PARTY reference parity for FbForwardBilinearCritic's z-INDEXED objective.

VERIFICATION HONESTY (read the module docstring of impls/critics/fb_forward_bilinear.py):
the z task-index `fb_forward_z` has NO external ground truth — not vendored, not torch. It is
the Phase 1 analogue of Phase 0's FQL actor. This fixture is a TRANSCRIPTION lock against the
FIRST-PARTY bridge (tools/wandb_mode_shim/gciql_flow.py), NOT an upstream-parity lock: it
proves our GCFbForwardValue(flat) + objective reproduce the bridge's own numbers. What that
buys: the flat forward math (GCFBValue), the z-indexed IQL value/critic backup (B1), and the
measure critic-leg DEFAULT path (C1). The measure math itself traces to torch
agents/fb/agent.py (FB's successor-measure TD); the z-in-IQL-backup does not, and is locked
here only as a faithful transcription.

Per-term (value/critic/ortho and measure offdiag/diag/orth dumped separately) so a mis-scaled
term cannot be masked by a compensating one. The measure bootstrap a+ is HELD FIXED (dumped
from the reference's deterministic ddpgbc mode()) and injected via boot['next_action'], so the
lock isolates the measure MATH from the actor's sampling.

Baked at d=8, ensemble critic, phi_module='flat', project_z=False, measure_legs='critic',
scale_norm=False, test rho/full, ortho coefs=1.0. project_z=False on purpose (our project_z
is symmetric across both nets, audit S3, while the reference's is critic-only).
"""
import os
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from flax.traverse_util import unflatten_dict

from impls.critics.fb_forward_bilinear import FbForwardBilinearCritic
from impls.utils.networks import GCBilinearPsiValue, GCFbForwardValue

ATOL = 1e-10
_FX = np.load(os.path.join(os.path.dirname(__file__), "fixtures",
                           "fb_forward_bilinear_reference.npz"))


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
    critic = GCFbForwardValue(hidden_dims=hd, latent_dim=d, layer_norm=True, ensemble=True,
                              phi_module="flat", project_z=False)
    return FbForwardBilinearCritic(
        value=value, critic=critic, phi_module="flat", objective="mix",
        discount=float(_FX["meta/discount"]), expectile=float(_FX["meta/expectile"]),
        tau=0.005, obs_dim=8, latent_dim=d, action_dim=5, project_z=False, double_q=True,
        actor_cond="goal", actor_z_from_critic=False, actor_z_infer=False,
        measure_coef=1.0, measure_legs="critic", measure_scale_norm=False,
        measure_stopgrad_phi=False, measure_ortho_coef=float(_FX["meta/measure_ortho_coef"]),
        measure_test_states="rho", measure_test_rank="full",
        iql_ortho_coef=float(_FX["meta/iql_ortho_coef"]),
        measure_insample="scalar_gate_hard", alpha_n=1.0, alpha_m=1.0, alpha_v=1.0,
        awr_beta=3.0, awr_wmax=100.0, awr_normalize_adv=True, fb_pessimism_penalty=0.0,
        goal_mask=True)


def _pt():
    params = {"value": _params("value"), "critic": _params("critic")}
    target = {"critic": _params("critic")}   # fixture target_critic == critic at init
    b = _batch()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    boot = {"next_action": b["a_next"]}
    return _critic(), params, target, b, cond, boot


def test_iql_value_and_critic_losses_match():
    critic, params, target, b, cond, _ = _pt()
    _, info = critic._iql_losses(params, target, b, cond)
    np.testing.assert_allclose(np.asarray(info["value_loss"]), _FX["exp/value_loss"], atol=ATOL)
    np.testing.assert_allclose(np.asarray(info["critic_loss"]), _FX["exp/critic_loss"], atol=ATOL)


def test_ortho_matches():
    critic, params, _, _, cond, _ = _pt()
    orth = critic._ortho_only(params["critic"], cond["value_goal"])
    np.testing.assert_allclose(np.asarray(orth), _FX["exp/ortho_loss"], atol=ATOL)


def test_measure_default_path_matches_per_term():
    critic, params, target, b, cond, boot = _pt()
    _, info = critic._measure_loss(params, target, b, cond, boot)
    np.testing.assert_allclose(np.asarray(info["measure_offdiag"]), _FX["exp/measure_offdiag"], atol=ATOL)
    np.testing.assert_allclose(np.asarray(info["measure_diag"]), _FX["exp/measure_diag"], atol=ATOL)
    np.testing.assert_allclose(np.asarray(info["measure_orth"]), _FX["exp/measure_orth"], atol=ATOL)
    np.testing.assert_allclose(np.asarray(info["measure_loss"]), _FX["exp/measure_loss"], atol=ATOL)


def test_v_and_q_fn_match():
    critic, params, _, b, cond, _ = _pt()
    v = critic.v_fn(params, b["observations"], cond)
    _, qs = critic.q_fn(params, b["observations"], cond, b["actions"])
    np.testing.assert_allclose(np.asarray(v), _FX["exp/v"], atol=ATOL)
    np.testing.assert_allclose(np.asarray(qs[0]), _FX["exp/q1"], atol=ATOL)
    np.testing.assert_allclose(np.asarray(qs[1]), _FX["exp/q2"], atol=ATOL)
