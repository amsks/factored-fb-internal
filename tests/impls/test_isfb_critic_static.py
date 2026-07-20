"""SG-ISFB critic parity against the torch reference (tests/impls/fixtures/isfb_reference.npz,
baked by _gen_isfb_fixture.py from agents/fb/isfb/).

EACH loss term is asserted separately at 1e-10. The fixture is baked with num_parallel=3 and
z_dim=8 (and every other dim distinct) precisely so the scale-factor cluster the scout flagged
— torch's absent 1/sqrt(d), head-SUMMED L_M, and the xP on L_V/L_N — is observable rather than
collapsing to 1. fb_pessimism_penalty is non-zero so the head-spread target reduction is live.
"""
import os

import jax.numpy as jnp
import numpy as np

from impls.critics.isfb import ISFBCritic
from impls.utils import torch_to_flax as t2f
from impls.utils.networks import ForwardMap, PhiMap

ATOL = 1e-10
_FX = dict(np.load(os.path.join(os.path.dirname(__file__), "fixtures", "isfb_reference.npz")))
_M = {k[len("meta/"):]: _FX[k] for k in _FX if k.startswith("meta/")}

Z, L, P, H = int(_M["z_dim"]), int(_M["L_dim"]), int(_M["num_parallel"]), int(_M["hidden_dim"])
MODES = ("coord", "scalar_gate_hard", "scalar_gate_awr")


def _critic(mode):
    return ISFBCritic(
        forward=ForwardMap(z_dim=Z, hidden_dim=H, num_parallel=P, hidden_layers=2,
                           embedding_layers=2),
        value_forward=ForwardMap(z_dim=Z, hidden_dim=H, num_parallel=P, hidden_layers=2,
                                 embedding_layers=2),
        backward=PhiMap(z_dim=Z, hidden_dim=H, hidden_layers=4, norm=True),
        left_encoder=PhiMap(z_dim=L, hidden_dim=H, hidden_layers=4, norm=True),
        z_dim=Z, action_dim=int(_M["action_dim"]), num_parallel=P, discount=float(_M["discount"]),
        ortho_coef=float(_M["ortho_coef"]), fb_pessimism_penalty=float(_M["fb_pessimism_penalty"]),
        train_goal_ratio=0.5, norm_z=True, f_tau=0.005, b_tau=0.005,
        expectile=float(_M["expectile"]), measure_insample=mode,
        alpha_n=float(_M["alpha_n"]), alpha_m=float(_M["alpha_m"]), alpha_v=1.0,
        awr_beta=float(_M["awr_beta"]), awr_wmax=float(_M["awr_wmax"]),
        awr_normalize_adv=True, goal_mask=True,
    )


def _params():
    online = {
        "forward": t2f.load_forward_params(_FX),
        "value_forward": t2f.load_forward_params(_FX, prefix="_value_forward_map"),
        "backward": t2f.load_backward_params(_FX),
        "left_encoder": t2f.load_left_encoder_params(_FX),
    }
    targets = {
        "forward": t2f.load_forward_params(_FX, prefix="_target_forward_map"),
        "value_forward": t2f.load_forward_params(_FX, prefix="_target_value_forward_map"),
        "backward": t2f.load_backward_params(_FX, prefix="_target_backward_map"),
        "left_encoder": t2f.load_backward_params(_FX, prefix="_target_left_encoder"),
    }
    return online, targets


def _batch():
    b = {k: jnp.asarray(_FX["in__" + k], jnp.float64)
         for k in ("observations", "actions", "next_observations", "masks")}
    return b


def _cond():
    return {"z": jnp.asarray(_FX["in__z"], jnp.float64)}


def test_intermediates_match():
    """Localize any parity failure: F, F_V, B, M, N, q, v and the reduced target measure."""
    critic = _critic("scalar_gate_hard")
    online, targets = _params()
    b, cond = _batch(), _cond()
    got = critic.debug_terms(online, targets, b, cond)
    for key in ("le", "Fs", "Fv", "B", "M", "N", "tN", "tN_M", "q", "v"):
        np.testing.assert_allclose(np.asarray(got[key]), _FX["exp/" + key], atol=ATOL,
                                   err_msg=f"intermediate {key} mismatch")


def test_each_loss_term_matches_per_mode():
    online, targets = _params()
    b, cond = _batch(), _cond()
    for mode in MODES:
        critic = _critic(mode)
        loss, info = critic.stage_loss("isfb", online, targets, b, cond, boot=None)
        for term, ikey in (("L_V", "isfb_L_V"), ("L_N", "isfb_L_N"), ("L_M", "isfb_L_M"),
                           ("ortho", "ortho_loss")):
            np.testing.assert_allclose(float(info[ikey]), float(_FX[f"exp/{mode}/{term}"]),
                                       atol=ATOL, err_msg=f"{mode}/{term}")
        np.testing.assert_allclose(float(loss), float(_FX[f"exp/{mode}/total"]), atol=ATOL,
                                   err_msg=f"{mode}/total")


def test_coord_mode_drops_the_scalar_gate():
    """isfb.tex's boxed L_Coord-ISFB contains no L_V; torch drops it and reports 0.0.
    (The jax shim always adds it — scout disagreement #7, torch/paper authoritative.)"""
    online, targets = _params()
    _, info = _critic("coord").stage_loss("isfb", online, targets, _batch(), _cond(), boot=None)
    assert float(info["isfb_L_V"]) == 0.0
    np.testing.assert_allclose(float(_FX["exp/coord/L_V"]), 0.0, atol=ATOL)


def test_stage_loss_ignores_bootstrap():
    """ISFB has no a'~pi bootstrap; a poisoned boot dict must be untouched."""
    online, targets = _params()
    critic = _critic("scalar_gate_hard")
    ref, _ = critic.stage_loss("isfb", online, targets, _batch(), _cond(), boot=None)

    class Boom(dict):
        def __getitem__(self, k):
            raise AssertionError("ISFB critic read boot[%r] (no a'~pi bootstrap allowed)" % k)

    got, _ = critic.stage_loss("isfb", online, targets, _batch(), _cond(), boot=Boom())
    np.testing.assert_allclose(float(got), float(ref), atol=ATOL)


def test_v_fn_matches_scalar_value_readout():
    """PROVIDES_V: v_fn is the head-reduced <F_V(s,z), z>, on the same scale as q_fn."""
    critic = _critic("scalar_gate_hard")
    online, _ = _params()
    b, cond = _batch(), _cond()
    v = critic.v_fn(online, b["observations"], cond)
    q, qs = critic.q_fn(online, b["observations"], cond, b["actions"])
    exp_v = _FX["exp/v"]
    exp_q = _FX["exp/q"]
    np.testing.assert_allclose(np.asarray(v), exp_v.mean(0), atol=ATOL)
    np.testing.assert_allclose(np.asarray(q), exp_q.mean(0), atol=ATOL)
    np.testing.assert_allclose(np.asarray(qs), exp_q, atol=ATOL)
