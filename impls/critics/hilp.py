"""HILP — Hilbert-representation successor features (Park et al., ICML 2024), transcribed
from the OFFICIAL repo: github.com/seohongpark/HILP, `hilp_zsrl/url_benchmark/agent/sf.py`
(+ `ddpg.py` for the extractor), cloned to ~/git/baselines/HILP.

WHY THE OFFICIAL REPO AND NOT A LOCAL COPY. Two other HILP implementations sit in
~/git/baselines (`RLU/controllable_agent/.../hilp.py`, Meta's reimplementation, and
`td_jepa/metamotivo/agents/hilp/`). Porting from a reimplementation is exactly what
produced the RLDP run-1 failure: the seam faithfully transcribed td_jepa's RLDP, which is
a structurally different algorithm from the official one, and the difference only surfaced
after 12 GPU-hours. See docs/notes/2026-08-28-official-parity-campaign-1.md.

THE FIVE TRANSCRIPTION TRAPS (docs/notes/2026-08-28-hilp-paper-eval.md records the audit):

  1. THREE reductions in one phi loss. `adv` is built from min(next_v1, next_v2) but
     subtracts the MEAN of the two TARGET heads; the two regression targets q1/q2 use
     next_v1/next_v2 SEPARATELY. Collapsing any of these to a single convention changes
     the objective.
  2. `hilp_expectile` ships at 0.5 (SFAgentConfig), NOT the 0.7 default in
     `expectile_loss`'s own signature. At 0.5 the weight is where(adv>=0, .5, .5) -- a
     CONSTANT -- so the IQL asymmetry is numerically INERT at the published setting and
     the loss is plain 0.5*diff^2. `adv` is still computed and logged. A port that
     "corrects" this to 0.7 is not running the published configuration.
  3. `feature_net` subtracts an EMA running mean from phi1 (0.995/0.005). `running_std` is
     maintained upstream in the same place and never read; carried here for fidelity and
     for the diagnostic, not used in any expression.
  4. The SF target uses the ONLINE feature net, detached -- NOT a target phi. The target
     phi nets exist only inside the phi value loss. (Compare the RLDP bug, where reading
     the target rather than the online backward tower diverged the run: same class of
     mistake, opposite direction.)
  5. `q_loss: True` by default, so the SF loss regresses the SCALAR Q = <F, z>, not the
     vector F; and `next_F` is selected PER SAMPLE by argmin over Q
     (`where(next_Q1 < next_Q2, next_F1, next_F2)`) -- a whole-vector choice, not an
     elementwise min.

STATUS: the phi learner below is transcribed and pinned by an independent numpy
reconstruction (tests/impls/test_hilp_seam.py). The SF layer and the extractor are
transcribed but NOT yet pinned by a parity test, and this critic is deliberately NOT
registered in impls/agents/__init__.py::CRITICS until they are -- an unregistered critic
cannot be selected by --critic and so cannot silently produce a number.
"""
from __future__ import annotations

import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.utils.networks import ForwardMap, PhiMap
from impls.utils.fb_math import project_z, targets_uncertainty


def hilp_value(phi_a, phi_b):
    """v = -||phi(a) - phi(b)||, with the reference's clamp INSIDE the sqrt.

    `-torch.sqrt(torch.clamp(squared_dist, min=1e-6))` (sf.py:139-142). Clamping the
    SQUARE, not the root, matters: it floors v at -1e-3, and it keeps the gradient finite
    at phi(a) == phi(b), which is exactly the case the hindsight-goal relabelling makes
    common (obs and goal coincide whenever the sampled future is the current step)."""
    sq = jnp.sum((phi_a - phi_b) ** 2, axis=-1)
    return -jnp.sqrt(jnp.clip(sq, 1e-6, None))


def expectile_loss(adv, diff, expectile):
    """weight * diff^2 with weight = expectile if adv >= 0 else 1 - expectile (sf.py:150).

    NOTE the shipped expectile is 0.5, which makes the weight a constant 0.5 and this an
    ordinary squared error -- see trap 2 in the module docstring."""
    weight = jnp.where(adv >= 0, expectile, 1.0 - expectile)
    return weight * diff ** 2


def hilp_phi_loss(phi1_p, phi2_p, tphi1_p, tphi2_p, apply_phi,
                  obs, goals, next_obs, discount, expectile):
    """The official HILP feature loss, sf.py:154-198, as a pure function.

    Split out of the critic so an independent numpy reconstruction can call exactly this
    with exactly these arrays (tests/impls/test_hilp_seam.py). `apply_phi(params, x)`
    applies one phi tower.

    The upstream `forward()` also soft-updates the target phis and the running mean as
    SIDE EFFECTS after computing the loss. Those are not side effects here -- the seam
    polyaks targets in Agent.apply_update and the running mean in `update_stats`, both
    strictly after the gradient step, which preserves the upstream ordering.
    """
    same = (jnp.linalg.norm(obs - goals, axis=-1) < 1e-6).astype(obs.dtype)
    masks = 1.0 - same
    rewards = same - 1.0

    # target heads, stop-gradded (upstream: is_target=True detaches)
    tn1 = jax.lax.stop_gradient(hilp_value(apply_phi(tphi1_p, next_obs), apply_phi(tphi1_p, goals)))
    tn2 = jax.lax.stop_gradient(hilp_value(apply_phi(tphi2_p, next_obs), apply_phi(tphi2_p, goals)))
    tv1 = jax.lax.stop_gradient(hilp_value(apply_phi(tphi1_p, obs), apply_phi(tphi1_p, goals)))
    tv2 = jax.lax.stop_gradient(hilp_value(apply_phi(tphi2_p, obs), apply_phi(tphi2_p, goals)))

    # trap 1: min for adv's q, MEAN of the target heads for its baseline, per-head for
    # the regression targets. Three different reductions, deliberately.
    next_v = jnp.minimum(tn1, tn2)
    q = rewards + discount * masks * next_v
    v_t = (tv1 + tv2) / 2.0
    adv = q - v_t

    q1 = rewards + discount * masks * tn1
    q2 = rewards + discount * masks * tn2
    v1 = hilp_value(apply_phi(phi1_p, obs), apply_phi(phi1_p, goals))
    v2 = hilp_value(apply_phi(phi2_p, obs), apply_phi(phi2_p, goals))

    loss = (jnp.mean(expectile_loss(adv, q1 - v1, expectile))
            + jnp.mean(expectile_loss(adv, q2 - v2, expectile)))
    info = {"hilp_value_loss": loss, "hilp_v_mean": jnp.mean((v1 + v2) / 2.0),
            "hilp_adv_mean": jnp.mean(adv), "hilp_abs_adv_mean": jnp.mean(jnp.abs(adv)),
            "hilp_accept_prob": jnp.mean((adv >= 0).astype(loss.dtype))}
    return loss, info
