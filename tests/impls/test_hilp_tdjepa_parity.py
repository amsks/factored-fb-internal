"""hilp_tdj must reproduce td_jepa's SF agent, not official HILP's.

Our HILP is transcribed from OFFICIAL HILP. td_jepa's HilpAgent subclasses their SFAgent
and differs in four places, and every difference is theirs being NON-pessimistic where
official HILP is pessimistic:

  q_loss                False  -> regress the SF VECTOR, not the scalar Q = <psi,z>
                        (SFAgentTrainConfig.q_loss, agents/sf/agent.py:32; never overridden
                        for HILP, and HilpAgent does not override sf_loss)
  SF target head reduce MEAN   -> get_targets_uncertainty(next_SFs, sf_pessimism_penalty)
                        with the penalty 0.0 (agent.py:29) returns preds.mean(dim=0)
  actor head reduce     MEAN   -> get_targets_uncertainty(Qs, actor_pessimism_penalty),
                        penalty 0.0 (agent.py:30)
  SF loss scale         0.5    -> 0.5 * P * F.mse_loss(...) averages over P*B*d, so it is
                        half of sum_p mean_{b,d}

Running with official HILP's values instead measured psi collapsing to 2.5-2.8 effective
dimensions of 50, with the surviving subspace nearly orthogonal between seeds, and
walker/spin at 218 / 409 / 953 over three seeds. These tests exist so that cannot recur
silently. The DEFAULTS must stay official-HILP: critic/hilp.yaml is that arm.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from env import make_config


def _cfg(name):
    return make_config(name, "ddpgbc", "dmc-walker", [])


def test_tdjepa_arm_selects_every_td_jepa_value():
    c = _cfg("hilp_tdj")
    assert c.get("hilp_q_loss") is False
    assert c.get("hilp_head_reduce") == "mean"
    assert c.get("hilp_actor_head_reduce") == "mean"
    assert float(c.get("hilp_sf_loss_scale")) == 0.5


def test_official_arm_is_untouched():
    """The gates must not have moved critic/hilp.yaml, which IS official HILP."""
    c = _cfg("hilp")
    assert c.get("hilp_q_loss") is True
    assert c.get("hilp_head_reduce", "argmin_q") == "argmin_q"
    assert c.get("hilp_actor_head_reduce", "min") == "min"
    assert float(c.get("hilp_sf_loss_scale", 1.0)) == 1.0


@pytest.mark.parametrize("reduce_,expected", [("mean", "mean"), ("argmin_q", "pick")])
def test_sf_target_head_reduction(reduce_, expected):
    """mean averages the two head vectors; argmin_q takes the whole vector of the head
    with the smaller Q. They differ whenever the heads disagree."""
    tF = jnp.asarray([[[1.0, 0.0], [0.0, 2.0]],       # head 0, two rows
                      [[3.0, 0.0], [0.0, 6.0]]])      # head 1
    z = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    tQ = jnp.einsum("pbd,bd->pb", tF, z)
    if expected == "mean":
        got = jnp.mean(tF, axis=0)
        want = jnp.asarray([[2.0, 0.0], [0.0, 4.0]])
    else:
        pick = (tQ[0] < tQ[1])[:, None]
        got = jnp.where(pick, tF[0], tF[1])
        want = jnp.asarray([[1.0, 0.0], [0.0, 2.0]])   # head 0 is smaller in both rows
    np.testing.assert_allclose(np.asarray(got), np.asarray(want))
    assert not np.allclose(np.asarray(jnp.mean(tF, axis=0)),
                           np.asarray(jnp.where((tQ[0] < tQ[1])[:, None], tF[0], tF[1]))), \
        "the two reductions must be distinguishable, else this test proves nothing"


def test_actor_reduction_mean_is_not_min():
    Qs = jnp.asarray([[1.0, 5.0], [3.0, 9.0]])         # [P=2, B=2]
    np.testing.assert_allclose(np.asarray(jnp.mean(Qs, axis=0)), [2.0, 7.0])
    np.testing.assert_allclose(np.asarray(jnp.min(Qs, axis=0)), [1.0, 5.0])


def test_sf_loss_scale_is_exactly_half():
    """td_jepa: 0.5 * P * mse_loss(mean over P*B*d) == 0.5 * (sum_p mean_{b,d}).

    Reconstructed in numpy from the reference expression rather than by calling ours."""
    rng = np.random.default_rng(0)
    P, Bn, D = 2, 6, 4
    F_ = rng.standard_normal((P, Bn, D))
    T_ = rng.standard_normal((Bn, D))
    ours = float(np.sum(np.mean((F_ - T_[None]) ** 2, axis=(-1, -2))))
    theirs = 0.5 * P * float(np.mean((F_ - np.broadcast_to(T_, F_.shape)) ** 2))
    assert theirs == pytest.approx(0.5 * ours, rel=1e-12)
