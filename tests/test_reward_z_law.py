"""Synthetic-reward z's in the training z law (ledger 9.10 next cell).

Spec-12 L2 found that zero-shot transfer works for spatially CONCENTRATED novel
rewards (ball0.4 gap_frac 0.87/0.74, twozone 0.84/0.89) but collapses as the
reward's bandwidth grows (smooth0.8 0.23/0.08, ball1.6 ~ floor). The 1/N-hat
readout fix (L2w) repaired PREDICTION scale but did not rescue CONTROL anywhere,
which localizes the defect to the ACTOR's z-manifold: the trained z law is 50%
random gaussian directions + 50% B(single goal), and a broad reward's inferred
z = project(sum_i r_i B(s'_i)) is an average of many B's that lies off both.

This adds a third component to sample_mixed_z: z's drawn the way the zero-shot
readout draws them, from random synthetic rewards over the batch. Default
reward_z_ratio=0.0 => byte-identical to the pre-existing z law.
"""
import math

import gymnasium as gym
import pytest
import torch

from agents.fb.flow_bc.agent import FBFlowBCAgent
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig

OBS_DIM, B = 40, 64
SLICE = (19, 22)


def _agent(**kw):
    return FBFlowBCAgent(
        obs_space=gym.spaces.Box(-1, 1, (OBS_DIM,)), action_dim=5, batch_size=B,
        z_dim=50, L_dim=50,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2,
                                                   embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4),
        device="cpu", **kw)


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(B, OBS_DIM, generator=g)
    return raw


def _train_goal(a, raw):
    """What FBAgent.update() passes as train_goal for train_goal_mode='perm':
    the bw-encoded, normalized next_obs."""
    return a.model._bw_encoder(a.model._normalize(raw))


# --------------------------------------------------------------- default-off --
def test_ratio_zero_is_bit_identical():
    """The knob must not perturb the RNG stream when off — otherwise every
    existing cell's z law shifts and no comparison to p2_s4{2,3} is matched."""
    raw = _batch()
    ref = _agent()
    new = _agent(reward_z_ratio=0.0)
    tg_ref, tg_new = _train_goal(ref, raw), _train_goal(new, raw)
    new.model.load_state_dict(ref.model.state_dict())
    tg_new = _train_goal(new, raw)

    torch.manual_seed(7)
    z_ref, g_ref = ref.sample_mixed_z(train_goal=tg_ref)
    torch.manual_seed(7)
    z_new, g_new = new.sample_mixed_z(train_goal=tg_new)
    assert torch.equal(z_ref, z_new)
    assert torch.equal(g_ref, g_new)


# ------------------------------------------------------- the equivalence gate --
@pytest.mark.parametrize("family", ["kernel", "rff"])
def test_reward_z_matches_reward_inference(family):
    """THE load-bearing test: a training reward-z must be the SAME object the
    zero-shot eval produces — project_z(sum_i r_i B(s'_i)) — so the actor
    practices on the z's it will later be driven with."""
    a = _agent(reward_z_ratio=1.0, reward_z_family=family, reward_z_slice=SLICE)
    raw = _batch(1)
    a._raw_next_obs = raw
    tg = _train_goal(a, raw)

    torch.manual_seed(3)
    z, g_used = a.sample_mixed_z(train_goal=tg)
    assert z.shape == (B, 50)
    # every row is a reward-z => the goal placeholder is zero everywhere
    assert (g_used.abs().sum(-1) == 0).all()

    # recompute the rewards this call drew, and check each z against FB's own
    # reward_inference on the RAW observations (the eval path, normalizer and all)
    r = a._last_reward_z_rewards                       # [B_rows, B] cached by the impl
    assert r.shape == (B, B) and (r >= 0).all() and (r.sum(-1) > 0).all()
    for i in range(0, B, 8):
        z_eval = a.model.reward_inference(next_obs=raw, reward=r[i][:, None])
        cos = torch.nn.functional.cosine_similarity(z[i][None], z_eval[None].reshape(1, -1))
        assert cos.item() > 1 - 1e-5, f"row {i}: cos={cos.item()}"


def test_reward_z_is_projected_to_the_z_sphere():
    a = _agent(reward_z_ratio=1.0)
    raw = _batch(2)
    a._raw_next_obs = raw
    torch.manual_seed(5)
    z, _ = a.sample_mixed_z(train_goal=_train_goal(a, raw))
    assert torch.allclose(z.norm(dim=-1), torch.full((B,), math.sqrt(50)), atol=1e-4)


# ------------------------------------------------------------ mixing marginal --
def test_mixture_marginals():
    """ratio r => r reward-z, (1-r)*train_goal_ratio goal-z, rest gaussian."""
    a = _agent(reward_z_ratio=0.25, train_goal_ratio=0.5)
    raw = _batch(3)
    a._raw_next_obs = raw
    tg = _train_goal(a, raw)
    goal_rows = 0
    reward_rows = 0
    torch.manual_seed(11)
    reps = 40
    for _ in range(reps):
        z, g_used = a.sample_mixed_z(train_goal=tg)
        is_goal = g_used.abs().sum(-1) > 0
        goal_rows += int(is_goal.sum())
        reward_rows += int(a._last_reward_z_mask.sum())
    n = reps * B
    assert 0.20 < reward_rows / n < 0.30
    # goal share is the residual mass: (1-0.25)*0.5 = 0.375
    assert 0.32 < goal_rows / n < 0.43


def test_dirac_limit_recovers_a_goal_z():
    """Sanity on the readout: a reward supported on ONE state must infer that
    state's own z = project_z(B(s)) — the narrow end of the family collapses onto
    the existing goal-z law rather than inventing a different object."""
    a = _agent(reward_z_ratio=1.0, reward_z_family="kernel", reward_z_bw=(1e-4, 1e-4))
    raw = _batch(4)
    a._raw_next_obs = raw
    tg = _train_goal(a, raw)
    torch.manual_seed(13)
    z, _ = a.sample_mixed_z(train_goal=tg)
    goal_z = a.model.project_z(a.model._backward_map(tg))
    # two-zone draws put mass on two states by construction; the Dirac claim is
    # about single-support rewards
    support = (a._last_reward_z_rewards > 0).float().sum(-1)
    single = support == 1
    assert single.float().mean() > 0.5, "fixture drew no single-support rewards"
    for i in range(B):
        if single[i]:
            assert any(torch.allclose(z[i], goal_z[j], atol=1e-4) for j in range(B))


def test_broad_reward_z_is_off_the_goal_z_manifold():
    """The cell's whole premise: a BROAD reward's z is an average of many B's and
    is not any single goal's z — that is the region L2 showed the actor cannot
    steer in, and the region this z law is meant to populate."""
    a = _agent(reward_z_ratio=1.0, reward_z_family="kernel", reward_z_bw=(2.0, 2.5))
    raw = _batch(4)
    a._raw_next_obs = raw
    tg = _train_goal(a, raw)
    torch.manual_seed(13)
    z, _ = a.sample_mixed_z(train_goal=tg)
    goal_z = a.model.project_z(a.model._backward_map(tg))
    # broad support (many rows in the ball) => z differs from every single goal-z
    support = (a._last_reward_z_rewards > 0).float().sum(-1)
    assert support.median() > 3, "bandwidth did not produce broad rewards"
    for i in range(B):
        if support[i] > 3:
            assert not any(torch.allclose(z[i], goal_z[j], atol=1e-4) for j in range(B))


# ------------------------------------------------------------------ bandwidth --
def test_kernel_family_spans_the_requested_bandwidths():
    """The whole point is broad rewards: the drawn supports must actually reach
    the wide end (L2's failing families were sigma/eps >= 0.8)."""
    a = _agent(reward_z_ratio=1.0, reward_z_family="kernel", reward_z_bw=(0.2, 2.0))
    raw = _batch(5)
    a._raw_next_obs = raw
    torch.manual_seed(17)
    a.sample_mixed_z(train_goal=_train_goal(a, raw))
    frac_on = (a._last_reward_z_rewards > 0.5).float().mean(-1)   # support share per reward
    assert frac_on.min() < 0.05, "no concentrated reward drawn"
    assert frac_on.max() > 0.25, "no broad reward drawn"


def test_bad_family_is_refused():
    with pytest.raises(AssertionError):
        _agent(reward_z_ratio=0.5, reward_z_family="nope")


def test_train_py_propagates_the_knobs():
    """train.py's `shared` is an explicit whitelist: a knob present in the yaml
    but absent there is accepted by hydra and SILENTLY DROPPED (caught exactly
    that way — two runs with different reward_z_family produced byte-identical
    losses). Guard the wiring, not just the sampler."""
    import gymnasium
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    import train as train_mod

    cfg_dir = str(Path(train_mod.__file__).resolve().parent / "configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="train",
                      overrides=["agent=fb_measure_flowbc", "domain=cube_single",
                                 "device=cpu", "reward_z_ratio=0.25",
                                 "reward_z_family=rff"])
    agent = train_mod.make_agent(cfg, gymnasium.spaces.Box(-1, 1, (28,)), 5)
    assert agent.reward_z_ratio == 0.25
    assert agent.reward_z_family == "rff"
    assert agent.reward_z_slice == (19, 22)
