"""critic_goal_mix / actor_goal_mix: OGBench GCDataset-style goal sampling at the agent
level. mix = [p_traj, p_cur, p_random]; goal in {current state, trajectory future, random
buffer state}. success (GCIQL reward) = goal IS the current state (the cur-goal anchor)."""
import numpy as np
import torch
from gymnasium.spaces import Box

from agents.fb.agent import FBAgent


def _agent(**kw):
    obs_space = Box(low=-1, high=1, shape=(12,), dtype=np.float32)
    return FBAgent(obs_space=obs_space, action_dim=4, batch_size=8,
                   z_dim=16, L_dim=16, device="cpu", **kw)


def test_all_cur_returns_current_state_and_success():
    ag = _agent()
    cur, traj = torch.randn(8, 12), torch.randn(8, 12)
    goal, is_cur = ag._sample_goal_mix(cur, traj, [0.0, 1.0, 0.0])
    assert torch.allclose(goal, cur)
    assert is_cur.all()


def test_all_traj_returns_future_and_no_success():
    ag = _agent()
    cur, traj = torch.randn(8, 12), torch.randn(8, 12)
    goal, is_cur = ag._sample_goal_mix(cur, traj, [1.0, 0.0, 0.0])
    assert torch.allclose(goal, traj)
    assert not is_cur.any()


def test_all_random_is_permutation_of_current_and_no_success():
    ag = _agent()
    cur = torch.arange(8).float().unsqueeze(-1).repeat(1, 12)  # distinct rows
    traj = torch.randn(8, 12)
    goal, is_cur = ag._sample_goal_mix(cur, traj, [0.0, 0.0, 1.0])
    assert not is_cur.any()
    for r in goal:                       # every goal row is some current-state row
        assert any(torch.allclose(r, c) for c in cur)


def _iql_batch():
    B = 8
    return {
        "observation": torch.randn(B, 12),
        "action": torch.randn(B, 4).clamp(-1, 1),
        "next": {"observation": torch.randn(B, 12), "terminated": torch.zeros(B, 1),
                 "action": torch.randn(B, 4)},
        "future": {"observation": torch.randn(B, 12)},   # no physics: goal_match doesn't need it
    }


def test_goal_match_all_cur_gives_full_success_and_no_physics_needed():
    ag = _agent(critic_mode="iql", train_goal_mode="traj", iql_reward="goal_match",
                critic_goal_mix=[0.0, 1.0, 0.0], fixed_b="none", ortho_coef=0.0, bc_coeff=3.0)
    m = ag.update(_iql_batch(), step=0)
    assert m["iql_r_pos"] == 1.0          # all cur-goals => every transition "reached"


def test_goal_match_all_traj_gives_zero_success():
    ag = _agent(critic_mode="iql", train_goal_mode="traj", iql_reward="goal_match",
                critic_goal_mix=[1.0, 0.0, 0.0], fixed_b="none", ortho_coef=0.0, bc_coeff=3.0)
    m = ag.update(_iql_batch(), step=0)
    assert m["iql_r_pos"] == 0.0          # all traj-goals => none reached
