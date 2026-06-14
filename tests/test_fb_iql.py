"""FB-IQL critic (critic_mode='iql'): a goal-conditioned expectile value V(s,z) on the
bilinear Q=F(s,a,z).z REPLACES the measure loss. z = B(trajectory future), so gradient
flows into B end-to-end; the goal-reaching reward is r = 1[||next_cube - goal_cube|| <
thresh] on physics[:,14:17]. Default 'measure' is stock FB (byte-identical). Spec:
PAPER/specs/2026-06-11-interaction-weighted-fb.md.
"""
import numpy as np
import pytest
import torch

from agents.fb.agent import expectile_loss


# --- expectile loss: the IQL asymmetry (pure function) -----------------------------

def test_expectile_reduces_to_half_mse():
    torch.manual_seed(0)
    diff = torch.randn(1000)
    assert torch.allclose(expectile_loss(diff, 0.5), 0.5 * diff.pow(2).mean())


def test_expectile_penalizes_underestimate_more():
    # tau>0.5 weights positive residuals (target above V) more than negative ones.
    pos = torch.full((100,), 1.0)    # target >> V  => want V to rise
    neg = torch.full((100,), -1.0)   # target << V
    tau = 0.9
    assert expectile_loss(pos, tau) > expectile_loss(neg, tau)
    assert torch.allclose(expectile_loss(pos, tau), torch.tensor(tau))         # 0.9 * 1
    assert torch.allclose(expectile_loss(neg, tau), torch.tensor(1.0 - tau))   # 0.1 * 1


def test_expectile_fits_upper_expectile():
    # Minimizing expectile(target - V) for tau>0.5 lands V strictly above the mean
    # (which is the tau=0.5 solution). Bimodal target: mean = 1.0.
    target = torch.cat([torch.zeros(90), torch.full((10,), 10.0)])
    V = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([V], lr=0.05)
    for _ in range(2000):
        opt.zero_grad()
        expectile_loss(target - V, 0.9).backward()
        opt.step()
    assert V.item() > target.mean().item()    # upper expectile sits above the mean


# --- agent build + integration -----------------------------------------------------

def _tiny_flowbc(**extra):
    import gymnasium
    from agents.fb.flow_bc.agent import FBFlowBCAgent
    from nn_models import IdentityNNConfig, NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig
    from normalizers import IdentityNormalizerConfig
    return FBFlowBCAgent(
        obs_space=gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        action_dim=2, z_dim=8, L_dim=8, batch_size=16,
        actor_cfg=NoiseConditionedActorArchiConfig(),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(),
        obs_normalizer_cfg=IdentityNormalizerConfig(),
        rgb_encoder_cfg=IdentityNNConfig(),
        augmentator_cfg=IdentityNNConfig(),
        device="cpu", **extra,
    )


def _batch(B=16, phys_next=None, phys_fut=None):
    """A traj-mode batch with the future + physics keys the IQL critic consumes."""
    phys_next = torch.zeros(B, 20) if phys_next is None else phys_next
    phys_fut = torch.zeros(B, 20) if phys_fut is None else phys_fut
    return {
        "observation": torch.randn(B, 4),
        "action": torch.rand(B, 2) * 2 - 1,
        "next": {
            "observation": torch.randn(B, 4),
            "terminated": torch.zeros(B, 1),
            "physics": phys_next,
        },
        "future": {
            "observation": torch.randn(B, 4),
            "physics": phys_fut,
        },
    }


def test_critic_mode_default_is_measure():
    a = _tiny_flowbc()
    assert a.critic_mode == "measure"
    assert not hasattr(a, "_value_mlp")        # value net built only in iql mode


def test_iql_builds_value_net_with_obs_plus_z_input():
    a = _tiny_flowbc(critic_mode="iql", train_goal_mode="traj")
    assert hasattr(a, "_value_mlp") and hasattr(a, "value_optimizer")
    assert a._value_mlp[0].in_features == 4 + a.z_dim   # encoded obs (4) + z (8)


def test_iql_update_runs_and_reports_critic_losses():
    a = _tiny_flowbc(critic_mode="iql", train_goal_mode="traj")
    m = a.update(_batch(), step=0)
    for k in ("q_loss", "value_loss", "orth_loss", "actor_loss"):
        assert k in m and np.isfinite(m[k]), k


def test_iql_requires_future_goals():
    a = _tiny_flowbc(critic_mode="iql", train_goal_mode="traj")
    b = _batch()
    del b["future"]                            # no train_goal_mode='traj' future buffer
    with pytest.raises(AssertionError):
        a.update(b, step=0)


def test_iql_reward_is_goal_reaching_on_cube_xyz():
    # next cube xyz == physics[:,14:17]; r = 1[||next - goal|| < iql_thresh].
    a = _tiny_flowbc(critic_mode="iql", train_goal_mode="traj", iql_thresh=0.04)
    B = 16
    pn = torch.zeros(B, 20)
    pf = torch.zeros(B, 20)
    pf[B // 2:, 14] = 1.0                       # half reached (dist 0), half far (dist 1)
    m = a.update(_batch(B=B, phys_next=pn, phys_fut=pf), step=0)
    assert np.isclose(m["iql_r_pos"], 0.5, atol=1e-6)
