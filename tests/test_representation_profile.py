import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals import representation_profile as rp


class _StubModel:
    """B = first z_dim obs dims; F = (obs0 + action0) broadcast, 2 heads."""
    z_dim = 3

    def backward_map(self, obs):
        return obs[:, : self.z_dim].clone()

    def forward_map(self, obs, z, action):
        base = (obs[:, 0:1] + action[:, 0:1]).expand(-1, self.z_dim)  # [B,zd]
        return torch.stack([base, base + 1.0], dim=0)                 # [2,B,zd]


class _StubAgent:
    device = "cpu"

    def act(self, obs, z):
        return torch.zeros((obs.shape[0], 2))


def test_ensemble_q_means_over_heads():
    F = torch.tensor([[[1.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]])  # [2,1,3]
    z = torch.tensor([2.0, 0.0, 0.0])
    q = rp._ensemble_q(F, z)
    assert q.shape == (1,)
    assert q[0] == 2.0 * ((1.0 + 3.0) / 2)  # mean over heads of (F·z)


def test_q_values_monotone_in_action():
    m = _StubModel()
    obs = torch.zeros((4, 5))
    z = torch.ones(3)
    a_lo = torch.zeros((4, 2))
    a_hi = torch.ones((4, 2))
    q_lo = rp.q_values(m, obs, a_lo, z)
    q_hi = rp.q_values(m, obs, a_hi, z)
    assert q_lo.shape == (4,)
    assert np.all(q_hi > q_lo)


def test_v_values_uses_policy_action():
    m, ag = _StubModel(), _StubAgent()
    obs = torch.zeros((3, 5))
    z = torch.ones(3)
    v = rp.v_values(m, ag, obs, z)
    # policy action is zeros => same as q_values with zero action
    assert np.allclose(v, rp.q_values(m, obs, torch.zeros((3, 2)), z))


from evals import phase_probe as pp


class _PhaseEnv:
    """3-step episode; manip-like; cube approaches goal so it 'succeeds'."""
    def __init__(self):
        self.unwrapped = _PhaseUnwrapped()
        self.t = 0

    def _info(self, eff, cube):
        return {
            "proprio/effector_pos": np.array(eff, np.float64),
            "proprio/gripper_opening": np.array([1.0]),
            "privileged/block_0_pos": np.array(cube, np.float64),
            "success": self.t >= 3,
        }

    def reset(self, seed=None):
        self.t = 0
        return np.zeros(5, np.float32), self._info([9, 9, 9], [0, 0, 0.02])

    def step(self, a):
        self.t += 1
        done = self.t >= 3
        return (np.full(5, self.t, np.float32), -1.0, done, False,
                self._info([0, 0, 0.20], [0, 0, 0.20]))

    def close(self):
        pass


class _PhaseUnwrapped:
    _target_block = 0
    # _data / _pinch_site_id only satisfy ensure_manip_env; never
    # dereferenced because every Task-2 rollout uses S0 (a no-op).
    _data = object()
    _pinch_site_id = 0
    cur_task_info = {"goal_xyzs": np.array([[0.0, 0.0, 0.20]]),
                     "init_xyzs": np.array([[0.0, 0.0, 0.02]])}


def test_phase_rollout_records_obs_when_requested():
    eps = pp.rollout_with_phase_signals(
        _PhaseEnv(), _StubAgent(), z=torch.zeros(3), n_episodes=1,
        thr=pp.Thresholds(), scenario="S0", record_obs=True)
    e = eps[0]
    assert "obs" in e and "action" in e
    assert e["obs"].shape[0] == e["length"]
    assert e["action"].shape[0] == e["length"] - 1  # actions between steps


def test_rollout_for_profile_attaches_outcome_and_d():
    eps = rp.rollout_for_profile(
        _PhaseEnv(), _StubAgent(), z=torch.zeros(3), n_episodes=1,
        thr=pp.Thresholds())
    e = eps[0]
    assert e["outcome"] in ("success", "transport_fail", "other")
    assert e["d"].shape[0] == e["obs"].shape[0]      # ||cube-goal|| per step
    assert e["transport_mask"].shape[0] == e["obs"].shape[0]
