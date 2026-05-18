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


def _fake_ep(outcome, d, v_at_d):
    """Build a profile-episode whose V(s_t) = v_at_d(d_t) via stub model."""
    T = len(d)
    obs = np.zeros((T, 5), np.float32)
    obs[:, 0] = [v_at_d(x) for x in d]      # _StubModel Q ~ obs[:,0]
    return {
        "obs": obs, "action": np.zeros((max(T - 1, 0), 2), np.float32),
        "d": np.asarray(d, np.float64),
        "transport_mask": np.ones(T, bool),
        "outcome": outcome, "length": T,
    }


def test_value_landscape_slope_success_vs_fail():
    m, ag = _StubModel(), _StubAgent()
    z = torch.tensor([1.0, 0.0, 0.0])
    # success: V rises as d -> 0 (slope wrt -d > 0); fail: V flat
    eps = [
        _fake_ep("success", [0.30, 0.20, 0.10, 0.02], lambda x: 1.0 - x),
        _fake_ep("transport_fail", [0.30, 0.20, 0.10, 0.05], lambda x: 0.5),
    ]
    per_ep, per_step = rp.probe_value_landscape(m, ag, eps, z)
    s = per_ep.set_index("outcome")["slope_V_negd"]
    assert s.loc["success"] > 0.5
    assert abs(s.loc["transport_fail"]) < 1e-6
    assert set(per_step.columns) >= {"outcome", "d", "V"}


def test_z_decoding_topk_at_goal():
    m = _StubModel()  # B(next_obs) = first 3 dims
    N = 20
    next_obs = torch.zeros((N, 5))
    # 5 states aligned with z=[1,0,0]; of those, 3 are at goal (d<0.04)
    next_obs[:5, 0] = 10.0
    physics = np.zeros((N, 20), np.float32)
    goal = np.array([0.0, 0.0, 0.02], np.float64)
    physics[:, 14:17] = goal              # all at goal by default
    physics[3:, 14] = 1.0                 # indices 3.. are FAR from goal
    z = torch.tensor([1.0, 0.0, 0.0])
    relabel_metrics = {"relabel_reward#nonzero": 3.0,
                       "relabel_reward#num_samples": 20.0}
    out = rp.probe_z_decoding(m, next_obs, physics, goal, z,
                              relabel_metrics, topk=5)
    assert out["relabel_pos_frac"] == 3.0 / 20.0
    assert 0.0 <= out["topk_pct_at_goal"] <= 1.0
    # top-5 by B·z are indices 0..4; of these, 0,1,2 at goal => 60%
    assert abs(out["topk_pct_at_goal"] - 0.6) < 1e-9
    assert out["topk_mean_d"] >= 0.0


def test_b_resolution_linear_vs_random():
    rng = np.random.default_rng(0)
    N = 400
    B_lin = rng.normal(size=(N, 4)).astype(np.float32)
    d_lin = (B_lin @ np.array([1.0, -2.0, 0.5, 0.0]) + 0.1).astype(np.float64)
    out_lin = rp.probe_b_resolution(B_lin, d_lin, seed=0)
    assert out_lin["r2"] > 0.95

    B_rnd = rng.normal(size=(N, 4)).astype(np.float32)
    d_rnd = rng.normal(size=N).astype(np.float64)
    out_rnd = rp.probe_b_resolution(B_rnd, d_rnd, seed=0)
    assert out_rnd["r2"] < 0.2

    # separability: placed (d<0.04) vs near (0.04..0.12) clearly offset
    B_sep = np.vstack([np.zeros((100, 4)), np.ones((100, 4))]).astype(np.float32)
    d_sep = np.concatenate([np.full(100, 0.01), np.full(100, 0.08)])
    out_sep = rp.probe_b_resolution(B_sep, d_sep, seed=0)
    assert out_sep["placed_vs_near_acc"] > 0.95
