"""Tests for scripts/rescuing_fb/eval_fb_zeroshot_rewards.py -- the plain-FB
column of `tab:zeroshot` (PAPER/Multistep-FB/density-fb.tex).

The script's job is comparability: FB's OWN reward readout (z_r = the reward
moment of B) evaluated on the SAME reward families, pool and rollout protocol
as `eval_density_zeroshot.py`. What can silently break that -- and what is
therefore pinned here -- is:

  (a) checkpoint gating: only `fb_flowbc` + `obs_type: state` is the column
      (agent 'fb' is the DDPG+BC row, fb_hitdirac_flowbc is another method,
      pixels is another observation space), and `--force` must warn, not lie;
  (b) the reference check against an archived density JSON: it must FAIL on a
      changed region center, horizon, episode count or pool size, because each
      of those silently makes the column incomparable;
  (c) the degenerate-z guard: a reward with no mass on the pool makes
      project_z(0) non-finite, which must be reported as a dropped cell rather
      than rolled out as if it were a task;
  (d) the rollout reduction: per-env `reset(seed=i)`, success latched over the
      episode (not just the final step), and early termination handled -- the
      exact reductions batched_goal_rollout_density uses.

Torch-free and env-free: the agent and envs are stubs, so this runs on CPU in
under a second (the real numbers come from the wave, not from tests).
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "rescuing_fb" / "eval_fb_zeroshot_rewards.py"


@pytest.fixture(scope="module")
def fbzs():
    spec = importlib.util.spec_from_file_location("eval_fb_zeroshot_rewards", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # numpy-only at module level (torch/ogbench are lazy)
    return mod


# ============================ (a) checkpoint gating ===============================

def test_accepts_only_flowbc_state(fbzs):
    assert fbzs.check_agent_consistency("fb_flowbc", "state") == []


@pytest.mark.parametrize("agent,obs", [
    ("fb", "state"),                   # Gaussian+BC actor -> the DDPG+BC row
    ("fb_hitdirac_flowbc", "state"),   # a different method entirely
    ("fb_measure_flowbc", "state"),
    ("fb_flowbc", "pixels"),           # different observation space
])
def test_refuses_other_variants(fbzs, agent, obs):
    with pytest.raises(fbzs.AgentMismatchError):
        fbzs.check_agent_consistency(agent, obs)


def test_force_warns_instead_of_raising(fbzs):
    warns = fbzs.check_agent_consistency("fb", "state", force=True, run_dir="/x/fbd_s0")
    assert len(warns) == 1 and "NOT the plain-FB column" in warns[0]
    assert "/x/fbd_s0" in warns[0]


# ============================ (b) protocol reference check ========================

def _ref_json(tmp_path, centers, **meta_over):
    meta = {"env": "cube-double-play-v0", "hit_spec": "19:22:0.4,28:31:0.4",
            "pool_size": 16384, "episodes": 25, "horizon": 500,
            "checkpoint_step": 500000, "seed": 0, "relabel_size": 50000}
    meta.update(meta_over)
    fam = {"region": {f"c{i}": {"center_coord": list(map(float, c))}
                      for i, c in enumerate(centers)}}
    p = tmp_path / "ref.json"
    p.write_text(json.dumps({"meta": meta, "families": fam}))
    return p, meta


def test_reference_check_passes_on_identical_protocol(fbzs, tmp_path):
    centers = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]])
    p, meta = _ref_json(tmp_path, centers)
    out = fbzs.check_reference(str(p), meta, centers)
    assert out["ok"] is True
    assert out["checks"]["region_centers"]["max_abs_diff"] == 0.0


def test_reference_check_fails_on_perturbed_center(fbzs, tmp_path):
    """A different rng stream (or a different dataset) shifts the hindsight
    centers -- the region family would then be a DIFFERENT set of rewards."""
    centers = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]])
    p, meta = _ref_json(tmp_path, centers)
    ours = centers.copy()
    ours[1, 2] += 1e-6
    out = fbzs.check_reference(str(p), meta, ours)
    assert out["ok"] is False
    assert out["checks"]["region_centers"]["match"] is False


def test_reference_check_fails_on_missing_center(fbzs, tmp_path):
    centers = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]])
    p, meta = _ref_json(tmp_path, centers)
    out = fbzs.check_reference(str(p), meta, centers[:1])
    assert out["ok"] is False


@pytest.mark.parametrize("field,bad", [("horizon", 200), ("episodes", 10),
                                       ("pool_size", 4096), ("hit_spec", "19:22:0.4"),
                                       ("checkpoint_step", 400000)])
def test_reference_check_fails_on_protocol_drift(fbzs, tmp_path, field, bad):
    centers = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    p, meta = _ref_json(tmp_path, centers)
    ours = dict(meta)
    ours[field] = bad
    out = fbzs.check_reference(str(p), ours, centers)
    assert out["ok"] is False
    assert out["checks"][field]["match"] is False


# ============================ (c) degenerate-z guard ==============================

class _StubModel:
    def __init__(self, z_dim=4, nan=False):
        self.z_dim, self.nan = z_dim, nan

    def sample_z(self, n):
        import torch
        return torch.ones((n, self.z_dim))


class _StubAgent:
    """Minimal FBAgent surface: infer_z (the reward moment) + act + reset."""

    def __init__(self, z_dim=4, act_dim=2, nan_z=False):
        self.model = _StubModel(z_dim)
        self.act_dim = act_dim
        self.nan_z = nan_z
        self.seen_z = []

    def infer_z(self, obs, reward):
        import torch
        if self.nan_z:
            return torch.full((self.model.z_dim,), float("nan"))
        # the real estimator's shape contract: [N,obs] x [N,1] -> [z_dim]
        assert reward.shape == (obs.shape[0], 1), reward.shape
        return (obs[:, : self.model.z_dim] * reward).sum(0)

    def reset(self):
        pass

    def act(self, obs, z, eval_mode=True):
        import torch
        self.seen_z.append(np.asarray(z))
        return torch.ones((obs.shape[0], self.act_dim)) * 5.0   # unclipped on purpose


def test_infer_z_refuses_zero_mass_reward(fbzs):
    pytest.importorskip("torch")
    agent = _StubAgent()
    z, diag = fbzs.fb_infer_z(agent, np.zeros((8, 6), np.float32), np.zeros(8))
    assert z is None and diag["degenerate"] is True
    assert diag["pool_r_nonzero"] == 0 and diag["pool_r_sum"] == 0.0


def test_infer_z_reports_nonzero_count_and_norm(fbzs):
    pytest.importorskip("torch")
    agent = _StubAgent()
    r = np.zeros(8)
    r[[1, 5]] = 1.0
    z, diag = fbzs.fb_infer_z(agent, np.ones((8, 6), np.float32), r)
    assert diag["degenerate"] is False and diag["pool_r_nonzero"] == 2
    assert diag["pool_size"] == 8 and diag["z_norm"] > 0
    assert z.shape == (4,)


def test_infer_z_flags_non_finite_z(fbzs):
    pytest.importorskip("torch")
    z, diag = fbzs.fb_infer_z(_StubAgent(nan_z=True), np.ones((4, 6), np.float32),
                              np.ones(4))
    assert diag["degenerate"] is True


# ============================ (d) rollout reduction ===============================

class _StubEnv:
    """Records its reset seed; reports success at `succ_at`; ends at `end_at`."""

    class _Space:
        shape = (6,)

    observation_space = _Space()

    def __init__(self, succ_at=None, end_at=None):
        self.succ_at, self.end_at = succ_at, end_at
        self.t = 0
        self.seed_seen = None
        self.steps = 0

    def reset(self, seed=None):
        self.seed_seen = seed
        self.t = 0
        return np.zeros(6, np.float32), {}

    def step(self, action):
        assert np.all(np.abs(action) <= 1.0 + 1e-9), "actions must reach the env clipped"
        self.t += 1
        self.steps += 1
        info = {"success": self.succ_at is not None and self.t == self.succ_at}
        done = self.end_at is not None and self.t >= self.end_at
        return np.full(6, float(self.t), np.float32), 0.0, done, False, info


def test_rollout_seeds_envs_by_index_and_clips_actions(fbzs):
    pytest.importorskip("torch")
    envs = [_StubEnv() for _ in range(4)]
    fbzs.batched_z_rollout(_StubAgent(), envs, np.ones(4), H=3)
    assert [e.seed_seen for e in envs] == [0, 1, 2, 3]


def test_rollout_latches_success_over_the_episode(fbzs):
    """Success at ANY step counts, exactly as the density harness latches
    `info['success']` -- an env that succeeds mid-episode and then leaves the
    region still scores 1 on the dirac/dense families."""
    pytest.importorskip("torch")
    envs = [_StubEnv(succ_at=2), _StubEnv(succ_at=None), _StubEnv(succ_at=5)]
    _ot, lengths, succ = fbzs.batched_z_rollout(_StubAgent(), envs, np.ones(4), H=6)
    assert list(succ) == [1.0, 0.0, 1.0]
    assert list(lengths) == [6, 6, 6]


def test_rollout_stops_stepping_terminated_envs(fbzs):
    pytest.importorskip("torch")
    envs = [_StubEnv(end_at=2), _StubEnv(end_at=None)]
    _ot, lengths, _succ = fbzs.batched_z_rollout(_StubAgent(), envs, np.ones(4), H=5)
    assert envs[0].steps == 2 and envs[1].steps == 5
    assert list(lengths) == [2, 5]


def test_rollout_records_observations_up_to_length(fbzs):
    """The region criterion reads obs_traj[:, :L]; rows past L must stay unset
    so a terminated episode's final state is the LAST REAL one."""
    pytest.importorskip("torch")
    envs = [_StubEnv(end_at=3)]
    ot, lengths, _s = fbzs.batched_z_rollout(_StubAgent(), envs, np.ones(4), H=5)
    L = int(lengths[0])
    assert L == 3
    assert ot[0, L - 1, 0] == pytest.approx(3.0)
    assert np.all(ot[0, L:] == 0.0)


def test_rollout_holds_z_fixed_for_the_whole_episode(fbzs):
    """FB's deployment step: ONE z_r per family/task, no per-step re-inference."""
    pytest.importorskip("torch")
    agent = _StubAgent()
    z = np.arange(4, dtype=np.float64)
    fbzs.batched_z_rollout(agent, [_StubEnv(), _StubEnv()], z, H=4)
    assert len(agent.seen_z) == 4
    for seen in agent.seen_z:
        assert seen.shape == (2, 4)
        assert np.allclose(seen, np.tile(z, (2, 1)))


# ============================ (e) CLI smoke =======================================

def test_help_runs():
    out = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0
    assert "--pool_size" in out.stdout and "--ref-json" in out.stdout


def test_parser_defaults_match_the_density_protocol(fbzs):
    d = vars(fbzs.build_parser().parse_args(["--run", "/x"]))
    assert d["pool_size"] == 16384 and d["episodes"] == 25
    assert d["checkpoint_step"] == 500000 and d["seed"] == 0
    assert d["families"] == "dirac,dense,region"
    assert d["regions"] == 4 and d["region_min_pos"] == 5
    assert d["relabel_size"] == 50000 and d["horizon"] == 0
