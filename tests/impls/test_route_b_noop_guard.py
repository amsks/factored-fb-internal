"""Route B (the eval_relabel reward-inference eval in impls/main.py) must not SCORE a rollout
whose inference was a no-op.

Run on its own, on CPU (running all of tests/impls in one process core-dumps):

    JAX_PLATFORMS=cpu python -m pytest tests/impls/test_route_b_noop_guard.py -q -p no:cacheprovider

THE BUG. Route B calls `agent.infer_eval_cond(obs, rewards)`, wraps the result in
`InferredCondAgent` and evaluates, but never asked whether inference did anything. For the
goal-conditioned density_fb critic (measure_density_z off) `infer_cond` returns the zero-goal
PLACEHOLDER, so the policy was rolled on goal=0 and the success rate landed in
eval_relabel.csv as if it were a zero-shot number (hub rows carried inferred_cond_norm=0.0).
`infer_cond_noop_reason` already detects exactly this, but was only wired into the
--infer_cond_episodes route.

Pinned here:
  1. A no-op inference is REFUSED as a score: the row is written with success=NaN,
     route_b_valid=0 and an `invalid_reason` naming the critic; a loud warning fires; no
     rollout is scored; and the agg jsonl (the campaign daemon's input) gets nothing.
  2. A live inference (fb, whose z genuinely changes) passes through unchanged: success is
     whatever evaluate() returned, route_b_valid=1, invalid_reason empty.
"""
import csv
import glob
import math
import os
import types

import numpy as np
import pytest

from env.ogbench import OGBenchAdapter
from impls.utils.config_utils import load_config
from impls.utils.datasets import FBDataset

ACT_DIM = 2
# density_fb's default hit spec is obs[19:22], so the stub observation must be at least
# that wide; the FB half of the file uses the same width so the two stubs are identical.
OBS_DIM = 40
SUCCESS_FROM_EVALUATE = 0.75    # a recognisable, non-zero score the stub evaluator returns


def _fake_env(task_infos):
    return types.SimpleNamespace(unwrapped=types.SimpleNamespace(task_infos=task_infos))


class _RelabelAdapter:
    """OGBench-shaped stub: the REAL relabel math, a fake env/dataset, and an `evaluate`
    that returns a fixed non-zero success so a scored-vs-refused row is unmistakable."""

    name = "ogbench"

    def __init__(self, n_tasks=2, n_rows=40):
        rng = np.random.default_rng(2)
        self.task_infos = [{"goal_xyzs": np.array([[0.1 * (t + 1), 0.0, 0.02]])}
                           for t in range(n_tasks)]
        self._env = _fake_env(self.task_infos)
        qpos = rng.uniform(-0.5, 0.5, size=(n_rows, 24)).astype(np.float64)
        for t, info in enumerate(self.task_infos):
            qpos[t + 1, 14:17] = info["goal_xyzs"][0]     # relabel reads qpos at idx+1
        observations = rng.standard_normal((n_rows, OBS_DIM)).astype(np.float32)
        valids = np.ones(n_rows, np.float32)
        valids[-1] = 0.0
        self._train_ds = FBDataset.create(
            observations=observations,
            actions=np.tanh(rng.standard_normal((n_rows, ACT_DIM))).astype(np.float32),
            qpos=qpos, valids=valids, terminals=1.0 - valids,
            rewards=np.zeros(n_rows, np.float32))
        self._train_ds.get_random_idxs = lambda n: np.arange(min(n, self._train_ds.size - 1))
        self.evaluate_calls = []

    def make_env_and_datasets(self, cfg):
        return self._env, self._train_ds, self._train_ds

    def eval_tasks(self, env, cfg):
        return list(range(1, len(self.task_infos) + 1))

    def evaluate(self, agent, env, task, cfg, num_episodes, num_video, any_step_success=False):
        self.evaluate_calls.append(dict(task=task, any_step_success=any_step_success))
        return {"success": SUCCESS_FROM_EVALUATE}, []

    def relabel_infer_batch(self, train_ds, env, task_id, num_samples, shift_reward=1.0):
        return OGBenchAdapter().relabel_infer_batch(
            train_ds, env, task_id, num_samples, shift_reward=shift_reward)


def _density_fb_cfg(**extra):
    """The shipped goal-conditioned density_fb (index OFF), shrunk to test size, with Route B
    switched on. Mirrors tests/impls/test_density_fb_reward_index.py::_agent."""
    cfg = load_config("density_fb", "flowbc", "cube_single")
    # env_name names the run dir (impls/main.py::get_exp_name); the adapter is stubbed.
    cfg = {**cfg, "env_name": "tiny",
           "batch_size": 16, "measure_density_dim": 8, "value_hidden_dims": [32, 32],
           "measure_density_z": False, "measure_density_z_eval_pool_m": 8,
           "eval_relabel_size": 20, "eval_relabel_shift": 1.0, "eval_relabel_episodes": 3}
    cfg.update(extra)
    return cfg


def _fb_cfg(**extra):
    cfg = dict(
        env_name="tiny", task="b2d-tiny", adapter="unified", horizon=8, batch_size=16,
        z_dim=8, L_dim=8, num_parallel=2, discount=0.99,
        f_target_tau=0.005, b_target_tau=0.005, ortho_coef=1.0, train_goal_ratio=0.5,
        fb_pessimism_penalty=0.0, norm_z=True, lr_f=1e-4, lr_b=1e-4, lr_actor=1e-4,
        forward=dict(hidden_dim=32, hidden_layers=2, embedding_layers=2),
        backward=dict(hidden_dim=32, hidden_layers=4, norm=True),
        left_encoder=dict(hidden_dim=32, hidden_layers=4, norm=True),
        actor=dict(type="flow", hidden_dim=32, hidden_layers=2, embedding_layers=2,
                   bc_coeff=3.0, q_coeff=1.0, flow_steps=5, lr_actor_vf=3e-4,
                   flow_vf_hidden_dim=32, flow_vf_hidden_layers=4),
        eval_relabel_size=20, eval_relabel_shift=1.0, eval_relabel_episodes=3,
    )
    cfg.update(extra)
    return cfg


def _run_main(tmp_path, monkeypatch, cfg, adapter, critic, actor="flowbc", steps=2, **over):
    import impls.main as m

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(m, "make_config", lambda *a, **k: dict(cfg))
    monkeypatch.setattr(m, "get_adapter", lambda c: adapter)
    monkeypatch.setattr(m, "save_agent", lambda agent, d, i: None)

    m.FLAGS(["test"], known_only=True)
    defaults = dict(critic=critic, actor=actor, domain="tiny", override=[], seed=0,
                    train_steps=steps, log_interval=10 ** 9, eval_interval=10 ** 9,
                    save_interval=10 ** 9, eval_episodes=1, num_video=0,
                    save_dir=str(tmp_path), restore_path="", warm_start_path="",
                    agg_dir=str(tmp_path / "agg"), infer_cond_episodes=0, online=False,
                    eval_on_inferred_cond=False)
    for k, v in {**defaults, **over}.items():
        monkeypatch.setattr(m.FLAGS, k, v)
    m.main(None)
    return os.path.join(tmp_path, f"tiny_{critic}_{actor}_seed_0")


def _relabel_rows(run_dir):
    with open(os.path.join(run_dir, "eval_relabel.csv")) as f:
        rows = list(csv.DictReader(f))
    assert rows, "Route B wrote no rows at all"
    return rows


# ======================================================================================
# 1. density_fb with the index OFF: inference is a placeholder -> the row must be refused.
# ======================================================================================
def test_route_b_refuses_to_score_a_noop_inference(tmp_path, monkeypatch):
    adapter = _RelabelAdapter(n_tasks=2)
    with pytest.warns(UserWarning, match="DensityFBCritic"):
        run_dir = _run_main(tmp_path, monkeypatch, _density_fb_cfg(), adapter, "density_fb")

    rows = _relabel_rows(run_dir)
    assert len(rows) == 2, "one Route B row per task, refused or not"
    for r in rows:
        assert "route_b_valid" in r and "invalid_reason" in r, \
            f"the row carries no validity flag -- it would be read as a real score: {r}"
        assert float(r["route_b_valid"]) == 0.0
        assert "DensityFBCritic" in r["invalid_reason"], r["invalid_reason"]
        assert math.isnan(float(r["success"])), \
            f"a goal=0 rollout was SCORED as success={r['success']}"
        # the goal-conditioned placeholder is the zero vector; the witness must say so
        assert float(r["inferred_cond_norm"]) == 0.0

    # The refused rows must never have been rolled: only the goal-conditioned route
    # (any_step_success=False) reached the evaluator.
    assert all(not c["any_step_success"] for c in adapter.evaluate_calls), \
        "Route B still rolled the policy on a no-op conditioning"
    # ... and nothing reached the campaign aggregator for the relabel group.
    assert glob.glob(os.path.join(tmp_path, "agg", "*_relabel__s0.jsonl")) == [], \
        "a refused Route B row was still appended to the agg jsonl (counted as scored)"


# ======================================================================================
# 2. fb: inference is live -> the row passes through exactly as before.
# ======================================================================================
def test_route_b_passes_a_live_inference_through_unchanged(tmp_path, monkeypatch):
    adapter = _RelabelAdapter(n_tasks=2)
    run_dir = _run_main(tmp_path, monkeypatch, _fb_cfg(), adapter, "fb")

    rows = _relabel_rows(run_dir)
    assert len(rows) == 2
    for r in rows:
        assert float(r["route_b_valid"]) == 1.0
        assert r["invalid_reason"] == ""
        assert float(r["success"]) == SUCCESS_FROM_EVALUATE
        assert float(r["inferred_cond_norm"]) > 0.0
    assert {c["task"] for c in adapter.evaluate_calls if c["any_step_success"]} == {1, 2}
    agg = glob.glob(os.path.join(tmp_path, "agg", "*_relabel__s0.jsonl"))
    assert len(agg) == 1, "a live Route B row must still be aggregated"
    assert len(open(agg[0]).read().strip().split("\n")) == 2


# ======================================================================================
# 3. The detector itself, in isolation (no main loop): zero-norm cond is refused even when
#    the noop detector's bit-identity check is not tripped.
# ======================================================================================
def test_route_b_invalid_reason_flags_a_zero_norm_cond():
    import jax.numpy as jnp
    import impls.main as m

    class _Critic:
        pass

    class _Agent:
        critic = _Critic()
        cond_eval = {"z": jnp.zeros((8,))}

    wrapped = m.InferredCondAgent(_Agent())
    pre = {"z": jnp.ones((8,))}      # cond DID change, so the bit-identity check passes...
    reason = m.route_b_invalid_reason(pre, wrapped, jnp.ones((8,)))
    assert reason is not None and "_Critic" in reason and "norm" in reason

    live = m.InferredCondAgent(types.SimpleNamespace(critic=_Critic(),
                                                     cond_eval={"z": jnp.ones((8,))}))
    assert m.route_b_invalid_reason({"z": jnp.zeros((8,))}, live, jnp.ones((8,))) is None
