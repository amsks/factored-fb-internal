"""The fb_tdj parity arm's missing piece: OGBench reward-relabel + reward-inference eval
(docs/notes/2026-09-03-fb-vs-tdjepa.md #7), transcribed from td_jepa's protocol
(~/git/baselines/td_jepa/metamotivo/evaluations/ogbench.py:90-110,
~/git/baselines/td_jepa/metamotivo/envs/ogbench.py:30-39).

Four things are pinned here:
  1. `cube_relabel_reward` against a hand-computed case, single and double cube.
  2. `OGBenchAdapter.relabel_infer_batch` reads the NEXT observation's stored qpos (never the
     dataset's own zeroed 'rewards' column) against the requested task's goal_xyzs.
  3. `evaluate(..., any_step_success=True)` scores a flickering success trajectory correctly,
     and is a no-op (bit-identical) when left at its default False.
  4. End to end through impls/main.py: the new eval_relabel/task<N> route runs ALONGSIDE the
     existing goal-conditioned route (not instead of it), infers a genuinely PER-TASK z, and
     is entirely absent -- not merely inert -- when cfg['eval_relabel_size'] is unset.

Run standalone on CPU (running all of tests/impls in one process core-dumps):

    PYTHONPATH=$PWD JAX_PLATFORMS=cpu python -m pytest tests/impls/test_ogbench_relabel.py -q \
        -p no:cacheprovider
"""
import csv
import os
import types

import numpy as np
import pytest

import jax.numpy as jnp

from env.ogbench import CUBE_QPOS_OFFSETS, OGBenchAdapter, cube_relabel_reward
from impls.utils.datasets import FBDataset
from impls.utils.evaluation import evaluate as ogbench_evaluate

OBS_DIM, ACT_DIM = 6, 2


# ======================================================================================
# 1. cube_relabel_reward -- hand computed.
# ======================================================================================
def test_single_cube_hand_computed():
    goal = np.array([[0.5, 0.0, 0.02]])
    qpos = np.zeros((3, 17))
    qpos[0, 14:17] = [0.5, 0.0, 0.02]     # exact match -> within threshold
    qpos[1, 14:17] = [0.5, 0.0, 0.10]     # 0.08 away -> outside threshold (0.04)
    qpos[2, 14:17] = [0.5, 0.0, 0.05]     # 0.03 away -> within threshold

    r = cube_relabel_reward(qpos, goal)
    np.testing.assert_allclose(r, [0.0, -1.0, 0.0])
    # shift_reward = 1.0 (the fb_tdj protocol) turns this into a 0/1 cube-at-goal indicator.
    np.testing.assert_allclose(r + 1.0, [1.0, 0.0, 1.0])


def test_double_cube_hand_computed():
    goal = np.array([[0.5, 0.0, 0.02], [0.3, 0.1, 0.02]])
    qpos = np.zeros((3, 24))
    qpos[0, 14:17], qpos[0, 21:24] = goal[0], goal[1]                    # both placed
    qpos[1, 14:17], qpos[1, 21:24] = goal[0], goal[1] + np.array([0.1, 0, 0])  # cube 1 off
    qpos[2, 14:17] = goal[0] + np.array([0.2, 0, 0])                     # both off
    qpos[2, 21:24] = goal[1] + np.array([0.2, 0, 0])

    r = cube_relabel_reward(qpos, goal)
    np.testing.assert_allclose(r, [0.0, -1.0, -2.0])


def test_threshold_is_strict_less_than():
    goal = np.array([[0.0, 0.0, 0.0]])
    qpos = np.zeros((2, 17))
    qpos[0, 14:17] = [0.04, 0.0, 0.0]   # exactly at threshold -> NOT a success (strict <)
    qpos[1, 14:17] = [0.0399, 0.0, 0.0]  # just inside -> success
    r = cube_relabel_reward(qpos, goal)
    np.testing.assert_allclose(r, [-1.0, 0.0])


def test_offsets_match_td_jepa():
    assert CUBE_QPOS_OFFSETS[:2] == (14, 21)


def test_rejects_qpos_too_narrow_for_the_requested_cube_count():
    goal = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # 2 cubes -> needs col 21:24
    qpos = np.zeros((1, 20))
    with pytest.raises(AssertionError, match="qpos is"):
        cube_relabel_reward(qpos, goal)


def test_rejects_unsupported_cube_count():
    with pytest.raises(AssertionError, match="1-4 cubes"):
        cube_relabel_reward(np.zeros((1, 50)), np.zeros((5, 3)))


# ======================================================================================
# 2. OGBenchAdapter.relabel_infer_batch -- reads the NEXT obs's stored qpos, per task.
# ======================================================================================
def _fake_train_ds(n=24, obs_dim=OBS_DIM, qpos_dim=24, seed=0):
    rng = np.random.default_rng(seed)
    observations = rng.standard_normal((n, obs_dim)).astype(np.float32)
    qpos = rng.uniform(-0.5, 0.5, size=(n, qpos_dim)).astype(np.float64)
    valids = np.ones(n, np.float32)
    valids[-1] = 0.0   # trajectory boundary, so get_random_idxs never draws the last row
    return FBDataset.create(
        observations=observations, actions=np.zeros((n, ACT_DIM), np.float32),
        qpos=qpos, valids=valids, terminals=1.0 - valids,
        rewards=np.zeros(n, np.float32))


def _fake_env(task_infos):
    return types.SimpleNamespace(unwrapped=types.SimpleNamespace(task_infos=task_infos))


def test_relabel_infer_batch_reads_next_obs_qpos_against_the_task_goal():
    ds = _fake_train_ds()
    goal_xyzs = np.array([[0.1, 0.2, 0.3]])
    env = _fake_env([{"goal_xyzs": goal_xyzs}])

    next_obs, rewards = OGBenchAdapter().relabel_infer_batch(ds, env, 1, 12, shift_reward=1.0)
    assert next_obs.shape == (12, OBS_DIM)
    assert rewards.shape == (12,)

    observations = np.asarray(ds["observations"])
    qpos = np.asarray(ds["qpos"])
    for row, r in zip(next_obs, rewards):
        # Recover which dataset row this came from (unique random rows -> exact match).
        idx = int(np.nonzero((observations == row).all(axis=1))[0][0])
        want = float(cube_relabel_reward(qpos[idx:idx + 1], goal_xyzs)[0]) + 1.0
        assert r == pytest.approx(want, abs=1e-5), (
            "relabel_infer_batch must relabel the NEXT observation's qpos, not the current "
            "row's or the dataset's own (zeroed) reward column")


def test_relabel_infer_batch_differs_by_task_goal(monkeypatch):
    """With random qpos and threshold 0.04, an UNMATCHED sample draw looks identical
    regardless of which goal it is scored against (every row just reads -1). So this pins a
    draw that is FORCED to include one deliberately-placed at-goal row per task, at an index
    each task's goal can actually reach (relabel reads qpos at idx+1, so a row placed at
    dataset row 0 can NEVER be read -- only rows 1..size-1 can)."""
    ds = _fake_train_ds(n=24)
    goal1 = np.array([[0.1, 0.2, 0.3]])
    goal2 = np.array([[-0.4, 0.5, 0.02]])
    qpos = np.asarray(ds["qpos"]).copy()
    qpos[1, 14:17] = goal1[0]   # reachable via idx=0 -> next_idx=1; matches ONLY task 1
    qpos[2, 14:17] = goal2[0]   # reachable via idx=1 -> next_idx=2; matches ONLY task 2
    ds2 = FBDataset.create(observations=np.asarray(ds["observations"]),
                           actions=np.asarray(ds["actions"]), qpos=qpos,
                           valids=np.asarray(ds["valids"]), terminals=np.asarray(ds["terminals"]),
                           rewards=np.asarray(ds["rewards"]))
    # Deterministic draw that is guaranteed to include idx 0 and 1 -- no dependence on the
    # global RNG state or on a large enough sample to make a miss improbable.
    monkeypatch.setattr(ds2, "get_random_idxs", lambda n: np.arange(min(n, ds2.size - 1)))
    env = _fake_env([{"goal_xyzs": goal1}, {"goal_xyzs": goal2}])

    _, r1 = OGBenchAdapter().relabel_infer_batch(ds2, env, 1, 8, shift_reward=1.0)
    _, r2 = OGBenchAdapter().relabel_infer_batch(ds2, env, 2, 8, shift_reward=1.0)
    assert not np.array_equal(r1, r2)
    # idxs = [0, 1, ...] -> next_idxs = [1, 2, ...]: reward[0] reads dataset row 1 (task 1's
    # exact match), reward[1] reads dataset row 2 (task 2's exact match).
    assert r1[0] == pytest.approx(1.0)   # next_obs row 1 IS task 1's goal (shift_reward=1)
    assert r2[0] == pytest.approx(0.0)   # ... but not task 2's (unmatched: -1 + shift 1 = 0)
    assert r2[1] == pytest.approx(1.0)   # next_obs row 2 IS task 2's goal
    assert r1[1] == pytest.approx(0.0)   # ... but not task 1's


def test_relabel_infer_batch_requires_qpos():
    ds = FBDataset.create(observations=np.zeros((4, OBS_DIM), np.float32),
                          rewards=np.zeros(4, np.float32))
    env = _fake_env([{"goal_xyzs": np.array([[0.0, 0.0, 0.0]])}])
    with pytest.raises(AssertionError, match="qpos"):
        OGBenchAdapter().relabel_infer_batch(ds, env, 1, 4)


def test_relabel_infer_batch_requires_goal_xyzs_on_the_task():
    ds = _fake_train_ds()
    # A task_info from no supported family (cube/antmaze/puzzle/scene keys all absent) must
    # fail loudly, naming the cube key among the supported ones.
    env = _fake_env([{"goal_foo": np.zeros(2)}])
    with pytest.raises(AssertionError, match="goal_xyzs"):
        OGBenchAdapter().relabel_infer_batch(ds, env, 1, 4)


# ======================================================================================
# 3. any-step success accounting.
# ======================================================================================
class _ZeroActionAgent:
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        return np.zeros(ACT_DIM, np.float32)


class _FlickerEnv:
    """success flips True at step 2 of 4, then False again -- final-step success is False,
    any-step success is True. terminate_at_goal=False semantics (episode runs to the end),
    which is the case the any-step reducer exists for."""

    def __init__(self):
        self.t = 0

    def reset(self, seed=None, options=None):
        self.t = 0
        return np.zeros(OBS_DIM, np.float32), {"success": False, "goal": None}

    def step(self, action):
        self.t += 1
        succ = self.t == 2
        done = self.t >= 4
        return np.zeros(OBS_DIM, np.float32), 0.0, done, False, {"success": succ}

    def render(self):
        return np.zeros((2, 2, 3), np.uint8)


def test_any_step_success_catches_a_mid_episode_flicker():
    stats, _, _ = ogbench_evaluate(_ZeroActionAgent(), _FlickerEnv(), config={},
                                   num_eval_episodes=1, any_step_success=True)
    assert stats["success"] == 1.0


def test_final_step_success_is_the_default_and_misses_the_flicker():
    stats, _, _ = ogbench_evaluate(_ZeroActionAgent(), _FlickerEnv(), config={},
                                   num_eval_episodes=1)  # any_step_success defaults False
    assert stats["success"] == 0.0


def test_any_step_success_off_is_bit_identical_to_no_such_parameter():
    """The parameter must not perturb the stats dict in any way when left at its default --
    this is the regression guard for 'the goal-conditioned route is unaffected'."""
    a = ogbench_evaluate(_ZeroActionAgent(), _FlickerEnv(), config={}, num_eval_episodes=1)
    b = ogbench_evaluate(_ZeroActionAgent(), _FlickerEnv(), config={}, num_eval_episodes=1,
                         any_step_success=False)
    assert dict(a[0]) == dict(b[0])


# ======================================================================================
# 4. add_info gating on cfg['eval_relabel_size'].
# ======================================================================================
def test_make_env_and_datasets_forces_add_info_when_eval_relabel_size_is_set(monkeypatch):
    import env.ogbench as ogb

    calls = []

    def fake_make(dataset_name, frame_stack, add_info=False):
        calls.append(add_info)
        ds = FBDataset.create(observations=np.zeros((2, OBS_DIM), np.float32))
        return types.SimpleNamespace(_max_episode_steps=None, spec=None), ds, ds

    monkeypatch.setattr(ogb, "make_env_and_datasets", fake_make)

    ogb.OGBenchAdapter().make_env_and_datasets({"env_name": "x"})
    assert calls == [False]
    calls.clear()

    ogb.OGBenchAdapter().make_env_and_datasets({"env_name": "x", "eval_relabel_size": 10_000})
    assert calls == [True]
    calls.clear()

    # measure_qpos_slices already forced it; eval_relabel_size must not turn that OFF.
    ogb.OGBenchAdapter().make_env_and_datasets({"env_name": "x", "measure_qpos_slices": "14:17"})
    assert calls == [True]


# ======================================================================================
# 5. End to end through impls/main.py: a new mode ALONGSIDE the goal-conditioned route.
# ======================================================================================
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
    )
    cfg.update(extra)
    return cfg


class _RelabelWiringAdapter:
    """Stub OGBench-shaped adapter: real cube-relabel math (delegates to the real
    OGBenchAdapter.relabel_infer_batch), fake env/dataset. Records every `evaluate()` call's
    (task, any_step_success, z) so the test can prove the two eval routes are DISTINCT and
    BOTH ran, without needing a real OGBench install."""

    name = "ogbench"

    def __init__(self, n_tasks=2, n_rows=40):
        rng = np.random.default_rng(2)
        self.task_infos = [{"goal_xyzs": np.array([[0.1 * (t + 1), 0.0, 0.02]])}
                           for t in range(n_tasks)]
        self._env = _fake_env(self.task_infos)
        qpos = rng.uniform(-0.5, 0.5, size=(n_rows, 24)).astype(np.float64)
        for t, info in enumerate(self.task_infos):
            # relabel reads qpos at idx+1 (the NEXT observation), so a row placed at index 0
            # is never reachable -- offset by 1 so every task's exact-match row can be drawn.
            qpos[t + 1, 14:17] = info["goal_xyzs"][0]
        observations = rng.standard_normal((n_rows, OBS_DIM)).astype(np.float32)
        valids = np.ones(n_rows, np.float32)
        valids[-1] = 0.0
        self._train_ds = FBDataset.create(
            observations=observations, actions=np.zeros((n_rows, ACT_DIM), np.float32),
            qpos=qpos, valids=valids, terminals=1.0 - valids,
            rewards=np.zeros(n_rows, np.float32))
        # Deterministic draw (no dependence on global RNG state or a large-enough sample to
        # make missing a specific placed row improbable): every relabel_infer_batch call
        # reads idx 0..num_samples-1, which is exactly where the exact-match rows above land.
        self._train_ds.get_random_idxs = lambda n: np.arange(min(n, self._train_ds.size - 1))
        self.evaluate_calls = []
        self.relabel_calls = []

    def make_env_and_datasets(self, cfg):
        return self._env, self._train_ds, self._train_ds

    def eval_tasks(self, env, cfg):
        return list(range(1, len(self.task_infos) + 1))

    def evaluate(self, agent, env, task, cfg, num_episodes, num_video, any_step_success=False):
        cond = dict(getattr(agent, "cond_eval", {}))
        z = np.asarray(cond["z"]) if "z" in cond else None
        self.evaluate_calls.append(dict(task=task, any_step_success=any_step_success, z=z))
        return {"success": 0.0}, []

    def relabel_infer_batch(self, train_ds, env, task_id, num_samples, shift_reward=1.0):
        self.relabel_calls.append(task_id)
        return OGBenchAdapter().relabel_infer_batch(
            train_ds, env, task_id, num_samples, shift_reward=shift_reward)


def _run_main(tmp_path, monkeypatch, cfg, adapter, steps=2, **over):
    import impls.main as m

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(m, "make_config", lambda *a, **k: dict(cfg))
    monkeypatch.setattr(m, "get_adapter", lambda c: adapter)
    monkeypatch.setattr(m, "save_agent", lambda agent, d, i: None)

    m.FLAGS(["test"], known_only=True)
    defaults = dict(critic="fb", actor="flowbc", domain="tiny", override=[], seed=0,
                    train_steps=steps, log_interval=10 ** 9, eval_interval=10 ** 9,
                    save_interval=10 ** 9, eval_episodes=1, num_video=0,
                    save_dir=str(tmp_path), restore_path="", warm_start_path="",
                    agg_dir=None, infer_cond_episodes=0, online=False,
                    eval_on_inferred_cond=False)
    for k, v in {**defaults, **over}.items():
        monkeypatch.setattr(m.FLAGS, k, v)
    m.main(None)
    return os.path.join(tmp_path, "tiny_fb_flowbc_seed_0")


def test_eval_relabel_route_runs_alongside_the_goal_conditioned_route(tmp_path, monkeypatch):
    adapter = _RelabelWiringAdapter(n_tasks=2)
    cfg = _fb_cfg(eval_relabel_size=20, eval_relabel_shift=1.0, eval_relabel_episodes=3)
    run_dir = _run_main(tmp_path, monkeypatch, cfg, adapter)

    assert adapter.relabel_calls == [1, 2], "relabel must run once per REAL task_id"

    normal = [c for c in adapter.evaluate_calls if not c["any_step_success"]]
    relabel = [c for c in adapter.evaluate_calls if c["any_step_success"]]
    assert {c["task"] for c in normal} == {1, 2}, "the existing goal-conditioned route must still run"
    assert {c["task"] for c in relabel} == {1, 2}, "the new relabel route must run for every task"

    # The relabel route's z is genuinely PER TASK (different goal_xyzs -> different reward
    # relabel -> different inferred z); the normal route's z is eval_cond0 (zero) for both,
    # since --infer_cond_episodes is off.
    z1 = next(c["z"] for c in relabel if c["task"] == 1)
    z2 = next(c["z"] for c in relabel if c["task"] == 2)
    assert not np.allclose(z1, z2), "relabel z must differ by task -- the whole point of #7"
    for c in normal:
        assert np.allclose(c["z"], 0.0), "the goal-conditioned route's z must be untouched (eval_cond0)"

    # The two routes land in SEPARATE files (CsvLogger freezes its header on the first row,
    # so a shared file would silently drop the relabel route's extra witness columns -- see
    # the comment at csv_eval_relabel's construction in impls/main.py).
    with open(os.path.join(run_dir, "eval.csv")) as f:
        normal_rows = list(csv.DictReader(f))
    with open(os.path.join(run_dir, "eval_relabel.csv")) as f:
        relabel_rows = list(csv.DictReader(f))
    assert len(normal_rows) == 2    # step 1, two tasks (eval_interval huge -> one eval point)
    assert len(relabel_rows) == 2
    assert all(r.get("inferred_cond") not in (None, "") for r in relabel_rows)
    assert all("inferred_cond" not in r for r in normal_rows)


def test_eval_relabel_absent_when_the_key_is_unset(tmp_path, monkeypatch):
    """Regression guard: with eval_relabel_size unset, relabel_infer_batch must never be
    called and no eval_relabel row must appear -- the new mode must be truly OFF, not just
    zero-sized."""
    adapter = _RelabelWiringAdapter(n_tasks=2)
    cfg = _fb_cfg()   # no eval_relabel_size key at all
    run_dir = _run_main(tmp_path, monkeypatch, cfg, adapter)

    assert adapter.relabel_calls == []
    assert all(not c["any_step_success"] for c in adapter.evaluate_calls)

    with open(os.path.join(run_dir, "eval.csv")) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2   # one row per task, goal-conditioned route only

    relabel_path = os.path.join(run_dir, "eval_relabel.csv")
    assert not os.path.exists(relabel_path) or os.path.getsize(relabel_path) == 0, (
        "the relabel route must be truly absent (no rows), not merely an empty-but-created "
        "file, when eval_relabel_size is unset")


def test_eval_relabel_size_zero_is_also_off(tmp_path, monkeypatch):
    adapter = _RelabelWiringAdapter(n_tasks=2)
    cfg = _fb_cfg(eval_relabel_size=0)
    _run_main(tmp_path, monkeypatch, cfg, adapter)
    assert adapter.relabel_calls == []


def test_unsupported_adapter_raises_a_named_error(tmp_path, monkeypatch):
    """An adapter without relabel_infer_batch (every non-cube domain) must fail loudly, not
    silently skip the requested relabel eval."""
    class _NoRelabelAdapter(_RelabelWiringAdapter):
        relabel_infer_batch = None

    adapter = _NoRelabelAdapter(n_tasks=1)
    cfg = _fb_cfg(eval_relabel_size=10)
    with pytest.raises(NotImplementedError, match="relabel_infer_batch"):
        _run_main(tmp_path, monkeypatch, cfg, adapter)


def test_relabel_unwraps_a_per_critic_dataset_wrapper():
    """REGRESSION: eval_relabel_size crashed every HILP arm at launch.

    `impls/main.py` hands the eval the dataset AFTER `make_dataset` wraps it in the critic's
    sampler. Those wrappers implement `sample` but not column access, so

        assert "qpos" in train_ds
        TypeError: argument of type 'HILPFutureDataset' is not iterable

    killed all 12 hilp_tdjc runs seconds after launch, while fb_tdj was fine because its
    dataset_class leaves the base unwrapped. `_column_backed` walks `.dataset` down to the
    real columns.
    """
    import numpy as np

    from env.ogbench import OGBenchAdapter

    class _Base:
        def __init__(self):
            self._cols = {"observations": np.zeros((4, 3)), "qpos": np.zeros((4, 2))}
        def __contains__(self, k):
            return k in self._cols
        def __getitem__(self, k):
            return self._cols[k]

    class _Wrapper:                      # no __contains__, exactly like HILPFutureDataset
        def __init__(self, base):
            self.dataset = base

    base = _Base()
    assert OGBenchAdapter._column_backed(base) is base
    assert OGBenchAdapter._column_backed(_Wrapper(base)) is base
    assert OGBenchAdapter._column_backed(_Wrapper(_Wrapper(base))) is base


def test_column_backed_terminates_on_a_self_referential_wrapper():
    """It must not spin if a wrapper points at itself."""
    from env.ogbench import OGBenchAdapter

    class _Loop:
        pass

    a = _Loop()
    a.dataset = a
    assert OGBenchAdapter._column_backed(a) is a
