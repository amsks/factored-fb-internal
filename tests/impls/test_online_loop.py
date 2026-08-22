"""Online (env-interactive) training mode, and the gate that keeps the OFFLINE path unchanged.

Run this file on its own, on CPU (running all of tests/impls in one process core-dumps):

    PYTHONPATH=$PWD JAX_PLATFORMS=cpu python -m pytest tests/impls/test_online_loop.py -q \
        -p no:cacheprovider

The first test is the important one. `--online` defaults to False and the whole online path
lives behind it, so a run launched today must produce BIT-IDENTICAL parameters to the loop as
it was before online mode existed. That reference loop is transcribed literally at the bottom
of `_reference_offline_params`; if impls/main.py's offline branch ever drifts from it, this
fails.

`env.online` is imported LAZILY inside each online test rather than at module scope, so the
identity test (which runs first) can also assert that an offline run never pulls the module
in at all.
"""
import os
import subprocess
import sys
import types

import numpy as np
import pytest

import jax
import jax.numpy as jnp

SEED = 0
OBS_DIM, ACT_DIM, HORIZON = 6, 2, 20


# --------------------------------------------------------------------------------------
# A tiny env + a stub adapter. No MuJoCo, no GPU, no dataset on disk.
# --------------------------------------------------------------------------------------
class TinyEnv:
    """Deterministic 6-D toy env with the gymnasium 5-tuple step contract the adapters use.

    reset(seed=) -> (obs, info); step(a) -> (obs, reward, terminated, truncated, info).
    That is exactly what env/unified.py::seeded_reset and ::evaluate drive, so the online
    collector talks to it through the same code path it uses on a real task.
    """

    def __init__(self, horizon=HORIZON):
        self.horizon = horizon
        self.t = 0
        self.obs = np.zeros(OBS_DIM, np.float32)
        self.n_resets = 0
        self.n_steps = 0

    def reset(self, seed=None, options=None):
        self.n_resets += 1
        rng = np.random.default_rng(0 if seed is None else int(seed) % (2 ** 31))
        self.obs = rng.uniform(-1.0, 1.0, OBS_DIM).astype(np.float32)
        self.t = 0
        # `goal` is what the ogbench-style reset path reads out of info.
        return self.obs.copy(), {"success": False, "goal": np.zeros(OBS_DIM, np.float32)}

    def step(self, action):
        self.n_steps += 1
        a = np.asarray(action, np.float32).reshape(-1)[:ACT_DIM]
        nxt = self.obs.copy()
        nxt[:ACT_DIM] = np.clip(nxt[:ACT_DIM] + 0.1 * a, -1.0, 1.0)
        nxt[ACT_DIM:] = 0.9 * nxt[ACT_DIM:] + 0.01
        self.obs = nxt.astype(np.float32)
        self.t += 1
        reward = float(-np.linalg.norm(self.obs[:ACT_DIM]))
        success = bool(reward > -0.05)
        return self.obs.copy(), reward, False, self.t >= self.horizon, {"success": success}

    def render(self):
        return np.zeros((4, 4, 3), np.uint8)


def _offline_arrays(n_episodes=8, ep_len=HORIZON, seed=0):
    """A flat offline dataset in the layout env/unified.py::load_unified produces:
    `terminals` marks each episode's LAST row (a trajectory boundary)."""
    rng = np.random.default_rng(seed)
    n = n_episodes * ep_len
    term = np.zeros((n_episodes, ep_len), np.float32)
    term[:, -1] = 1.0
    return dict(
        observations=rng.standard_normal((n, OBS_DIM)).astype(np.float32),
        actions=np.tanh(rng.standard_normal((n, ACT_DIM))).astype(np.float32),
        next_observations=rng.standard_normal((n, OBS_DIM)).astype(np.float32),
        rewards=rng.standard_normal(n).astype(np.float32),
        terminals=term.reshape(-1),
    )


class StubAdapter:
    """Minimal DomainAdapter. `name` selects which reset path the online collector uses."""

    def __init__(self, name="unified"):
        self.name = name
        self.env = None
        self._horizon = HORIZON
        self._goal = np.zeros(OBS_DIM, np.float32)
        self.eval_calls = 0

    def make_env_and_datasets(self, cfg):
        from impls.utils.datasets import FBDataset
        self.env = TinyEnv()
        data = _offline_arrays()
        return self.env, FBDataset.create(**data), FBDataset.create(**data)

    def eval_tasks(self, env, cfg):
        return [None]

    def evaluate(self, agent, env, task, cfg, num_episodes, num_video):
        # Constant and env-free: eval is NOT what these tests are about, and a real rollout
        # here would make the identity comparison depend on the policy.
        self.eval_calls += 1
        return {"success": 0.0}, []


def _gciql_cfg():
    """gciql x ddpgbc over GCDataset -- the goal-conditioned case, so the goal keys are
    exercised. Same shape as tests/impls/test_resume.py::_cfg plus the dataset knobs."""
    return dict(
        env_name="tiny", task="b2d-tiny", adapter="unified", horizon=HORIZON,
        batch_size=16, dataset_class="GCDataset",
        value_p_curgoal=0.2, value_p_trajgoal=0.5, value_p_randomgoal=0.3,
        actor_p_curgoal=0.0, actor_p_trajgoal=1.0, actor_p_randomgoal=0.0,
        value_geom_sample=True, actor_geom_sample=False,
        value_hidden_dims=[32, 32], layer_norm=True, discount=0.99, expectile=0.9,
        tau=0.005, lr=3e-4, lr_actor=3e-4, gc_negative=True, q_bound=0.0, z_dim=8,
        actor=dict(hidden_dims=[32, 32], alpha=3.0, const_std=True, layer_norm=False),
    )


def _fb_cfg():
    """fb x flowbc -- the z-conditioned case, whose offline dataset is a plain FBDataset
    emitting FIVE keys, not GCDataset's nine."""
    return dict(
        env_name="tiny", task="b2d-tiny", adapter="unified", horizon=HORIZON, batch_size=16,
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


def _flat_params(agent):
    leaves = []
    for g in sorted(agent.net):
        leaves += [np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(agent.net[g].params)]
    return np.concatenate(leaves)


# --------------------------------------------------------------------------------------
# main() driver
# --------------------------------------------------------------------------------------
def _run_main(tmp_path, monkeypatch, cfg, adapter, critic, actor, steps, **flag_overrides):
    """Run impls/main.py::main end to end against the stub adapter, returning the agent it
    would have checkpointed."""
    import impls.main as m

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(m, "make_config", lambda *a, **k: dict(cfg))
    monkeypatch.setattr(m, "get_adapter", lambda c: adapter)
    captured = []
    monkeypatch.setattr(m, "save_agent", lambda agent, d, i: captured.append(agent))

    m.FLAGS(["test"], known_only=True)          # absl: mark the flags parsed
    defaults = dict(critic=critic, actor=actor, domain="tiny", override=[], seed=SEED,
                    train_steps=steps, log_interval=max(1, steps // 2),
                    eval_interval=10 ** 9, save_interval=steps, eval_episodes=1,
                    num_video=0, save_dir=str(tmp_path), restore_path="", warm_start_path="",
                    agg_dir=None, infer_cond_episodes=0, online=False)
    for k, v in {**defaults, **flag_overrides}.items():
        monkeypatch.setattr(m.FLAGS, k, v)

    m.main(None)
    assert captured, "main() did not reach the save_interval checkpoint"
    return captured[-1]


def _reference_offline_params(cfg, adapter, critic, actor, steps):
    """The offline loop EXACTLY as it stood before online mode was added (impls/main.py at
    e579555, lines 153/183-198). Anything main() does differently on the offline path shows
    up as a parameter mismatch."""
    from impls.agents import Agent
    from impls.utils.datasets import make_dataset

    np.random.seed(SEED)
    env, train_ds, _ = adapter.make_env_and_datasets(cfg)
    train_ds = make_dataset(train_ds, cfg)
    ex = train_ds.sample(1)
    agent = Agent.create(critic, actor, SEED,
                         jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)
    for _ in range(1, steps + 1):
        batch = train_ds.sample(cfg["batch_size"])
        agent, _ = agent.update(batch)
    return _flat_params(agent)


# ======================================================================================
# 1. THE GATE: the offline path is untouched when --online is off.
# ======================================================================================
def test_offline_default_is_off():
    import impls.main as m
    assert m.FLAGS["online"].default is False


@pytest.mark.parametrize("critic,actor,cfg_fn", [("gciql", "ddpgbc", _gciql_cfg),
                                                 ("fb", "flowbc", _fb_cfg)])
def test_offline_path_is_bit_identical_with_the_flag_off(tmp_path, monkeypatch,
                                                         critic, actor, cfg_fn):
    """Both dataset paths: GCDataset (goal-conditioned) and the plain FBDataset."""
    cfg, steps = cfg_fn(), 12
    got = _flat_params(_run_main(tmp_path, monkeypatch, cfg, StubAdapter(),
                                 critic, actor, steps))
    want = _reference_offline_params(cfg, StubAdapter(), critic, actor, steps)
    np.testing.assert_array_equal(got, want)


def test_offline_run_never_imports_the_online_module(tmp_path, monkeypatch):
    """Belt and braces on the gate above: with the flag off, env/online.py is not merely
    unused, it is never imported. (This test runs before any online test in this file, which
    is why the online tests import env.online lazily.)"""
    assert "env.online" not in sys.modules, \
        "this test must run before anything imports env.online"
    _run_main(tmp_path, monkeypatch, _gciql_cfg(), StubAdapter(), "gciql", "ddpgbc", 3)
    assert "env.online" not in sys.modules, \
        "the offline path imported env/online.py; it must stay inside `if FLAGS.online:`"


def test_importing_main_does_not_import_the_online_module():
    """Same property at module scope, in a clean interpreter."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; import impls.main; "
         "assert 'env.online' not in sys.modules, sorted(k for k in sys.modules if 'online' in k)"],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        env={**os.environ, "PYTHONPATH": os.getcwd(), "JAX_PLATFORMS": "cpu",
             "WANDB_MODE": "disabled"},
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]


# ======================================================================================
# 2. Replay buffer: keys, shapes, episode structure, compaction, seeding.
# ======================================================================================
def _buffer(cfg, capacity=200, required=None):
    from env.online import ReplayBuffer
    return ReplayBuffer(capacity, OBS_DIM, ACT_DIM, cfg, required_keys=required)


def _fill(buf, n_episodes, ep_len=5, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n_episodes):
        for _ in range(ep_len):
            buf.add(rng.standard_normal(OBS_DIM), rng.standard_normal(ACT_DIM),
                    rng.standard_normal(OBS_DIM), float(rng.standard_normal()))
        buf.end_episode()
    return buf


def test_buffer_emits_the_gcdataset_key_set_with_correct_shapes():
    from impls.utils.datasets import GCDataset
    cfg = _gciql_cfg()
    buf = _fill(_buffer(cfg), n_episodes=6)
    assert buf.size == 30 and buf.n_episodes == 6
    assert isinstance(buf.dataset(), GCDataset)

    batch = buf.sample(8)
    for k in ("observations", "actions", "next_observations", "next_actions",
              "rewards", "masks", "terminals", "value_goals", "actor_goals"):
        assert k in batch, f"missing {k}"
    assert batch["observations"].shape == (8, OBS_DIM)
    assert batch["actions"].shape == (8, ACT_DIM)
    assert batch["next_observations"].shape == (8, OBS_DIM)
    assert batch["next_actions"].shape == (8, ACT_DIM)
    assert batch["value_goals"].shape == (8, OBS_DIM)
    assert batch["actor_goals"].shape == (8, OBS_DIM)
    assert batch["rewards"].shape == (8,) and batch["masks"].shape == (8,)
    for k in ("observations", "actions", "next_observations", "value_goals", "actor_goals"):
        assert batch[k].dtype == np.float32, k


def test_buffer_matches_the_offline_batch_key_set_for_a_non_gc_critic():
    """FB's offline dataset is a plain FBDataset: five keys, rewards shaped (B, 1). The
    online buffer must emit the same thing, not GCDataset's superset."""
    from impls.utils.datasets import FBDataset, make_dataset
    cfg = _fb_cfg()
    offline_keys = set(make_dataset(FBDataset.create(**_offline_arrays()), cfg).sample(1))
    batch = _fill(_buffer(cfg, required=tuple(offline_keys)), n_episodes=6).sample(8)
    assert set(batch) == offline_keys
    assert batch["rewards"].shape == (8, 1)
    assert batch["terminals"].shape == (8,)


def test_goal_keys_are_real_hindsight_goals_not_zeros():
    """The documented decision: GC critics get goals relabelled from the ONLINE trajectories
    by the same GCDataset the offline run uses. Zeros would be silently wrong, so assert the
    goals are (a) not zeros and (b) rows that actually occur in the buffer."""
    cfg = _gciql_cfg()
    buf = _fill(_buffer(cfg), n_episodes=10, ep_len=8)
    batch = buf.sample(64)
    assert np.abs(batch["value_goals"]).sum() > 0
    assert np.abs(batch["actor_goals"]).sum() > 0
    stored = buf.observations[:buf.size]
    for name in ("value_goals", "actor_goals"):
        for row in batch[name]:
            assert (np.abs(stored - row).sum(axis=1) < 1e-6).any(), \
                f"{name} row is not a state from the buffer"


def test_trajectory_goals_stay_inside_their_own_episode():
    """Boundary `terminals` is the whole reason episodes are stored whole: a p_trajgoal=1.0
    draw must land in the sampled row's OWN episode."""
    cfg = {**_gciql_cfg(), "value_p_curgoal": 0.0, "value_p_trajgoal": 1.0,
           "value_p_randomgoal": 0.0}
    ep_len = 7
    buf = _fill(_buffer(cfg), n_episodes=9, ep_len=ep_len)
    ds = buf.dataset()
    idxs = np.arange(buf.size)
    goals = ds.sample_goals(idxs, 0.0, 1.0, 0.0, cfg["value_geom_sample"])
    assert np.all(goals >= idxs)                       # goals are in the FUTURE
    assert np.all(goals // ep_len == idxs // ep_len)   # ... of the same episode


def test_in_progress_episode_is_invisible_until_it_ends():
    cfg = _gciql_cfg()
    buf = _buffer(cfg)
    _fill(buf, n_episodes=2, ep_len=5)
    for _ in range(3):
        buf.add(np.zeros(OBS_DIM), np.zeros(ACT_DIM), np.zeros(OBS_DIM), 0.0)
    assert buf.n_written == 13 and buf.size == 10      # 3 rows pending, not sampled from
    assert buf.dataset().size == 10
    buf.end_episode()
    assert buf.size == 13 and buf.dataset().size == 13


def test_empty_buffer_raises_a_message_that_says_why():
    with pytest.raises(RuntimeError, match="no finished episode"):
        _buffer(_gciql_cfg()).sample(4)


def test_missing_key_is_named_up_front():
    """The requirement: a missing key must not surface as a KeyError deep inside a critic."""
    from env.online import _check_keys
    with pytest.raises(KeyError, match="value_goals"):
        _check_keys({"observations": 1}, ("observations", "value_goals"), "unit test")

    buf = _fill(_buffer(_fb_cfg(), required=("observations", "masks")), n_episodes=3)
    with pytest.raises(KeyError, match="masks"):
        buf.sample(4)


def test_compaction_drops_whole_episodes_and_keeps_the_dataset_valid():
    cfg, ep_len = _gciql_cfg(), 5
    buf = _buffer(cfg, capacity=40)
    _fill(buf, n_episodes=20, ep_len=ep_len)           # 100 rows into a 40-row buffer
    assert buf.n_written <= 40
    assert buf.size % ep_len == 0                      # only whole episodes survived
    assert buf.n_episodes == buf.size // ep_len
    assert buf.terminals[buf.size - 1] == 1.0          # still ends on a boundary
    assert np.count_nonzero(buf.terminals[:buf.size]) == buf.n_episodes
    buf.sample(8)                                      # GCDataset's boundary assert holds


def test_compaction_refuses_a_buffer_too_small_for_one_episode():
    buf = _buffer(_gciql_cfg(), capacity=8)
    with pytest.raises(RuntimeError, match="online_buffer_size"):
        for _ in range(20):
            buf.add(np.zeros(OBS_DIM), np.zeros(ACT_DIM), np.zeros(OBS_DIM), 0.0)


def test_seed_from_offline_copies_whole_episodes():
    cfg = _gciql_cfg()
    data = _offline_arrays(n_episodes=8, ep_len=HORIZON)
    buf = _buffer(cfg, capacity=400)
    n = buf.seed_from(data, max_transitions=100)
    assert n == 5 * HORIZON                            # 100 rows -> 5 whole episodes
    assert buf.size == n and buf.n_episodes == 5
    np.testing.assert_array_equal(buf.observations[:n], data["observations"][:n])
    buf.observations[0, 0] = 12345.0                   # a copy, not a view
    assert data["observations"][0, 0] != 12345.0
    buf.sample(8)


def test_seed_from_offline_then_collect_more():
    cfg = _gciql_cfg()
    buf = _buffer(cfg, capacity=400)
    buf.seed_from(_offline_arrays(n_episodes=4, ep_len=HORIZON))
    before = buf.size
    _fill(buf, n_episodes=2, ep_len=6)
    assert buf.size == before + 12 and buf.n_episodes == 6


# ======================================================================================
# 3. Online loop smoke: really steps the env, really fills the buffer, really trains.
# ======================================================================================
def _trainer(cfg, adapter, env, **kw):
    from env.online import OnlineTrainer, ReplayBuffer
    buf = ReplayBuffer(kw.pop("capacity", 2000), OBS_DIM, ACT_DIM, cfg)
    return OnlineTrainer(adapter, env, cfg, buf, seed=SEED, **kw)


def _agent(cfg, critic, actor):
    from impls.agents import Agent
    return Agent.create(critic, actor, SEED, jnp.zeros((2, OBS_DIM)), jnp.zeros((2, ACT_DIM)), cfg)


@pytest.mark.parametrize("critic,actor,cfg_fn", [("gciql", "ddpgbc", _gciql_cfg),
                                                 ("fb", "flowbc", _fb_cfg)])
def test_online_smoke_steps_env_fills_buffer_and_trains(critic, actor, cfg_fn):
    cfg = cfg_fn()
    adapter, env = StubAdapter(), TinyEnv()
    agent = _agent(cfg, critic, actor)
    tr = _trainer(cfg, adapter, env, warmup_steps=60, action_noise=0.2,
                  env_steps_per_update=1.0)

    tr.prefill(agent)
    assert tr.env_steps == 60 and env.n_steps == 60
    assert tr.buffer.n_episodes == 3 and tr.buffer.size == 60   # horizon 20

    steps = 200
    losses = []
    for _ in range(steps):
        tr.collect(agent)
        batch = tr.sample(cfg["batch_size"])
        agent, info = agent.update(batch)
        losses.append(float(jnp.asarray(info["actor_loss"]).mean()))

    assert env.n_steps == 60 + steps                    # the env really was stepped
    assert tr.buffer.n_episodes == (60 + steps) // HORIZON
    assert tr.buffer.size == 60 + steps
    assert env.n_resets == tr.buffer.n_episodes
    assert np.isfinite(losses).all()
    assert _flat_params(agent).shape[0] > 0
    st = tr.stats()
    assert st["online/env_steps"] == 260 and st["online/buffer_size"] == 260
    assert "online/return" in st and np.isfinite(st["online/return"])


def test_online_smoke_through_main(tmp_path, monkeypatch):
    """The real entrypoint with --online=True: prefill, env stepping between gradient steps,
    replay sampling and checkpointing, all through impls/main.py."""
    cfg = _gciql_cfg()
    adapter = StubAdapter()
    steps = 120
    agent = _run_main(tmp_path, monkeypatch, cfg, adapter, "gciql", "ddpgbc", steps,
                      online=True, online_start_steps=40, online_buffer_size=5000,
                      online_env_steps_per_update=1.0, online_action_noise=0.1,
                      online_temperature=1.0, online_seed_offline=False,
                      online_seed_transitions=0, online_reset_seed_offset=1_000_000)
    assert adapter.env.n_steps == 40 + steps
    assert np.isfinite(_flat_params(agent)).all()
    # The online stats reached the train.csv row.
    csv = os.path.join(tmp_path, "tiny_gciql_ddpgbc_seed_0", "train.csv")
    assert "online/env_steps" in open(csv).readline()


def test_online_through_main_can_seed_from_the_offline_dataset(tmp_path, monkeypatch):
    cfg = _gciql_cfg()
    adapter = StubAdapter()
    _run_main(tmp_path, monkeypatch, cfg, adapter, "gciql", "ddpgbc", 20,
              online=True, online_start_steps=0, online_buffer_size=5000,
              online_env_steps_per_update=1.0, online_action_noise=0.1,
              online_temperature=1.0, online_seed_offline=True,
              online_seed_transitions=60, online_reset_seed_offset=1_000_000)
    # 60 offline rows (3 episodes of 20) + 20 online steps; prefill still runs to the first
    # completed online episode.
    assert adapter.env.n_steps >= 20


def test_utd_ratio_controls_env_steps_per_gradient_step():
    cfg = _gciql_cfg()
    for ratio, expected in [(0.0, 0), (0.5, 50), (1.0, 100), (3.0, 300)]:
        env = TinyEnv()
        tr = _trainer(cfg, StubAdapter(), env, warmup_steps=20,
                      env_steps_per_update=ratio)
        agent = _agent(cfg, "gciql", "ddpgbc")
        tr.prefill(agent)
        base = env.n_steps
        for _ in range(100):
            tr.collect(agent)
        assert env.n_steps - base == expected, f"ratio {ratio}"


def test_action_noise_is_configurable_and_actually_applied():
    """Same policy, same key, different --online_action_noise => different stored actions."""
    cfg = _gciql_cfg()
    agent = _agent(cfg, "gciql", "ddpgbc")
    acts = {}
    for noise in (0.0, 0.5):
        tr = _trainer(cfg, StubAdapter(), TinyEnv(), warmup_steps=0, action_noise=noise,
                      env_steps_per_update=1.0)
        for _ in range(10):
            tr.step_env(agent)
        acts[noise] = tr.buffer.actions[:10].copy()
    assert not np.allclose(acts[0.0], acts[0.5])
    assert np.abs(acts[0.0]).max() <= 1.0 and np.abs(acts[0.5]).max() <= 1.0


def test_warmup_actions_are_uniform_random_not_the_policy():
    cfg = _gciql_cfg()
    agent = _agent(cfg, "gciql", "ddpgbc")

    def _boom(*a, **k):
        raise AssertionError("sample_actions was called during warmup")

    tr = _trainer(cfg, StubAdapter(), TinyEnv(), warmup_steps=25, action_noise=0.0,
                  env_steps_per_update=1.0)
    stub = types.SimpleNamespace(sample_actions=_boom, critic=agent.critic)
    for _ in range(25):
        tr.step_env(stub)
    assert tr.env_steps == 25
    assert np.abs(tr.buffer.actions[:25]).max() <= 1.0


def test_online_collection_does_not_disturb_the_global_numpy_stream():
    """Exploration noise, warmup actions and env resets must all stay off the global stream
    GCDataset draws its batch indices from -- otherwise --seed stops meaning anything."""
    cfg = _gciql_cfg()
    agent = _agent(cfg, "gciql", "ddpgbc")
    tr = _trainer(cfg, StubAdapter(), TinyEnv(), warmup_steps=10, action_noise=0.3,
                  env_steps_per_update=1.0)
    np.random.seed(1234)
    before = np.random.get_state()[1].copy(), np.random.get_state()[2]
    for _ in range(40):
        tr.step_env(agent)
    after = np.random.get_state()[1].copy(), np.random.get_state()[2]
    np.testing.assert_array_equal(before[0], after[0])
    assert before[1] == after[1]


def test_online_reset_seeds_avoid_the_eval_initial_states():
    """evaluate() resets on seeds 0..num_episodes; online rollouts must not reuse them."""
    cfg = _gciql_cfg()
    seen = []

    class _Env(TinyEnv):
        def reset(self, seed=None, options=None):
            seen.append(seed)
            return super().reset(seed=seed, options=options)

    tr = _trainer(cfg, StubAdapter(), _Env(), warmup_steps=45, action_noise=0.0,
                  reset_seed_offset=1_000_000, env_steps_per_update=1.0)
    tr.prefill(_agent(cfg, "gciql", "ddpgbc"))
    assert seen == [1_000_000, 1_000_001, 1_000_002]


def test_ogbench_style_reset_reads_the_goal_out_of_info():
    """The other adapter's rollout contract: env.reset(options=...) and info['goal']."""
    cfg = _gciql_cfg()
    agent = _agent(cfg, "gciql", "ddpgbc")
    env = TinyEnv()
    tr = _trainer(cfg, StubAdapter(name="ogbench"), env, warmup_steps=0, action_noise=0.0,
                  env_steps_per_update=1.0)
    tr.step_env(agent)
    assert tr._ep_goal is not None and tr._ep_goal.shape == (OBS_DIM,)


def test_fb_gets_the_eval_goal_frame_exactly_as_evaluate_does():
    """FB conditions on z, but it exposes `map_eval_cond`, which embeds a goal FRAME into
    that z -- so env/unified.py::evaluate hands it one. Online collection must follow the
    same rule or FB would explore against a zero (goal-blind) z."""
    cfg = _fb_cfg()
    agent = _agent(cfg, "fb", "flowbc")
    adapter = StubAdapter()
    adapter._goal = np.arange(OBS_DIM, dtype=np.float32)
    tr = _trainer(cfg, adapter, TinyEnv(), warmup_steps=0, action_noise=0.0,
                  env_steps_per_update=1.0)
    tr.step_env(agent)
    np.testing.assert_array_equal(tr._ep_goal, adapter._goal)


def test_a_pure_z_critic_needs_no_goal_frame():
    """A critic that publishes only cond['z'] and has no map_eval_cond hook (isfb) is
    goal-free at rollout time, exactly as in evaluate()."""
    cfg = _gciql_cfg()
    agent = _agent(cfg, "gciql", "ddpgbc")
    z_only = types.SimpleNamespace(PROVIDES_COND=frozenset({"z"}))
    stub = types.SimpleNamespace(critic=z_only, sample_actions=agent.sample_actions)
    tr = _trainer(cfg, StubAdapter(), TinyEnv(), warmup_steps=0, action_noise=0.0,
                  env_steps_per_update=1.0)
    tr.step_env(stub)
    assert tr._ep_goal is None


def test_goal_conditioned_critic_without_a_goal_source_says_so():
    """The documented failure mode, raised loudly rather than silently emitting zeros."""
    cfg = _gciql_cfg()
    agent = _agent(cfg, "gciql", "ddpgbc")
    adapter = StubAdapter()
    adapter._goal = None                                # no offline data to take a goal from
    tr = _trainer(cfg, adapter, TinyEnv(), warmup_steps=0, action_noise=0.0,
                  env_steps_per_update=1.0)
    with pytest.raises(RuntimeError, match="eval goal frame"):
        tr.step_env(agent)


def test_buffer_stores_raw_observations_and_masking_happens_in_the_batch():
    """Offline stores raw frames and the dataset applies obs_mask_dims on the way out; the
    online buffer must do the same, or a masked run would train on doubly-blanked data."""
    cfg = {**_gciql_cfg(), "obs_mask_dims": [0, 1]}
    agent = _agent(cfg, "gciql", "ddpgbc")
    tr = _trainer(cfg, StubAdapter(), TinyEnv(), warmup_steps=0, action_noise=0.0,
                  env_steps_per_update=1.0)
    for _ in range(HORIZON):
        tr.step_env(agent)
    assert np.abs(tr.buffer.observations[:HORIZON, :2]).sum() > 0     # raw in storage
    batch = tr.sample(8)
    assert np.abs(batch["observations"][:, :2]).sum() == 0            # masked on the way out
    assert np.abs(batch["next_observations"][:, :2]).sum() == 0
