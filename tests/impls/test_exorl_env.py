"""ExoRL (dm_control) family: registry resolution, domain configs, and the loader.

The load-bearing assertion in this file is REWARDS. ExoRL is the only reward-specified
family in the tree -- its tasks are dense reward functions, not goals -- and it exists
precisely so `Critic.infer_cond(net, obs, rewards)` (density_fb / fb / td_fb / psm / rldp)
can be tested on something that was never a goal-reaching problem. Two ways that silently
dies, both covered here:

  * the ExoRL episode files' own `reward` column is the reward of the task the data was
    COLLECTED under, not the one you asked for (on point_mass_maze/rnd: four identical
    copies of `reach_top_left`), so the loader must relabel from the stored `physics`;
  * `GCDataset` overwrites `batch['rewards']` with its goal-reaching reward, which is what
    `impls/main.py` hands to `infer_cond`. `keep_dataset_rewards` is the opt-out.

Asserting the KEY is present is not enough for either: a zeroed or clobbered reward column
is present and the right dtype. Every reward assertion below is on the VALUES.
"""
import os

import numpy as np
import pytest

from env import registry
from env import exorl_env as E
from env.config import make_config
from impls.utils.datasets import FBDataset, GCDataset, make_dataset

REAL_BUFFER = os.path.join(
    os.environ.get("EXORL_DATA_ROOT", os.path.expanduser("~/exorl_data/exorl/datasets")),
    "point_mass_maze", "rnd", "buffer")


def _dm_control_available():
    try:
        E._mujoco_overlay()
        return True
    except Exception:                                              # noqa: BLE001
        return False


needs_dmc = pytest.mark.skipif(
    not _dm_control_available(),
    reason="no mujoco/dm_control pairing this interpreter can drive (see "
           "env/exorl_env.py::_mujoco_overlay / FB_MUJOCO_SITE_PACKAGES)")
needs_real_data = pytest.mark.skipif(
    not os.path.isdir(REAL_BUFFER),
    reason=f"ExoRL point_mass_maze/rnd buffer not downloaded at {REAL_BUFFER}")


# --------------------------------------------------------------------------------------
# 1. registry
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["dmc-point-mass-maze", "dmc-walker", "dmc-cheetah"])
def test_prefix_resolves_family_and_horizon(task):
    assert registry.family(task) == "exorl"
    assert registry.horizon(task) == 1000        # ExoRL episode length, action_repeat 1


def test_data_dir_uses_the_exorl_family_folder():
    d = registry.data_dir("dmc-walker")
    assert d.endswith("ExoRL__dmc-walker")
    assert os.path.dirname(d) == registry.COLLECTED


def test_other_families_are_untouched():
    """The prefix table is shared; adding one must not move the existing four."""
    assert (registry.family("mw-push"), registry.horizon("mw-push")) == ("metaworld", 100)
    assert (registry.family("ms-push-cube"), registry.horizon("ms-push-cube")) == ("maniskill", 25)
    assert (registry.family("b2d-center-wall"), registry.horizon("b2d-center-wall")) == ("box2d", 200)


def test_unknown_prefix_still_raises():
    with pytest.raises(KeyError):
        registry.family("nope-whatever")


# --------------------------------------------------------------------------------------
# 2. task table + domain configs
# --------------------------------------------------------------------------------------

def test_reward_task_defaults_and_validates():
    assert E.resolve_reward_task("dmc-walker") == "walk"
    assert E.resolve_reward_task("dmc-walker", "run") == "run"
    # A typo must NOT quietly fall back to the default: it would relabel the whole dataset
    # for a task nobody asked about and every number downstream would be for that task.
    with pytest.raises(ValueError, match="unknown reward task"):
        E.resolve_reward_task("dmc-walker", "sprint")
    with pytest.raises(ValueError, match="unknown ExoRL task"):
        E.domain_of("dmc-humanoid")


@pytest.mark.parametrize("domain", ["dmc-point-mass-maze", "dmc-walker", "dmc-cheetah"])
def test_domain_yaml_composes(domain, monkeypatch):
    monkeypatch.setenv("EXORL_DATA_ROOT", "/tmp/exorl-root")
    cfg = make_config("fb", "flowbc", domain)
    assert cfg["adapter"] == "unified"
    assert cfg["task"] == domain
    # the yaml's reward task must be one this domain actually has
    assert cfg["exorl_task"] in E.reward_tasks(domain)
    assert cfg["data_dir"].startswith("/tmp/exorl-root/")
    assert E.domain_of(domain) in cfg["data_dir"]
    assert cfg["keep_dataset_rewards"] is True


def test_domain_yaml_reaches_the_exorl_loader(monkeypatch):
    """env/unified.py must route dmc- tasks to the ExoRL loader, not the collected-npz one
    (whose <data_dir>/buffer/episode_*.npz layout happens to look similar and whose
    `reward` field is the trap this family exists to avoid)."""
    from env import unified
    seen = {}
    monkeypatch.setattr(E, "load_domain_data", lambda cfg: seen.setdefault("cfg", cfg))
    unified.load_domain_data({"task": "dmc-walker", "data_path": "/should/be/ignored.npz"})
    assert seen["cfg"]["task"] == "dmc-walker"


# --------------------------------------------------------------------------------------
# 3. the loader, on a synthetic ExoRL-shaped buffer
# --------------------------------------------------------------------------------------

def _write_fake_buffer(tmp_path, n_ep=3, T=7, obs_dim=4, act_dim=2, seed=0):
    """An ExoRL buffer with the dummy-first-transition convention and a `reward` column of
    the real shape (T+1, 4) whose values are a DIFFERENT, clearly identifiable signal from
    the relabeller's. Anything that reads the stored column instead of relabelling shows up
    below as the wrong values, not merely as a missing key."""
    rs = np.random.RandomState(seed)
    d = tmp_path / "buffer"
    d.mkdir()
    for e in range(n_ep):
        obs = rs.randn(T + 1, obs_dim).astype(np.float32)
        act = rs.randn(T + 1, act_dim).astype(np.float32) * 3.0   # out of [-1,1], to test clip
        act[0] = 0.0                                              # the dummy padding row
        np.savez(d / f"episode_{e:06d}_{T}.npz",
                 observation=obs, action=act,
                 reward=np.tile(rs.rand(T + 1, 1).astype(np.float32) * 0.01, (1, 4)),
                 discount=np.ones((T + 1, 1), np.float32),
                 physics=obs.astype(np.float64))
    return d


def _fake_relabel(physics, domain, reward_task, env=None):
    """Stand-in for the mujoco relabeller: a deterministic, clearly non-constant function of
    the physics state, so "did the REAL reward survive" is checkable without mujoco."""
    return (np.asarray(physics)[:, 0] * 10.0).astype(np.float32)


@pytest.fixture()
def fake_data(tmp_path, monkeypatch):
    buf = _write_fake_buffer(tmp_path)
    monkeypatch.setattr(E, "relabel_rewards", _fake_relabel)
    data = E.load_exorl(buf, "dmc-point-mass-maze", reward_task="reach_top_left",
                        use_cache=False)
    return buf, data


def test_loader_emits_the_flat_dataset_contract(fake_data):
    _, data = fake_data
    assert set(data) == {"observations", "actions", "next_observations", "rewards", "terminals"}
    n = 3 * 7
    for k in data:
        assert data[k].dtype == np.float32, k
        assert len(data[k]) == n, k
    assert data["observations"].shape == (n, 4)
    assert data["actions"].shape == (n, 2)
    assert data["next_observations"].shape == (n, 4)
    assert data["rewards"].shape == (n,)


def test_loader_drops_the_dummy_action_and_aligns_next_observations(fake_data):
    buf, data = fake_data
    z = np.load(buf / "episode_000000_7.npz")
    np.testing.assert_allclose(data["observations"][:7], z["observation"][:-1])
    np.testing.assert_allclose(data["next_observations"][:7], z["observation"][1:])
    # actions are act[1:] (act[0] is padding), clipped to the env's own bounds
    np.testing.assert_allclose(data["actions"][:7], np.clip(z["action"][1:], -1.0, 1.0))
    assert not np.allclose(data["actions"][:7], np.clip(z["action"][:-1], -1.0, 1.0))


def test_loader_marks_episode_boundaries_not_absorbing_states(fake_data):
    _, data = fake_data
    assert list(np.nonzero(data["terminals"])[0]) == [6, 13, 20]


def test_loader_relabels_and_does_not_pass_through_the_stored_reward_column(fake_data):
    """The whole point. The emitted rewards must be the RELABELLED task reward at the next
    state, not the stored column (which is the collection task's reward, right for at most
    one task per domain and silently wrong for every other)."""
    buf, data = fake_data
    r = data["rewards"]
    z = np.load(buf / "episode_000000_7.npz")
    assert not np.allclose(r, 0.0), "rewards collapsed to zero"
    assert np.ptp(r) > 1.0, f"rewards are near-constant (ptp={np.ptp(r)}) -- no task signal"
    assert np.abs(r[:7]).max() > 10 * np.abs(z["reward"]).max(), \
        "emitted rewards are on the stored column's scale -- the loader passed it through"
    expected = _fake_relabel(z["physics"], None, None)[1:]        # reward at s_{t+1}
    np.testing.assert_allclose(r[:7], expected, rtol=1e-6)


def test_loader_refuses_to_fall_back_when_physics_is_missing(tmp_path):
    d = tmp_path / "buffer"
    d.mkdir()
    T = 5
    np.savez(d / f"episode_000000_{T}.npz",
             observation=np.zeros((T + 1, 4), np.float32),
             action=np.zeros((T + 1, 2), np.float32),
             reward=np.ones((T + 1, 4), np.float32))   # tempting, and wrong
    with pytest.raises(ValueError, match="physics"):
        E.load_exorl(d, "dmc-walker", use_cache=False)


def test_reward_cache_round_trips(tmp_path, monkeypatch):
    buf = _write_fake_buffer(tmp_path)
    calls = []

    def counting(physics, domain, reward_task, env=None):
        calls.append(1)
        return _fake_relabel(physics, domain, reward_task)

    monkeypatch.setattr(E, "relabel_rewards", counting)
    monkeypatch.setattr(E, "_load_dmc", lambda *a, **k: None)
    cache = str(tmp_path / "cache")
    a = E.load_exorl(buf, "dmc-walker", cache_dir=cache)
    n_first = len(calls)
    b = E.load_exorl(buf, "dmc-walker", cache_dir=cache)
    assert n_first == 3 and len(calls) == n_first, "second load re-relabelled instead of caching"
    np.testing.assert_array_equal(a["rewards"], b["rewards"])


# --------------------------------------------------------------------------------------
# 4. real rewards through make_dataset -> batch["rewards"]
# --------------------------------------------------------------------------------------

def _fb_cfg():
    return dict(z_dim=50)


def _gc_cfg(**over):
    cfg = dict(dataset_class="GCDataset", discount=0.99, gc_negative=True,
               value_p_curgoal=0.2, value_p_trajgoal=0.5, value_p_randomgoal=0.3,
               value_geom_sample=True, actor_p_curgoal=0.0, actor_p_trajgoal=1.0,
               actor_p_randomgoal=0.0, actor_geom_sample=False)
    cfg.update(over)
    return cfg


def test_fbdataset_carries_the_real_rewards(fake_data):
    """fb / psm read the base FBDataset, which passes `rewards` straight through."""
    _, data = fake_data
    ds = make_dataset(FBDataset.create(**data), _fb_cfg())
    idxs = np.array([0, 3, 9, 15])
    b = ds.sample(len(idxs), idxs=idxs)
    for k in ("observations", "actions", "next_observations", "rewards", "terminals"):
        assert k in b, k
        assert np.asarray(b[k]).dtype == np.float32, k
    np.testing.assert_allclose(b["rewards"].reshape(-1), data["rewards"][idxs], rtol=1e-6)
    assert np.ptp(b["rewards"]) > 1.0


def test_gcdataset_clobbers_the_real_rewards_by_default(fake_data):
    """Regression guard on the DEFAULT, and the reason `keep_dataset_rewards` exists: the
    goal-reaching reward is in {-1, 0}, has nothing to do with the ExoRL task, and nothing
    in impls/main.py warns (infer_cond_noop_reason only flags an ALL-ZERO vector)."""
    _, data = fake_data
    ds = make_dataset(FBDataset.create(**data), _gc_cfg())
    b = ds.sample(16)
    assert isinstance(ds, GCDataset)
    assert set(np.unique(b["rewards"])) <= {-1.0, 0.0}


def test_gcdataset_keep_dataset_rewards_restores_them(fake_data):
    """density_fb / td_fb are GCDataset critics; this is the only path by which a real task
    reward reaches their `infer_cond`."""
    _, data = fake_data
    ds = make_dataset(FBDataset.create(**data), _gc_cfg(keep_dataset_rewards=True))
    idxs = np.array([1, 4, 8, 12, 19])
    b = ds.sample(len(idxs), idxs=idxs)
    for k in ("observations", "actions", "next_observations", "next_actions",
              "rewards", "masks", "value_goals", "actor_goals"):
        assert k in b, k
        assert np.asarray(b[k]).dtype == np.float32, k
    np.testing.assert_allclose(b["rewards"], data["rewards"][idxs], rtol=1e-6)
    assert np.ptp(b["rewards"]) > 1.0, "reward column is near-constant -- no task signal"
    # dm_control episodes end on a time limit, never on an absorbing state.
    np.testing.assert_array_equal(b["masks"], np.ones(len(idxs), np.float32))


def test_keep_dataset_rewards_fails_loudly_on_unlabeled_data():
    n = 100
    base = FBDataset.create(
        observations=np.zeros((n, 4), np.float32), actions=np.zeros((n, 2), np.float32),
        terminals=np.array([1.0 if (i + 1) % 25 == 0 else 0.0 for i in range(n)], np.float32))
    ds = make_dataset(base, _gc_cfg(keep_dataset_rewards=True))
    with pytest.raises(KeyError, match="keep_dataset_rewards"):
        ds.sample(8)


def test_keep_dataset_rewards_off_is_a_no_op_on_the_draw(fake_data):
    """Default-off must change nothing, including the np.random call sequence -- every
    existing goal-conditioned run is scored against recorded numbers."""
    _, data = fake_data
    base = FBDataset.create(**data)
    out = []
    for cfg in (_gc_cfg(), _gc_cfg(keep_dataset_rewards=False)):
        np.random.seed(7)
        out.append(make_dataset(base, cfg).sample(16))
    for k in out[0]:
        np.testing.assert_array_equal(out[0][k], out[1][k], err_msg=k)


# --------------------------------------------------------------------------------------
# 5. against the REAL downloaded buffer (skipped when absent)
# --------------------------------------------------------------------------------------

@needs_dmc
@needs_real_data
def test_real_exorl_rewards_are_the_task_reward_not_the_stored_column():
    """End-to-end on real ExoRL data. Two claims:

    (a) the stored `reward` column is NOT usable as `batch['rewards']` -- it is (T+1, 4),
        not a scalar per transition, so there is no column to take without choosing one;
    (b) the relabelled reward IS the point_mass_maze `near_target` reward: bounded in
        [0, 1] and peaking exactly at the closest approach to the requested target.

    Note what (a) does NOT say: on `point_mass_maze/rnd` those four columns happen to be
    the `reach_top_left` reward, so a passthrough would look fine for THAT target and be
    wrong for the other three. That is the whole hazard, and it is why (b) is asserted
    against the target's own geometry rather than against the stored column."""
    files = E.episode_files(REAL_BUFFER, 200)
    stored = np.load(files[0])["reward"]
    assert stored.ndim == 2 and stored.shape[1] == 4, (
        f"the premise changed: stored reward is {stored.shape}, no longer the 4-wide "
        f"collection-task column this family's loader refuses to read")

    # Pick episodes that actually visit the target, or the (correct) reward is ~0 everywhere
    # and the assertion below would be testing nothing.
    tgt = np.array([-0.15, 0.15])
    hits = []
    for f in files:
        no = E.read_episode(f)[2]
        if np.linalg.norm(no[:, :2] - tgt, axis=1).min() < 0.015:
            hits.append(f)
        if len(hits) >= 2:
            break
    assert hits, "no episode in the first 200 reaches reach_top_left's full-credit radius"

    for f in hits:
        _o, _a, no, phys = E.read_episode(f)
        r = E.relabel_rewards(phys, "point_mass_maze", "reach_top_left")[1:]
        assert r.shape == (no.shape[0],)
        assert r.min() >= 0.0 and r.max() <= 1.0 + 1e-6
        assert r.max() > 0.5, f"{f}: visits the target but the relabelled reward stays {r.max()}"
        d = np.linalg.norm(no[:, :2] - tgt, axis=1)
        assert d[int(np.argmax(r))] < 0.015, "peak reward is not at the closest state"


@needs_dmc
@needs_real_data
def test_real_loader_reaches_a_batch(tmp_path):
    data = E.load_exorl(REAL_BUFFER, "dmc-point-mass-maze", reward_task="reach_top_left",
                        max_episodes=40, cache_dir=str(tmp_path))
    assert data["observations"].shape[1] == 4 and data["actions"].shape[1] == 2
    assert data["rewards"].max() > 0.0, "40 real episodes produced no reward signal at all"
    ds = make_dataset(FBDataset.create(**data), _gc_cfg(keep_dataset_rewards=True))
    hot = int(np.argmax(data["rewards"]))
    b = ds.sample(4, idxs=np.array([hot, hot + 1, 0, 1]))
    assert b["rewards"][0] == pytest.approx(float(data["rewards"][hot]), rel=1e-6)
    assert b["rewards"][0] > 0.0


# --------------------------------------------------------------------------------------
# 6. the env
# --------------------------------------------------------------------------------------

@needs_dmc
def test_env_rolls_out_and_matches_the_dataset_obs_layout():
    from env.unified import make_task_env
    env, horizon = make_task_env("dmc-point-mass-maze", reward_task="reach_top_left")
    assert horizon == 1000
    obs, info = env.reset(seed=0)
    assert obs.shape == (4,) and obs.dtype == np.float32     # [x, y, vx, vy], as the buffers
    assert "success" in info                                 # point_mass_maze has a goal rule
    total = 0.0
    for _ in range(5):
        obs, r, term, trunc, info = env.step(np.array([0.5, -0.5], np.float32))
        total += r
        assert obs.shape == (4,) and not term
    assert total >= 0.0
    env.close()


@needs_dmc
def test_env_reset_is_seed_deterministic():
    """Two checkpoints must be scored on the SAME initial states. point_mass_maze draws its
    start qpos from the GLOBAL np.random, so this fails if only task.random is seeded."""
    from env.unified import make_task_env
    env, _ = make_task_env("dmc-point-mass-maze")
    a, _ = env.reset(seed=3)
    b, _ = env.reset(seed=4)
    c, _ = env.reset(seed=3)
    np.testing.assert_allclose(a, c)
    assert not np.allclose(a, b)
    env.close()


@needs_dmc
def test_env_reset_restores_the_global_rng():
    """Eval must not perturb training: every dataset class draws its batch indices from the
    global np.random, so a reset that leaves it seeded makes the post-eval batch stream a
    pure function of the last episode index."""
    from env.unified import make_task_env
    env, _ = make_task_env("dmc-point-mass-maze")
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    env.reset(seed=99)
    np.testing.assert_array_equal(np.random.get_state()[1], before)
    env.close()


@needs_dmc
@pytest.mark.parametrize("slug,reward_task", [(s, t) for s in E.TASKS
                                              for t in E.TASKS[s][2]])
def test_every_registered_reward_task_builds_and_steps(slug, reward_task):
    """All 12 (domain, reward task) pairs, including the ones that are ExoRL ADDITIONS
    rather than dm_control suite tasks (walker flip, cheetah run_backward/flip/
    flip_backward, all four maze targets) -- those resolve through
    third_party/exorl_dmc_tasks and are exactly the ones a wrong suite-first/custom-second
    resolution order would silently mis-load."""
    env, horizon = E.make_exorl_env(slug, reward_task=reward_task)
    assert horizon == 1000
    obs, _info = env.reset(seed=0)
    expected_dim = {"point_mass_maze": 4, "walker": 24, "cheetah": 17,
                        "quadruped": 78}[E.domain_of(slug)]
    assert obs.shape == (expected_dim,), f"{slug}/{reward_task}: obs {obs.shape}"
    obs, r, term, trunc, info = env.step(np.zeros(env.env.action_spec().shape))
    assert obs.shape == (expected_dim,) and not term
    assert np.isfinite(r) and info["reward"] == r
    env.close()


@needs_dmc
@pytest.mark.parametrize("slug,phys_dim", [("dmc-walker", 18), ("dmc-cheetah", 18)])
def test_relabeller_runs_for_the_locomotion_domains(slug, phys_dim):
    """walker/cheetah RND buffers are not downloaded on this box, so this is the furthest
    the relabelling path can be exercised for them: real mujoco `set_state` + the real
    `task.get_reward`, on physics states drawn from the env itself (NOT random vectors,
    which can be non-physical and would test the wrong thing)."""
    env, _ = E.make_exorl_env(slug)
    env.reset(seed=0)
    states = []
    for _ in range(6):
        env.step(np.zeros(env.env.action_spec().shape))
        states.append(env.env.physics.get_state().copy())
    states = np.stack(states)
    assert states.shape[1] == phys_dim, f"{slug} physics state dim {states.shape[1]}"
    r = E.relabel_rewards(states, E.domain_of(slug), E.default_reward_task(slug))
    assert r.shape == (6,) and r.dtype == np.float32
    assert np.all(np.isfinite(r)) and r.min() >= 0.0
    env.close()


@needs_dmc
def test_walker_has_no_success_key():
    """A fabricated success=0.0 would read as "the policy failed" in eval.csv; walker's
    benchmark metric is episode return, which env/unified.py reports as episode_return."""
    from env.unified import make_task_env
    env, _ = make_task_env("dmc-walker", reward_task="walk")
    obs, info = env.reset(seed=0)
    assert "success" not in info
    assert obs.shape == (24,)                    # walker flat obs: 14 position + 9 velocity...
    env.close()


# --------------------------------------------------------------------------------------
# The OTHER half of "walker has no success key": the ADAPTER must not invent one.
# No mujoco needed -- the wrapper's info contract is all this exercises.
# --------------------------------------------------------------------------------------
class _NoSuccessEnv:
    """ExoRLWrapper's surface for walker/cheetah: dense reward, and NO 'success' in info."""

    def __init__(self, obs_dim=24, horizon=10):
        self.obs_dim, self.horizon, self.t = obs_dim, horizon, 0

    def reset(self, seed=None, **_kw):
        self.t = 0
        return np.zeros(self.obs_dim, np.float32), {}

    def step(self, _a):
        self.t += 1
        return (np.zeros(self.obs_dim, np.float32), 0.83, False, self.t >= self.horizon,
                {"reward": 0.83})


class _ZCritic:
    PROVIDES_COND = frozenset({"z"})


class _ZAgent:
    critic = _ZCritic()

    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        return np.zeros(6, np.float32)


def test_evaluate_does_not_fabricate_a_success_metric():
    """env/unified.py::evaluate used to emit `success: 0.0` unconditionally. On walker /
    cheetah that reads as "the policy failed every episode" in eval.csv, and
    impls/main.py::_append_agg shipped it to the campaign aggregator as the run's
    success_rate while `episode_return` -- the domain's actual benchmark metric -- was never
    aggregated at all. ExoRLWrapper's own docstring already refuses to fabricate it; this is
    the adapter half of that contract."""
    from env.unified import UnifiedAdapter
    ad = UnifiedAdapter()
    ad._horizon = 10
    row, _ = ad.evaluate(_ZAgent(), _NoSuccessEnv(), None,
                         {"task": "dmc-walker", "exorl_task": "walk"},
                         num_episodes=5, num_video=0)
    assert "success" not in row, f"fabricated a success metric: {row}"
    assert row["episode_return"] == pytest.approx(8.3)


def test_evaluate_still_reports_success_when_the_env_has_a_success_rule():
    """The guard must not cost the families that DO have one (Box2D / MW / MS / pmm)."""
    from env.unified import UnifiedAdapter

    class _WithSuccess(_NoSuccessEnv):
        def reset(self, seed=None, **_kw):
            self.t = 0
            return np.zeros(self.obs_dim, np.float32), {"success": 0.0}

        def step(self, _a):
            self.t += 1
            return (np.zeros(self.obs_dim, np.float32), 0.0, False, self.t >= self.horizon,
                    {"success": 1.0})

    ad = UnifiedAdapter()
    ad._horizon = 10
    row, _ = ad.evaluate(_ZAgent(), _WithSuccess(), None,
                         {"task": "dmc-point-mass-maze", "exorl_task": "reach_top_left"},
                         num_episodes=3, num_video=0)
    assert row["success"] == pytest.approx(1.0)


def test_append_agg_does_not_report_a_missing_success_as_zero():
    """A row with no success metric must not become success_rate=0.0 in the agg jsonl."""
    import json
    import tempfile
    from impls.main import _append_agg
    d = tempfile.mkdtemp()
    _append_agg(d, "g", 0, 100, {"episode_return": 8.3}, "dmc-walker")
    rec = json.loads(open(os.path.join(d, "g__s0.jsonl")).read())
    assert np.isnan(rec["success_rate"]), rec
