import numpy as np
import pytest
from pathlib import Path
from gymnasium import spaces
from hydra import compose, initialize_config_dir

from data.episodes import load_episode_transitions
from data.rlmini import load_rlmini_dataset
from envs.rlmini import create_rlmini_env
from scripts.data.export_rlmini_random import collect_episode
from train import make_agent


REPO_ROOT = Path(__file__).resolve().parent.parent


RL_MINI_SPECS = [
    ("Box2D", "goal-ctrl-target", {"continuous": True}),
    ("AirHockey", "default", {"continuous": True}),
    ("Libero", "default", {"continuous": True}),
]


def _cfg(env_name, variant, extra=None, horizon=8):
    return {
        "rlmini_env": env_name,
        "rlmini_variant": variant,
        "rlmini_horizon": horizon,
        "rlmini_render": False,
        "rlmini_obs_keys": None,
        "rlmini_obs_slices": None,
        "rlmini_extra": extra or {},
    }


def _skip_optional_dependency(exc):
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        pytest.skip(f"optional rlmini dependency missing: {exc}")
    raise exc


@pytest.mark.parametrize("env_name,variant,extra", RL_MINI_SPECS)
def test_rlmini_gymnasium_wrapper_contract(env_name, variant, extra):
    try:
        env, _ = create_rlmini_env(
            f"rlmini-{env_name.lower()}",
            cfg=_cfg(env_name, variant, extra),
            seed=0,
        )
    except Exception as exc:
        _skip_optional_dependency(exc)

    try:
        obs, info = env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert obs.ndim == 1
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Box)
        assert env.action_space.shape is not None and len(env.action_space.shape) == 1
        next_obs, reward, terminated, truncated, step_info = env.step(env.action_space.sample())
        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape == obs.shape
        assert isinstance(float(reward), float)
        assert isinstance(bool(terminated), bool)
        assert isinstance(bool(truncated), bool)
        assert isinstance(info, dict)
        assert isinstance(step_info, dict)
    finally:
        env.close()


@pytest.mark.parametrize("env_name,variant,extra", RL_MINI_SPECS)
def test_rlmini_random_episode_loads_as_transition_data(env_name, variant, extra, tmp_path):
    try:
        env, _ = create_rlmini_env(
            f"rlmini-{env_name.lower()}",
            cfg=_cfg(env_name, variant, extra),
            seed=0,
        )
    except Exception as exc:
        _skip_optional_dependency(exc)

    try:
        ep = collect_episode(env, max_steps=3)
    finally:
        env.close()

    path = tmp_path / "episode_000000_3.npz"
    np.savez_compressed(path, **ep)
    storage = load_episode_transitions([path], action_alignment="transition")
    assert storage["observation"].shape[0] >= 1
    assert storage["action"].shape[-1] == ep["action"].shape[-1]
    assert storage["next"]["observation"].shape == storage["observation"].shape
    assert "reward" in storage
    assert "physics" in storage["next"]


def test_rlmini_loader_and_agent_update_on_tiny_offline_data(tmp_path):
    domain = "rlmini-box2d-goal-target"
    buf_dir = tmp_path / domain / "buffer"
    buf_dir.mkdir(parents=True)
    obs = np.random.randn(6, 8).astype(np.float32)
    act = np.random.uniform(-1, 1, size=(6, 2)).astype(np.float32)
    np.savez_compressed(
        buf_dir / "episode_000000_5.npz",
        observation=obs,
        physics=obs.copy(),
        action=act,
        reward=np.zeros((6, 1), dtype=np.float32),
        discount=np.ones((6, 1), dtype=np.float32),
        terminated=np.zeros((6, 1), dtype=bool),
    )
    buffer = load_rlmini_dataset(domain, data_path=tmp_path, device="cpu", load_n_episodes=1)
    batch = buffer.sample(4)
    assert batch["observation"].shape == (4, 8)
    assert batch["action"].shape == (4, 2)

    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[
                "domain=rlmini_box2d_goal_target",
                "device=cpu",
                "batch_size=4",
                "z_dim=8",
                "L_dim=8",
                "forward.hidden_dim=32",
                "backward.hidden_dim=32",
                "left_encoder.hidden_dim=32",
                "actor.hidden_dim=32",
            ],
        )
    agent = make_agent(cfg, spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32), 2)
    metrics = agent.update(batch, step=0)
    assert metrics
    assert all(np.isfinite(float(v.detach() if hasattr(v, "detach") else v)) for v in metrics.values())
