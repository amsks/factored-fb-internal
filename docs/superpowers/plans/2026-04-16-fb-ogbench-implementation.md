# FB Baseline on OGBench — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the ForwardBackward (FB) baseline from `../td_jepa/metamotivo/` into a clean, standalone `Factored-FB` repository that can train on OGBench state-observation tasks via a simple `python train.py` command.

**Architecture:** Flat package structure under the project root; Python adds the project root to `sys.path` when running `train.py`, so all imports are absolute (e.g. `from base_config import BaseConfig`). Within a package, relative imports are used (e.g. `from .model import FBModel`). Code is copied verbatim from `td_jepa/metamotivo/` with only import paths updated; `train.py` is rewritten with `argparse` (replacing `tyro`); `data/ogbench.py` and `evals/ogbench.py` are simplified (state-obs only, no pydantic configs needed at the call site).

**Tech Stack:** Python ≥ 3.11, PyTorch ≥ 2.3, ogbench ≥ 1.1.2, pydantic ≥ 2.0, h5py, safetensors, wandb, tqdm, numpy

**Source:** All core logic lives in `/Users/adityamohan/git/Austin/td_jepa/metamotivo/`. Read those files when implementing; the plan shows only the import block for copied files and full source for new files.

---

## File Map

| New file | Source | Changes |
|---|---|---|
| `requirements.txt` | new | — |
| `configs/.gitkeep` | new | — |
| `tests/conftest.py` | new | — |
| `base_config.py` | `td_jepa/metamotivo/base.py` | none |
| `base_model.py` | `td_jepa/metamotivo/base_model.py` | 2 imports |
| `normalizers.py` | `td_jepa/metamotivo/normalizers.py` | 1 import |
| `utils.py` | `td_jepa/metamotivo/utils.py` | none |
| `nn_models.py` | `td_jepa/metamotivo/nn_models.py` + `td_jepa/metamotivo/agents/td3/nn_models.py` | 1 import + append VectorField |
| `envs/__init__.py` | new | — |
| `envs/gym_spaces.py` | `td_jepa/metamotivo/envs/utils/gym_spaces.py` | none |
| `envs/wrappers.py` | `td_jepa/metamotivo/envs/utils/wrappers.py` | none |
| `envs/rollout.py` | `td_jepa/metamotivo/envs/utils/rollout.py` | none |
| `envs/ogbench.py` | `td_jepa/metamotivo/envs/ogbench.py` | drop OGBenchEnvConfig, add get_relabel_fn, fix 2 imports |
| `buffers/__init__.py` | new | — |
| `buffers/transition.py` | `td_jepa/metamotivo/buffers/transition.py` | none |
| `data/__init__.py` | new | — |
| `data/ogbench.py` | merge of `td_jepa/metamotivo/data_loading/base.py` + `ogbench.py` | new (state-obs only) |
| `agents/__init__.py` | new | — |
| `agents/fb/__init__.py` | new | — |
| `agents/fb/model.py` | `td_jepa/metamotivo/agents/fb/model.py` | 4 imports |
| `agents/fb/agent.py` | `td_jepa/metamotivo/agents/fb/agent.py` | 3 imports |
| `agents/fb/flow_bc/__init__.py` | new | — |
| `agents/fb/flow_bc/model.py` | `td_jepa/metamotivo/agents/fb/flow_bc/model.py` | 3 imports |
| `agents/fb/flow_bc/agent.py` | `td_jepa/metamotivo/agents/fb/flow_bc/agent.py` | none (already relative) |
| `evals/__init__.py` | new | — |
| `evals/ogbench.py` | simplified from `td_jepa/metamotivo/evaluations/ogbench.py` | new (no pydantic config) |
| `train.py` | new | argparse entry point |

---

## Task 1: Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `configs/.gitkeep`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `agents/__init__.py`, `agents/fb/__init__.py`, `agents/fb/flow_bc/__init__.py`
- Create: `envs/__init__.py`, `buffers/__init__.py`, `data/__init__.py`, `evals/__init__.py`

- [ ] **Step 1: Write requirements.txt**

```
torch>=2.3
ogbench>=1.1.2
wandb>=0.22.3
h5py>=3.11.0
safetensors>=0.4.4
pydantic>=2.0
tqdm
numpy
gymnasium
```

Save to `requirements.txt`.

- [ ] **Step 2: Install dependencies**

```bash
uv pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 3: Create empty __init__.py and stub files**

Create the following as empty files:
- `configs/.gitkeep`
- `tests/__init__.py`
- `agents/__init__.py`
- `agents/fb/__init__.py`
- `agents/fb/flow_bc/__init__.py`
- `envs/__init__.py`
- `buffers/__init__.py`
- `data/__init__.py`
- `evals/__init__.py`

- [ ] **Step 4: Create tests/conftest.py**

```python
import sys
from pathlib import Path

# Add project root to sys.path so all absolute imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt configs/.gitkeep tests/ agents/ envs/ buffers/ data/ evals/
git commit -m "feat: project scaffolding — packages, requirements, test config"
```

---

## Task 2: Core utilities

**Files:**
- Create: `base_config.py` (copy)
- Create: `base_model.py` (copy + 2 import changes)
- Create: `normalizers.py` (copy + 1 import change)
- Create: `utils.py` (copy)
- Create: `envs/gym_spaces.py` (copy)
- Test: `tests/test_core.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core.py
def test_base_config_import():
    from base_config import BaseConfig
    assert BaseConfig is not None

def test_base_model_import():
    from base_model import BaseModel, BaseModelConfig, load_model, save_model
    assert BaseModel is not None

def test_normalizers_import():
    from normalizers import AVAILABLE_NORMALIZERS, IdentityNormalizerConfig, BatchNormNormalizerConfig
    assert IdentityNormalizerConfig is not None

def test_utils_import():
    from utils import set_seed_everywhere, EveryNStepsChecker, get_local_workdir
    assert set_seed_everywhere is not None

def test_gym_spaces_import():
    from envs.gym_spaces import space_to_json, json_to_space
    import gymnasium
    import numpy as np
    space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    json_repr = space_to_json(space)
    recovered = json_to_space(json_repr)
    assert recovered.shape == space.shape
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/adityamohan/git/Austin/Factored-FB && python -m pytest tests/test_core.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create base_config.py**

Copy `/Users/adityamohan/git/Austin/td_jepa/metamotivo/base.py` verbatim to `base_config.py`. No import changes needed (only imports `pydantic`).

- [ ] **Step 4: Create base_model.py**

Copy `/Users/adityamohan/git/Austin/td_jepa/metamotivo/base_model.py` to `base_model.py` with these import changes at the top:

```python
# CHANGE from:
from .base import BaseConfig
from .envs.utils.gym_spaces import json_to_space, space_to_json

# TO:
from base_config import BaseConfig
from envs.gym_spaces import json_to_space, space_to_json
```

The rest of the file (save_model, load_model, BaseModelConfig, BaseModel) is verbatim.

- [ ] **Step 5: Create normalizers.py**

Copy `/Users/adityamohan/git/Austin/td_jepa/metamotivo/normalizers.py` to `normalizers.py` with one import change:

```python
# CHANGE from:
from metamotivo.base import BaseConfig

# TO:
from base_config import BaseConfig
```

The rest of the file (BatchNormNormalizerConfig, IdentityNormalizerConfig, RGBNormalizerConfig, AVAILABLE_NORMALIZERS) is verbatim.

- [ ] **Step 6: Create utils.py**

Copy `/Users/adityamohan/git/Austin/td_jepa/metamotivo/utils.py` verbatim to `utils.py`. No import changes (only uses `datetime`, `random`, `string`, `numpy`, `torch`).

- [ ] **Step 7: Create envs/gym_spaces.py**

Copy `/Users/adityamohan/git/Austin/td_jepa/metamotivo/envs/utils/gym_spaces.py` verbatim to `envs/gym_spaces.py`. No import changes (only uses `gymnasium.spaces`, `numpy`, `typing`).

- [ ] **Step 8: Run tests to verify they pass**

```bash
python -m pytest tests/test_core.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 9: Commit**

```bash
git add base_config.py base_model.py normalizers.py utils.py envs/gym_spaces.py tests/test_core.py
git commit -m "feat: core utilities — base_config, base_model, normalizers, utils, gym_spaces"
```

---

## Task 3: Neural network architectures

**Files:**
- Create: `nn_models.py` (copy + 1 import change + append VectorField)
- Test: `tests/test_nn_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_nn_models.py
import torch
import numpy as np
import gymnasium

def _obs_space(dim):
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

def test_backward_map():
    from nn_models import BackwardArchiConfig
    cfg = BackwardArchiConfig(hidden_dim=32, hidden_layers=1)
    net = cfg.build(_obs_space(10), z_dim=8)
    x = torch.randn(4, 10)
    out = net(x)
    assert out.shape == (4, 8)

def test_forward_map():
    from nn_models import ForwardArchiConfig
    cfg = ForwardArchiConfig(hidden_dim=32, hidden_layers=1, num_parallel=2)
    net = cfg.build(_obs_space(10), z_dim=8, action_dim=4)
    obs = torch.randn(4, 10)
    z = torch.randn(4, 8)
    act = torch.randn(4, 4)
    out = net(obs, z, act)
    assert out.shape == (2, 4, 8)  # (num_parallel, batch, z_dim)

def test_simple_actor():
    from nn_models import SimpleActorArchiConfig
    cfg = SimpleActorArchiConfig(hidden_dim=32, hidden_layers=1, embedding_layers=2)
    net = cfg.build(_obs_space(10), z_dim=8, action_dim=4)
    obs = torch.randn(4, 10)
    z = torch.randn(4, 8)
    dist = net(obs, z, std=0.2)
    action = dist.sample()
    assert action.shape == (4, 4)

def test_noise_conditioned_actor():
    from nn_models import NoiseConditionedActorArchiConfig
    cfg = NoiseConditionedActorArchiConfig(hidden_dim=32, hidden_layers=1, embedding_layers=2)
    net = cfg.build(_obs_space(10), z_dim=8, action_dim=4)
    obs = torch.randn(4, 10)
    z = torch.randn(4, 8)
    noise = torch.randn(4, 4)
    out = net(obs, z, noise)
    assert out.shape == (4, 4)

def test_vector_field():
    from nn_models import SimpleVectorFieldArchiConfig
    cfg = SimpleVectorFieldArchiConfig(hidden_dim=32, hidden_layers=1)
    net = cfg.build(_obs_space(10), action_dim=4)
    obs = torch.randn(4, 10)
    action = torch.randn(4, 4)
    t = torch.rand(4, 1)
    out = net(obs, action, t)
    assert out.shape == (4, 4)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_nn_models.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create nn_models.py**

Copy `/Users/adityamohan/git/Austin/td_jepa/metamotivo/nn_models.py` to `nn_models.py` with one import change at the top:

```python
# CHANGE from:
from .base import BaseConfig

# TO:
from base_config import BaseConfig
```

Then **append** the following to the end of `nn_models.py` (taken from `td_jepa/metamotivo/agents/td3/nn_models.py`):

```python
##########################
# Vector field (flow matching)
##########################


class SimpleVectorFieldArchiConfig(BaseConfig):
    name: tp.Literal["SimpleVectorFieldArchi"] = "SimpleVectorFieldArchi"
    model: tp.Literal["simple"] = "simple"
    hidden_dim: int = 1024
    hidden_layers: int = 1

    def build(self, obs_space, action_dim: int) -> "VectorField":
        return VectorField(obs_space, action_dim, self)


class VectorField(nn.Module):
    def __init__(self, obs_space, action_dim, cfg: SimpleVectorFieldArchiConfig) -> None:
        super().__init__()
        self.cfg: SimpleVectorFieldArchiConfig = cfg

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        # +1 for the time dimension t
        seq = [linear(obs_dim + action_dim + 1, cfg.hidden_dim), nn.GELU()]
        for _ in range(cfg.hidden_layers - 1):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU()]
        seq += [linear(cfg.hidden_dim, action_dim)]
        self.net = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        embedding = torch.cat([obs, action, t], dim=-1)
        return self.net(embedding)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_nn_models.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add nn_models.py tests/test_nn_models.py
git commit -m "feat: nn_models — forward/backward maps, actor, noise-conditioned actor, vector field"
```

---

## Task 4: Replay buffer

**Files:**
- Create: `buffers/transition.py` (copy, no changes)
- Test: `tests/test_buffers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_buffers.py
import numpy as np
import torch

def test_dict_buffer_extend_and_sample():
    from buffers.transition import DictBuffer
    buf = DictBuffer(capacity=100, device="cpu")
    data = {
        "observation": np.random.randn(50, 10).astype(np.float32),
        "action": np.random.randn(50, 4).astype(np.float32),
        "next": {
            "observation": np.random.randn(50, 10).astype(np.float32),
            "terminated": np.zeros((50, 1), dtype=bool),
        },
    }
    buf.extend(data)
    assert len(buf) == 50

    batch = buf.sample(16)
    assert batch["observation"].shape == (16, 10)
    assert batch["action"].shape == (16, 4)
    assert batch["next"]["observation"].shape == (16, 10)

def test_dict_buffer_wraps_on_overflow():
    from buffers.transition import DictBuffer
    buf = DictBuffer(capacity=30, device="cpu")
    data = {
        "observation": np.random.randn(50, 4).astype(np.float32),
        "x": np.random.randn(50, 2).astype(np.float32),
    }
    buf.extend(data)
    assert len(buf) == 30  # capped at capacity
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_buffers.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create buffers/transition.py**

Copy `/Users/adityamohan/git/Austin/td_jepa/metamotivo/buffers/transition.py` verbatim to `buffers/transition.py`. No import changes needed (only uses `h5py`, `numpy`, `torch`, `dataclasses`, and standard library — no metamotivo imports).

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_buffers.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add buffers/transition.py tests/test_buffers.py
git commit -m "feat: DictBuffer for offline replay storage"
```

---

## Task 5: Environment and rollout utilities

**Files:**
- Create: `envs/wrappers.py` (copy)
- Create: `envs/rollout.py` (copy)
- Create: `envs/ogbench.py` (copy + simplify)
- Test: `tests/test_envs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_envs.py
def test_domain_lists():
    from envs.ogbench import ANT_DOMAINS, CUBE_DOMAINS, ALL_DOMAINS, ALL_TASKS
    assert "antmaze-medium-navigate-v0" in ANT_DOMAINS
    assert "cube-single-play-v0" in CUBE_DOMAINS
    assert len(ALL_TASKS) == len(ALL_DOMAINS)
    # Each domain has 5 tasks
    for domain, tasks in ALL_TASKS.items():
        assert len(tasks) == 5, f"{domain} should have 5 tasks, got {len(tasks)}"

def test_rollout_import():
    from envs.rollout import rollout
    assert callable(rollout)

def test_pixel_wrapper_import():
    from envs.wrappers import PixelWrapper
    assert PixelWrapper is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_envs.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create envs/wrappers.py**

Copy `/Users/adityamohan/git/Austin/td_jepa/metamotivo/envs/utils/wrappers.py` verbatim to `envs/wrappers.py`. No import changes.

- [ ] **Step 4: Create envs/rollout.py**

Copy `/Users/adityamohan/git/Austin/td_jepa/metamotivo/envs/utils/rollout.py` verbatim to `envs/rollout.py`. No import changes.

- [ ] **Step 5: Create envs/ogbench.py**

Create `envs/ogbench.py` with the following content (based on `td_jepa/metamotivo/envs/ogbench.py`, with `OGBenchEnvConfig` removed and `get_relabel_fn` added as a standalone function):

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# CC BY-NC 4.0 license

import typing as tp
from functools import partial

import gymnasium
import numpy as np
from ogbench.utils import make_env_and_datasets

from envs.wrappers import PixelWrapper

CUBE_DOMAINS = ["cube-single-play-v0", "cube-double-play-v0"]
PUZZLE_DOMAINS = ["puzzle-3x3-play-v0"]
SCENE_DOMAINS = ["scene-play-v0"]
ANT_DOMAINS = []
for size in ["medium", "large", "giant"]:
    for data_type in ["navigate", "stitch", "explore"]:
        ANT_DOMAINS += [f"antmaze-{size}-{data_type}-v0"]
ALL_DOMAINS = CUBE_DOMAINS + PUZZLE_DOMAINS + SCENE_DOMAINS + ANT_DOMAINS
ALL_TASKS = {}
for d in ALL_DOMAINS:
    ALL_TASKS[d] = [d[:-2] + f"singletask-task{i + 1}-" + d[-2:] for i in range(5)]


def cube_reward_fn(qpos: np.ndarray, action: np.ndarray, *, target_position: np.ndarray, threshold: float = 0.04) -> np.ndarray:
    num_cubes = target_position.shape[0]
    cube_positions = [qpos[..., 14:17], qpos[..., 21:24], qpos[..., 28:31], qpos[..., 35:38]][:num_cubes]
    distances = [np.linalg.norm(cpos - tpos, axis=-1) for cpos, tpos in zip(cube_positions, target_position)]
    successes = sum([(d < threshold).astype(float) for d in distances])
    return (successes - num_cubes).reshape(-1, 1)


def puzzle_reward_fn(qpos: np.ndarray, action: np.ndarray, *, target_position: np.ndarray) -> np.ndarray:
    return (qpos[:, -len(target_position):] == target_position).sum(-1, keepdims=True).astype(float) - len(target_position)


def scene_reward_fn(qpos: np.ndarray, action: np.ndarray, *, target_position: np.ndarray, threshold: float = 0.04) -> np.ndarray:
    cube = (np.linalg.norm(qpos[..., 14:17] - target_position[:3], axis=-1) <= threshold).astype(float)
    button1 = (qpos[..., -2] == target_position[3]).astype(float)
    button2 = (qpos[..., -1] == target_position[4]).astype(float)
    drawer = (np.abs(qpos[..., -4] - target_position[5]) <= threshold).astype(float)
    window = (np.abs(qpos[..., -3] - target_position[6]) <= threshold).astype(float)
    return (cube + button1 + button2 + drawer + window - 5).reshape(-1, 1)


def ant_reward_fn(qpos: np.ndarray, action: np.ndarray, *, target_position: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.linalg.norm(qpos[..., :2] - target_position, axis=-1).reshape(-1, 1) <= threshold) - 1.0


def create_ogbench_env(
    task: str,
    wrappers: tp.List = None,
    seed: tp.Optional[int] = None,
    render_height: int = 64,
    render_width: int = 64,
    obs_type: str = "state",
):
    if wrappers is None:
        wrappers = []
    match obs_type:
        case "state":
            pass
        case "pixels":
            task = "visual-" + task
        case _:
            raise ValueError(f"Unsupported observation type {obs_type}")
    env_kwargs = {"height": render_height, "width": render_width}
    env = make_env_and_datasets(task, env_only=True, **env_kwargs)
    for wrapper in wrappers:
        env = wrapper(env)
    env.reset(seed=seed)
    return env, {}


def get_relabel_fn(domain: str, task: str):
    """Return a reward function (qpos, action) -> reward for a specific task."""
    env = make_env_and_datasets(task, env_only=True)
    env.reset()  # required for antmaze to initialize goal
    if domain in CUBE_DOMAINS:
        return partial(cube_reward_fn, target_position=env.unwrapped.cur_task_info["goal_xyzs"])
    if domain in PUZZLE_DOMAINS:
        return partial(puzzle_reward_fn, target_position=env.unwrapped.cur_task_info["goal_button_states"])
    if domain in SCENE_DOMAINS:
        return partial(
            scene_reward_fn,
            target_position=np.concatenate([np.array([v]).ravel() for v in env.unwrapped.cur_task_info["goal"].values()]),
        )
    if domain in ANT_DOMAINS:
        return partial(ant_reward_fn, target_position=env.unwrapped.get_oracle_rep())
    raise NotImplementedError(f"Unknown domain: {domain}")
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/test_envs.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add envs/wrappers.py envs/rollout.py envs/ogbench.py tests/test_envs.py
git commit -m "feat: env utilities — OGBench wrapper, rollout, reward relabelers"
```

---

## Task 6: Offline data loading

**Files:**
- Create: `data/ogbench.py` (new)
- Test: `tests/test_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data.py
# These tests use synthetic .npz files — no real OGBench data needed.
import numpy as np
import tempfile
from pathlib import Path


def _make_fake_episode(obs_dim=10, action_dim=4, physics_dim=5, n=50):
    return {
        "observation": np.random.randn(n, obs_dim).astype(np.float32),
        "action": np.random.randn(n, action_dim).astype(np.float32),
        "physics": np.random.randn(n, physics_dim).astype(np.float32),
        "discount": np.ones(n, dtype=np.float32),
    }


def test_load_transitions():
    from data.ogbench import load_transitions
    with tempfile.TemporaryDirectory() as tmpdir:
        ep = _make_fake_episode()
        fname = Path(tmpdir) / "ep0.npz"
        np.savez(fname, **ep)

        storage = load_transitions([fname])
        assert storage["observation"].shape == (49, 10)   # N-1 transitions
        assert storage["action"].shape == (49, 4)
        assert "next" in storage
        assert storage["next"]["observation"].shape == (49, 10)
        assert storage["next"]["terminated"].shape == (49,)


def test_load_ogbench_data():
    from data.ogbench import load_ogbench_data
    with tempfile.TemporaryDirectory() as tmpdir:
        buf_dir = Path(tmpdir) / "antmaze-medium-navigate-v0" / "buffer"
        buf_dir.mkdir(parents=True)
        for i in range(3):
            np.savez(buf_dir / f"ep{i}.npz", **_make_fake_episode())

        replay_buffer = load_ogbench_data(
            domain="antmaze-medium-navigate-v0",
            data_path=tmpdir,
            load_n_episodes=2,
            device="cpu",
        )
        assert "train" in replay_buffer
        assert len(replay_buffer["train"]) == 2 * 49  # 2 episodes × 49 transitions
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_data.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create data/ogbench.py**

```python
# data/ogbench.py
"""Offline data loading for OGBench state-observation tasks."""

import typing as tp
from pathlib import Path

import numpy as np
import torch
from torch.utils._pytree import tree_map

from buffers.transition import DictBuffer


def load_transitions(
    episode_fns: tp.List,
    obs_type: tp.Literal["state", "pixels"] = "state",
) -> dict:
    """Load a list of .npz episode files into a flat transition dict."""
    storage = {
        "observation": [],
        "action": [],
        "physics": [],
        "next": {"observation": [], "terminated": [], "physics": []},
    }
    for f in episode_fns:
        data = np.load(str(f))
        match obs_type:
            case "state":
                obs_key = "observation"
            case "pixels":
                obs_key = "pixels"
            case _:
                raise ValueError(f"Unknown observation type {obs_type}")
        storage["observation"].append(data[obs_key][:-1].astype(np.float32))
        storage["next"]["observation"].append(data[obs_key][1:].astype(np.float32))
        storage["action"].append(data["action"][1:].astype(np.float32))
        storage["next"]["terminated"].append(np.array(1 - data["discount"][1:], dtype=bool))
        storage["physics"].append(data["physics"][:-1])
        storage["next"]["physics"].append(data["physics"][1:])

    storage = tree_map(
        lambda x: np.concatenate(x),
        storage,
        is_leaf=lambda x: isinstance(x, list),
    )
    return storage


def load_ogbench_data(
    domain: str,
    data_path: str,
    load_n_episodes: int = 1_000,
    device: str = "cpu",
) -> tp.Dict[str, DictBuffer]:
    """Load offline OGBench state-obs data from .npz files into a DictBuffer.

    Args:
        domain: OGBench domain string, e.g. "antmaze-medium-navigate-v0".
        data_path: Root directory containing domain subdirectories.
        load_n_episodes: Maximum number of episodes to load.
        device: PyTorch device for the buffer ("cpu" or "cuda").

    Returns:
        {"train": DictBuffer} with keys observation, action, physics, next/{...}.
    """
    path = Path(data_path) / f"{domain}" / "buffer"
    files = sorted(path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {path}")
    num_episodes = min(load_n_episodes, len(files))
    print(f"Loading {num_episodes}/{len(files)} episodes from {path}")
    data = load_transitions(files[:num_episodes], obs_type="state")
    capacity = data["observation"].shape[0]
    replay_buffer: DictBuffer = DictBuffer(capacity=capacity, device=device)
    replay_buffer.extend(data)
    del data
    return {"train": replay_buffer}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_data.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add data/ogbench.py tests/test_data.py
git commit -m "feat: offline OGBench data loader (state obs, DictBuffer)"
```

---

## Task 7: FB model

**Files:**
- Create: `agents/fb/model.py` (copy + 4 import changes)
- Test: `tests/test_fb_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fb_model.py
import numpy as np
import torch
import gymnasium


def _obs_space(dim=10):
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def test_fb_model_instantiation():
    from nn_models import ForwardArchiConfig, BackwardArchiConfig, SimpleActorArchiConfig, IdentityNNConfig
    from normalizers import IdentityNormalizerConfig
    from agents.fb.model import FBModel, FBModelArchiConfig, FBModelConfig

    archi = FBModelArchiConfig(
        L_dim=8,
        z_dim=8,
        f=ForwardArchiConfig(hidden_dim=32, hidden_layers=1),
        b=BackwardArchiConfig(hidden_dim=32, hidden_layers=1),
        left_encoder=IdentityNNConfig(),
    )
    cfg = FBModelConfig(device="cpu", archi=archi)
    model = cfg.build(_obs_space(10), action_dim=4)
    assert model is not None


def test_fb_model_reward_inference():
    from nn_models import ForwardArchiConfig, BackwardArchiConfig, IdentityNNConfig
    from agents.fb.model import FBModel, FBModelArchiConfig, FBModelConfig

    archi = FBModelArchiConfig(
        L_dim=8, z_dim=8,
        f=ForwardArchiConfig(hidden_dim=32, hidden_layers=1),
        b=BackwardArchiConfig(hidden_dim=32, hidden_layers=1),
        left_encoder=IdentityNNConfig(),
    )
    cfg = FBModelConfig(device="cpu", archi=archi, inference_batch_size=50)
    model = cfg.build(_obs_space(10), action_dim=4)
    model._prepare_for_train()

    next_obs = torch.randn(100, 10)
    reward = torch.randn(100, 1)
    z = model.reward_inference(next_obs, reward)
    assert z.shape == (1, 8)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_fb_model.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create agents/fb/model.py**

Copy `/Users/adityamohan/git/Austin/td_jepa/metamotivo/agents/fb/model.py` to `agents/fb/model.py` with these import changes:

```python
# CHANGE from:
import copy
import math
import typing as tp

import numpy as np
import pydantic
import torch
import torch.nn.functional as F
from torch.amp import autocast

from ...base import BaseConfig
from ...base_model import BaseModel, BaseModelConfig, load_model, save_model
from ...nn_models import (
    ActorArchiConfig,
    AugmentatorArchiConfig,
    BackwardArchiConfig,
    DrQEncoderArchiConfig,
    ForwardArchiConfig,
    IdentityNNConfig,
    SimpleActorArchiConfig,
    eval_mode,
)
from ...normalizers import AVAILABLE_NORMALIZERS, IdentityNormalizerConfig

# TO:
import copy
import math
import typing as tp

import numpy as np
import pydantic
import torch
import torch.nn.functional as F
from torch.amp import autocast

from base_config import BaseConfig
from base_model import BaseModel, BaseModelConfig, load_model, save_model
from nn_models import (
    ActorArchiConfig,
    AugmentatorArchiConfig,
    BackwardArchiConfig,
    DrQEncoderArchiConfig,
    ForwardArchiConfig,
    IdentityNNConfig,
    SimpleActorArchiConfig,
    eval_mode,
)
from normalizers import AVAILABLE_NORMALIZERS, IdentityNormalizerConfig
```

The rest of the file (FBModelArchiConfig, FBModelConfig, FBModel) is verbatim.

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_fb_model.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/fb/model.py tests/test_fb_model.py
git commit -m "feat: FBModel — forward/backward/left encoder + reward inference"
```

---

## Task 8: FB agent

**Files:**
- Create: `agents/fb/agent.py` (copy + 3 import changes)
- Test: `tests/test_fb_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fb_agent.py
import numpy as np
import torch
import gymnasium


def _obs_space(dim=10):
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def _make_fake_buffer(obs_dim=10, action_dim=4, physics_dim=5, n=100, batch_size=16):
    from buffers.transition import DictBuffer
    buf = DictBuffer(capacity=n, device="cpu")
    buf.extend({
        "observation": np.random.randn(n, obs_dim).astype(np.float32),
        "action": np.random.randn(n, action_dim).astype(np.float32),
        "physics": np.random.randn(n, physics_dim).astype(np.float32),
        "next": {
            "observation": np.random.randn(n, obs_dim).astype(np.float32),
            "terminated": np.zeros((n, 1), dtype=bool),
            "physics": np.random.randn(n, physics_dim).astype(np.float32),
        },
    })
    return {"train": buf}


def _build_fb_agent(obs_dim=10, action_dim=4, z_dim=8, L_dim=8, batch_size=16):
    from nn_models import ForwardArchiConfig, BackwardArchiConfig, SimpleActorArchiConfig, IdentityNNConfig
    from agents.fb.model import FBModelArchiConfig, FBModelConfig
    from agents.fb.agent import FBAgent, FBAgentConfig, FBAgentTrainConfig

    archi = FBModelArchiConfig(
        L_dim=L_dim, z_dim=z_dim, norm_z=True,
        f=ForwardArchiConfig(hidden_dim=32, hidden_layers=1),
        b=BackwardArchiConfig(hidden_dim=32, hidden_layers=1),
        left_encoder=IdentityNNConfig(),
    )
    model_cfg = FBModelConfig(device="cpu", archi=archi, actor_encode_obs=False, actor_std=0.2)
    train_cfg = FBAgentTrainConfig(batch_size=batch_size, discount=0.99, ortho_coef=1.0)
    return FBAgent(_obs_space(obs_dim), action_dim, FBAgentConfig(model=model_cfg, train=train_cfg))


def test_fb_agent_instantiates():
    agent = _build_fb_agent()
    assert agent is not None
    assert agent.device == "cpu"


def test_fb_agent_update_returns_metrics():
    agent = _build_fb_agent()
    replay_buffer = _make_fake_buffer()
    metrics = agent.update(replay_buffer, step=0)
    assert "fb_loss" in metrics
    assert "fb_diag" in metrics
    assert "fb_offdiag" in metrics
    assert "orth_loss" in metrics
    assert "actor_loss" in metrics


def test_fb_agent_act():
    agent = _build_fb_agent()
    obs = torch.randn(1, 10)
    z = agent._model.sample_z(1, device="cpu")
    action = agent.act(obs, z, mean=True)
    assert action.shape == (1, 4)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_fb_agent.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create agents/fb/agent.py**

Copy `/Users/adityamohan/git/Austin/td_jepa/metamotivo/agents/fb/agent.py` to `agents/fb/agent.py` with these import changes:

```python
# CHANGE from:
import json
import pickle
from pathlib import Path
from typing import Dict, Literal, Tuple

import safetensors
import torch
import torch.nn.functional as F
from torch.amp import autocast

from ...base import BaseConfig
from ...envs.utils.gym_spaces import json_to_space, space_to_json
from ...nn_models import _soft_update_params, eval_mode, weight_init
from .model import FBModel, FBModelConfig

# TO:
import json
import pickle
from pathlib import Path
from typing import Dict, Literal, Tuple

import safetensors
import torch
import torch.nn.functional as F
from torch.amp import autocast

from base_config import BaseConfig
from envs.gym_spaces import json_to_space, space_to_json
from nn_models import _soft_update_params, eval_mode, weight_init
from .model import FBModel, FBModelConfig
```

The rest of the file (FBAgentTrainConfig, FBAgentConfig, FBAgent) is verbatim.

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_fb_agent.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/fb/agent.py tests/test_fb_agent.py
git commit -m "feat: FBAgent — FB loss, ortho constraint, TD3 actor update"
```

---

## Task 9: FlowBC model and agent

**Files:**
- Create: `agents/fb/flow_bc/model.py` (copy + 3 import changes)
- Create: `agents/fb/flow_bc/agent.py` (copy, no changes)
- Test: `tests/test_fb_flowbc.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fb_flowbc.py
import numpy as np
import torch
import gymnasium


def _obs_space(dim=10):
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def _make_fake_buffer(obs_dim=10, action_dim=4, n=100):
    from buffers.transition import DictBuffer
    buf = DictBuffer(capacity=n, device="cpu")
    buf.extend({
        "observation": np.random.randn(n, obs_dim).astype(np.float32),
        "action": np.random.randn(n, action_dim).astype(np.float32),
        "physics": np.random.randn(n, 5).astype(np.float32),
        "next": {
            "observation": np.random.randn(n, obs_dim).astype(np.float32),
            "terminated": np.zeros((n, 1), dtype=bool),
            "physics": np.random.randn(n, 5).astype(np.float32),
        },
    })
    return {"train": buf}


def _build_flowbc_agent(obs_dim=10, action_dim=4, z_dim=8, L_dim=8, batch_size=16):
    from nn_models import (
        ForwardArchiConfig, BackwardArchiConfig,
        NoiseConditionedActorArchiConfig, IdentityNNConfig, SimpleVectorFieldArchiConfig,
    )
    from agents.fb.flow_bc.model import FBFlowBCModelArchiConfig, FBFlowBCModelConfig
    from agents.fb.flow_bc.agent import FBFlowBCAgent, FBFlowBCAgentConfig, FBFlowBCAgentTrainConfig

    archi = FBFlowBCModelArchiConfig(
        L_dim=L_dim, z_dim=z_dim, norm_z=True,
        f=ForwardArchiConfig(hidden_dim=32, hidden_layers=1),
        b=BackwardArchiConfig(hidden_dim=32, hidden_layers=1),
        left_encoder=IdentityNNConfig(),
        actor=NoiseConditionedActorArchiConfig(hidden_dim=32, hidden_layers=1, embedding_layers=2),
        actor_vf=SimpleVectorFieldArchiConfig(hidden_dim=32, hidden_layers=1),
    )
    model_cfg = FBFlowBCModelConfig(device="cpu", archi=archi, actor_encode_obs=False)
    train_cfg = FBFlowBCAgentTrainConfig(
        batch_size=batch_size, discount=0.99, bc_coeff=0.3, flow_steps=3,
    )
    return FBFlowBCAgent(_obs_space(obs_dim), action_dim, FBFlowBCAgentConfig(model=model_cfg, train=train_cfg))


def test_flowbc_agent_instantiates():
    agent = _build_flowbc_agent()
    assert agent is not None


def test_flowbc_agent_update_returns_metrics():
    agent = _build_flowbc_agent()
    replay_buffer = _make_fake_buffer()
    metrics = agent.update(replay_buffer, step=0)
    assert "fb_loss" in metrics
    assert "actor_loss" in metrics
    assert "bc_flow_loss" in metrics


def test_flowbc_agent_act():
    agent = _build_flowbc_agent()
    obs = torch.randn(1, 10)
    z = agent._model.sample_z(1, device="cpu")
    action = agent.act(obs, z)
    assert action.shape == (1, 4)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_fb_flowbc.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create agents/fb/flow_bc/model.py**

Copy `/Users/adityamohan/git/Austin/td_jepa/metamotivo/agents/fb/flow_bc/model.py` to `agents/fb/flow_bc/model.py` with these import changes:

```python
# CHANGE from:
import typing as tp

import gymnasium
import numpy as np
import pydantic
import torch
from torch.amp import autocast

from ....base_model import load_model
from ....nn_models import NoiseConditionedActorArchiConfig
from ...td3.nn_models import SimpleVectorFieldArchiConfig
from ..model import FBModel, FBModelArchiConfig, FBModelConfig

# TO:
import typing as tp

import gymnasium
import numpy as np
import pydantic
import torch
from torch.amp import autocast

from base_model import load_model
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig
from ..model import FBModel, FBModelArchiConfig, FBModelConfig
```

The rest of the file (FBFlowBCModelArchiConfig, FBFlowBCModelConfig, FBFlowBCModel) is verbatim.

- [ ] **Step 4: Create agents/fb/flow_bc/agent.py**

Copy `/Users/adityamohan/git/Austin/td_jepa/metamotivo/agents/fb/flow_bc/agent.py` verbatim to `agents/fb/flow_bc/agent.py`. No import changes needed — the file already uses relative imports:

```python
from ..agent import FBAgent, FBAgentConfig, FBAgentTrainConfig
from .model import FBFlowBCModelConfig
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_fb_flowbc.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add agents/fb/flow_bc/model.py agents/fb/flow_bc/agent.py tests/test_fb_flowbc.py
git commit -m "feat: FBFlowBCAgent — flow-matching actor for manipulation tasks"
```

---

## Task 10: Evaluation

**Files:**
- Create: `evals/ogbench.py` (new)
- Test: `tests/test_evals.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evals.py

def test_evaluator_import():
    from evals.ogbench import OGBenchEvaluator
    assert OGBenchEvaluator is not None

def test_evaluator_instantiation():
    from evals.ogbench import OGBenchEvaluator
    ev = OGBenchEvaluator(
        domain="antmaze-medium-navigate-v0",
        tasks=["antmaze-medium-singletask-task1-v0"],
        num_episodes=2,
        num_inference_samples=100,
        seed=0,
    )
    assert ev.domain == "antmaze-medium-navigate-v0"
    assert len(ev.tasks) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_evals.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create evals/ogbench.py**

```python
# evals/ogbench.py
"""Offline evaluation for OGBench: reward inference + rollout."""

import typing as tp

import numpy as np
import torch
import tqdm

from envs.ogbench import create_ogbench_env, get_relabel_fn
from envs.rollout import rollout
from nn_models import eval_mode


class OGBenchEvaluator:
    """Evaluates a trained FB agent on OGBench tasks via reward inference + rollout.

    For each task:
    1. Sample a batch from the replay buffer, relabel with ground-truth rewards.
    2. Run reward_inference to get a task-conditioned z.
    3. Roll out the policy conditioned on z for num_episodes episodes.
    4. Record success rate and mean episode return.
    """

    def __init__(
        self,
        domain: str,
        tasks: tp.List[str],
        num_episodes: int = 10,
        num_inference_samples: int = 50_000,
        seed: int = 0,
    ):
        self.domain = domain
        self.tasks = tasks
        self.num_episodes = num_episodes
        self.num_inference_samples = num_inference_samples
        self.seed = seed

    def run(
        self,
        *,
        timestep: int,
        agent,
        replay_buffer: tp.Dict,
    ) -> tp.Tuple[tp.Dict, tp.Dict]:
        """Run evaluation over all tasks.

        Returns:
            eval_metrics: per-task dict of {"reward": [...], "success": [...]}
            wandb_dict: flat dict suitable for logging to W&B
        """
        wandb_dict: tp.Dict[str, float] = {}
        eval_metrics: tp.Dict[str, tp.Dict] = {}
        model = agent._model

        for task in tqdm.tqdm(self.tasks, desc=f"Eval @ step {timestep}"):
            eval_env, _ = create_ogbench_env(task, wrappers=[], seed=self.seed)

            ctx, relabel_metrics = self._reward_inference(agent, task, replay_buffer)

            with torch.no_grad(), eval_mode(model):
                st, infos = rollout(
                    eval_env,
                    agent=model,
                    num_episodes=self.num_episodes,
                    ctx=ctx,
                )
            st["success"] = [
                any(step.get("success", False) for step in info) for info in infos
            ]

            eval_metrics[task] = st
            wandb_dict[f"{task}/reward"] = float(np.mean(st["reward"]))
            wandb_dict[f"{task}/success"] = float(np.mean(st["success"]))
            for k, v in relabel_metrics.items():
                wandb_dict[f"{task}/{k}"] = v

            eval_env.close()

        all_rewards = np.concatenate([v["reward"] for v in eval_metrics.values()])
        all_successes = np.concatenate([v["success"] for v in eval_metrics.values()])
        wandb_dict["eval/reward"] = float(np.mean(all_rewards))
        wandb_dict["eval/success"] = float(np.mean(all_successes))

        return eval_metrics, wandb_dict

    def _reward_inference(
        self,
        agent,
        task: str,
        replay_buffer: tp.Dict,
    ) -> tp.Tuple[tp.Optional[torch.Tensor], tp.Dict]:
        if not hasattr(agent._model, "reward_inference"):
            return None, {}

        batch = replay_buffer["train"].sample(self.num_inference_samples)
        relabel_fn = get_relabel_fn(self.domain, task)
        rewards = relabel_fn(
            batch["next"]["physics"].cpu().numpy(),
            batch["action"].cpu().numpy(),
        )
        non_zero = int(np.count_nonzero(rewards.ravel()))
        relabel_metrics = {
            "relabel_reward#mean": float(np.mean(rewards)),
            "relabel_reward#nonzero": non_zero,
        }
        z = agent._model.reward_inference(
            next_obs=batch["next"]["observation"].to(agent.device),
            reward=torch.tensor(rewards, dtype=torch.float32, device=agent.device),
        )
        return z.reshape(1, -1), relabel_metrics
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_evals.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add evals/ogbench.py tests/test_evals.py
git commit -m "feat: OGBenchEvaluator — reward inference + rollout evaluation"
```

---

## Task 11: Training entry point

**Files:**
- Create: `train.py` (new)
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
import subprocess
import sys


def test_train_help():
    """Verify train.py parses --help without error."""
    result = subprocess.run(
        [sys.executable, "train.py", "--help"],
        capture_output=True, text=True,
        cwd="/Users/adityamohan/git/Austin/Factored-FB",
    )
    assert result.returncode == 0
    assert "--domain" in result.stdout
    assert "--agent" in result.stdout


def test_argparse_defaults():
    """Verify default argument values are set correctly."""
    import sys
    sys.argv = ["train.py", "--domain", "antmaze-medium-navigate-v0", "--data_path", "/tmp/data"]
    from train import parse_args
    args = parse_args()
    assert args.agent == "fb"
    assert args.z_dim == 50
    assert args.batch_size == 256
    assert args.discount == 0.99
    assert args.norm_z is True
    assert args.actor_encode_obs is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_train.py -v
```

Expected: FAIL (train.py doesn't exist yet).

- [ ] **Step 3: Create train.py**

```python
#!/usr/bin/env python3
"""Train ForwardBackward (FB) representations on OGBench offline datasets.

Usage (antmaze):
    python train.py --domain antmaze-medium-navigate-v0 --data_path datasets/

Usage (cube manipulation):
    python train.py --domain cube-single-play-v0 --data_path datasets/ \
        --agent fb_flowbc --bc_coeff 0.3 --ortho_coef 100

Future: replace argparse with @hydra.main(config_path="configs", config_name="train")
and move defaults to configs/train.yaml.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MUJOCO_GL"] = "egl"

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from utils import set_seed_everywhere, EveryNStepsChecker
from envs.ogbench import ALL_TASKS, create_ogbench_env
from data.ogbench import load_ogbench_data
from evals.ogbench import OGBenchEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Train FB representations on OGBench")

    # Required
    parser.add_argument("--domain", type=str, required=True,
                        help="OGBench domain, e.g. antmaze-medium-navigate-v0")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Root directory for offline datasets")

    # Agent type
    parser.add_argument("--agent", type=str, default="fb", choices=["fb", "fb_flowbc"],
                        help="Agent variant: fb (antmaze) or fb_flowbc (manipulation)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Training schedule
    parser.add_argument("--num_train_steps", type=int, default=1_000_000)
    parser.add_argument("--eval_every", type=int, default=50_000)
    parser.add_argument("--log_every", type=int, default=10_000)
    parser.add_argument("--checkpoint_every", type=int, default=250_000)
    parser.add_argument("--work_dir", type=str, default="workdir")

    # Data
    parser.add_argument("--load_n_episodes", type=int, default=1_000)

    # Architecture
    parser.add_argument("--z_dim", type=int, default=50)
    parser.add_argument("--L_dim", type=int, default=50)
    parser.add_argument("--f_hidden_dim", type=int, default=512)
    parser.add_argument("--f_hidden_layers", type=int, default=2)
    parser.add_argument("--b_hidden_dim", type=int, default=512)
    parser.add_argument("--b_hidden_layers", type=int, default=4)
    parser.add_argument("--left_encoder_hidden_dim", type=int, default=512)
    parser.add_argument("--left_encoder_hidden_layers", type=int, default=4)
    parser.add_argument("--norm_z", action="store_true", default=True)
    parser.add_argument("--no_norm_z", dest="norm_z", action="store_false")
    parser.add_argument("--actor_encode_obs", action="store_true", default=False)
    parser.add_argument("--actor_std", type=float, default=0.2)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--lr_f", type=float, default=1e-4)
    parser.add_argument("--lr_b", type=float, default=1e-4)
    parser.add_argument("--lr_actor", type=float, default=1e-4)
    parser.add_argument("--ortho_coef", type=float, default=1.0)
    parser.add_argument("--f_target_tau", type=float, default=0.005)
    parser.add_argument("--b_target_tau", type=float, default=0.005)
    parser.add_argument("--train_goal_ratio", type=float, default=0.5)
    parser.add_argument("--bc_coeff", type=float, default=0.0)
    parser.add_argument("--fb_pessimism_penalty", type=float, default=0.0)
    parser.add_argument("--actor_pessimism_penalty", type=float, default=0.0)
    parser.add_argument("--stddev_clip", type=float, default=0.3)
    parser.add_argument("--q_loss_coef", type=float, default=0.0)
    parser.add_argument("--clip_grad_norm", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # FlowBC-specific
    parser.add_argument("--flow_steps", type=int, default=10)
    parser.add_argument("--lr_actor_vf", type=float, default=3e-4)

    # Evaluation
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--num_inference_samples", type=int, default=50_000)

    # Logging
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="factored-fb")
    parser.add_argument("--wandb_group", type=str, default=None)

    return parser.parse_args()


def build_agent(args, obs_space, action_dim):
    """Construct FBAgent or FBFlowBCAgent from parsed args."""
    from nn_models import (
        ForwardArchiConfig,
        BackwardArchiConfig,
        SimpleActorArchiConfig,
        NoiseConditionedActorArchiConfig,
        IdentityNNConfig,
        SimpleVectorFieldArchiConfig,
    )

    f_cfg = ForwardArchiConfig(hidden_dim=args.f_hidden_dim, hidden_layers=args.f_hidden_layers)
    b_cfg = BackwardArchiConfig(hidden_dim=args.b_hidden_dim, hidden_layers=args.b_hidden_layers, norm=True)
    left_enc_cfg = BackwardArchiConfig(
        hidden_dim=args.left_encoder_hidden_dim,
        hidden_layers=args.left_encoder_hidden_layers,
        norm=True,
    )

    if args.agent == "fb":
        from agents.fb.model import FBModelArchiConfig, FBModelConfig
        from agents.fb.agent import FBAgent, FBAgentConfig, FBAgentTrainConfig

        archi = FBModelArchiConfig(
            L_dim=args.L_dim, z_dim=args.z_dim, norm_z=args.norm_z,
            f=f_cfg, b=b_cfg, left_encoder=left_enc_cfg,
            actor=SimpleActorArchiConfig(),
        )
        model_cfg = FBModelConfig(
            device=args.device, archi=archi,
            actor_encode_obs=args.actor_encode_obs, actor_std=args.actor_std,
        )
        train_cfg = FBAgentTrainConfig(
            lr_f=args.lr_f, lr_b=args.lr_b, lr_actor=args.lr_actor,
            batch_size=args.batch_size, discount=args.discount,
            ortho_coef=args.ortho_coef,
            f_target_tau=args.f_target_tau, b_target_tau=args.b_target_tau,
            train_goal_ratio=args.train_goal_ratio,
            bc_coeff=args.bc_coeff,
            fb_pessimism_penalty=args.fb_pessimism_penalty,
            actor_pessimism_penalty=args.actor_pessimism_penalty,
            stddev_clip=args.stddev_clip, q_loss_coef=args.q_loss_coef,
            clip_grad_norm=args.clip_grad_norm, weight_decay=args.weight_decay,
        )
        return FBAgent(obs_space, action_dim, FBAgentConfig(model=model_cfg, train=train_cfg))

    elif args.agent == "fb_flowbc":
        from agents.fb.flow_bc.model import FBFlowBCModelArchiConfig, FBFlowBCModelConfig
        from agents.fb.flow_bc.agent import (
            FBFlowBCAgent, FBFlowBCAgentConfig, FBFlowBCAgentTrainConfig,
        )

        archi = FBFlowBCModelArchiConfig(
            L_dim=args.L_dim, z_dim=args.z_dim, norm_z=args.norm_z,
            f=f_cfg, b=b_cfg, left_encoder=left_enc_cfg,
            actor=NoiseConditionedActorArchiConfig(),
            actor_vf=SimpleVectorFieldArchiConfig(),
        )
        model_cfg = FBFlowBCModelConfig(
            device=args.device, archi=archi,
            actor_encode_obs=args.actor_encode_obs, actor_std=args.actor_std,
        )
        train_cfg = FBFlowBCAgentTrainConfig(
            lr_f=args.lr_f, lr_b=args.lr_b, lr_actor=args.lr_actor,
            lr_actor_vf=args.lr_actor_vf,
            batch_size=args.batch_size, discount=args.discount,
            ortho_coef=args.ortho_coef,
            f_target_tau=args.f_target_tau, b_target_tau=args.b_target_tau,
            train_goal_ratio=args.train_goal_ratio,
            bc_coeff=args.bc_coeff,
            fb_pessimism_penalty=args.fb_pessimism_penalty,
            actor_pessimism_penalty=args.actor_pessimism_penalty,
            stddev_clip=args.stddev_clip, q_loss_coef=args.q_loss_coef,
            clip_grad_norm=args.clip_grad_norm, weight_decay=args.weight_decay,
            flow_steps=args.flow_steps,
        )
        return FBFlowBCAgent(obs_space, action_dim, FBFlowBCAgentConfig(model=model_cfg, train=train_cfg))

    else:
        raise ValueError(f"Unknown agent: {args.agent!r}")


def main():
    args = parse_args()

    # Setup
    set_seed_everywhere(args.seed)
    work_dir = Path(args.work_dir) / args.domain / f"seed_{args.seed}"
    work_dir.mkdir(parents=True, exist_ok=True)
    with (work_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=4)

    # W&B
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            config=vars(args),
            dir=str(work_dir),
        )

    # Build env once to get spaces
    print(f"Domain: {args.domain}")
    env, _ = create_ogbench_env(args.domain, wrappers=[], seed=args.seed)
    obs_space = env.observation_space
    action_dim = env.action_space.shape[0]
    env.close()
    print(f"obs_space={obs_space.shape}, action_dim={action_dim}")

    # Load offline data
    replay_buffer = load_ogbench_data(
        domain=args.domain,
        data_path=args.data_path,
        load_n_episodes=args.load_n_episodes,
        device=args.device,
    )
    print(f"Buffer: {len(replay_buffer['train'])} transitions")

    # Build agent
    agent = build_agent(args, obs_space, action_dim)
    print(f"Agent: {type(agent).__name__} on {agent.device}")

    # Build evaluator
    eval_tasks = ALL_TASKS[args.domain]
    evaluator = OGBenchEvaluator(
        domain=args.domain,
        tasks=eval_tasks,
        num_episodes=args.num_eval_episodes,
        num_inference_samples=args.num_inference_samples,
        seed=args.seed,
    )

    # Training loop
    print(f"Training for {args.num_train_steps} steps...")
    total_metrics = None
    fps_start = time.time()
    eval_checker = EveryNStepsChecker(0, args.eval_every, step_zero_should_trigger=False)
    log_checker = EveryNStepsChecker(0, args.log_every)
    ckpt_checker = EveryNStepsChecker(0, args.checkpoint_every, step_zero_should_trigger=False)

    for t in range(args.num_train_steps + 1):
        # Checkpoint
        if t > 0 and ckpt_checker.check(t):
            ckpt_checker.update_last_step(t)
            agent.save(str(work_dir / "checkpoint"))
            print(f"Checkpoint saved at step {t}")

        # Evaluate
        if eval_checker.check(t):
            eval_checker.update_last_step(t)
            agent._model.train(False)
            _, wandb_dict = evaluator.run(timestep=t, agent=agent, replay_buffer=replay_buffer)
            agent._model.train(True)
            if args.use_wandb:
                wandb.log({f"eval/{k}": v for k, v in wandb_dict.items()}, step=t)
            print(f"[step {t}] eval: { {k: round(v, 4) for k, v in wandb_dict.items()} }")

        # Train
        metrics = agent.update(replay_buffer, t)

        if total_metrics is None:
            total_metrics = {k: metrics[k].clone() for k in metrics}
        else:
            total_metrics = {k: total_metrics[k] + metrics[k] for k in metrics}

        # Log
        if log_checker.check(t):
            log_checker.update_last_step(t)
            n = max(1, args.log_every if t > 0 else 1)
            m = {k: round(float(v / n), 6) for k, v in total_metrics.items()}
            m["fps"] = round(n / (time.time() - fps_start), 1)
            m["step"] = t
            if args.use_wandb:
                wandb.log({f"train/{k}": v for k, v in m.items()}, step=t)
            print(f"[step {t}] {m}")
            total_metrics = None
            fps_start = time.time()

    # Final checkpoint
    agent.save(str(work_dir / "final"))
    print(f"Training complete. Results in {work_dir}")
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_train.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all tests PASS. Any failure here is a bug — fix before committing.

- [ ] **Step 6: Commit**

```bash
git add train.py tests/test_train.py
git commit -m "feat: train.py — argparse entry point for FB on OGBench"
```

---

## Self-Review

**Spec coverage check:**
- [x] `FBAgent` for antmaze → Task 8
- [x] `FBFlowBCAgent` for manipulation → Task 9
- [x] OGBench env factory + reward relabelers → Task 5
- [x] Offline data loader (state obs) → Task 6
- [x] DictBuffer → Task 4
- [x] Reward-inference evaluation → Task 10
- [x] `argparse` entry point → Task 11
- [x] `configs/.gitkeep` stub → Task 1
- [x] `requirements.txt` with uv pip → Task 1

**Placeholder scan:** None found.

**Type consistency check:**
- `FBAgentConfig(model=FBModelConfig, train=FBAgentTrainConfig)` — used consistently in Task 8 test and Task 11 `build_agent()`
- `FBFlowBCAgentConfig(model=FBFlowBCModelConfig, train=FBFlowBCAgentTrainConfig)` — used consistently in Task 9 test and Task 11 `build_agent()`
- `load_ogbench_data(domain, data_path, load_n_episodes, device)` → returns `{"train": DictBuffer}` — called correctly in Task 11
- `OGBenchEvaluator.run(timestep, agent, replay_buffer)` → returns `(eval_metrics, wandb_dict)` — called correctly in Task 11
