# Pixel-FB + GCIQL-pixel Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore faithful pixel-based FB support (DrQ encoder + DrQ augmentation, ported verbatim from td_jepa `metamotivo`) for `visual-cube-single-play-v0`, plus a GCIQL-pixel runner script. Code + unit tests only — no training runs.

**Architecture:** Port `DrQEncoder`/`Augmentator` + their config classes into `nn_models.py`; the FB model/agent already build and call encoder/augmentator/normalizer slots so no agent logic changes. Add an `obs_type`-gated pixel path to the offline data loader (CHW, uint8, episode-relative `timestep`) and sample-time frame-stacking to `DictBuffer` (faithful to td_jepa's offset/clamp math, behaviorally equivalent for offline data). Wire pixel config selection in `train.py make_agent` and `create_ogbench_env`, add a `visual_cube_single` domain config mirroring td_jepa's `launch_fb_ogbench.py`, and add a GCIQL-pixel runner reusing the existing `cube_single_visual.yaml`.

**Tech Stack:** Python, PyTorch, Hydra/OmegaConf, pydantic (`BaseConfig`), gymnasium, pytest, numpy.

**Spec:** `docs/superpowers/specs/2026-05-19-pixel-fb-gciql-integration-design.md`

**Source of truth for ports:** `/home/adityamohan/git/baselines/td_jepa/metamotivo/nn_models.py:329-409`, `metamotivo/buffers/parallel.py:73-89`, `scripts/train/pixel/launch_fb_ogbench.py`.

**Plan correction (Task 4):** The original Task 4 wrongly assumed "no agent logic change" because `FBModel` accepts `rgb_encoder_cfg`/`augmentator_cfg`. In fact `FBAgent.__init__` (the layer between `make_agent` and the model) has no `**kwargs` and did not declare these params, so passing them broke *all* runs incl. the state path. Task 4 therefore also threads `rgb_encoder_cfg`/`augmentator_cfg` through `agents/fb/agent.py::FBAgent.__init__` → `_make_model(...)` (mirroring `obs_normalizer_cfg`); `FBFlowBCAgent` already forwards via `**fb_kwargs`. An agent-construction regression test (state + pixel-FlowBC) was added to `tests/test_pixel_wiring.py` to catch this class of break.

**Conventions (locked, documented deviation per spec):**
- Offline pixel observations are stored **CHW** per frame: `(N, C, H, W)` `uint8`, matching `DrQEncoder`'s `Conv2d` input and `PixelWrapper`'s live-env CHW output.
- Frame stacking concatenates along the **channel axis**, order **oldest→newest**, clamped at the episode start via a stored episode-relative `timestep` (replicates `np.maximum(-i, -timesteps)` from td_jepa `parallel.py`).
- State path is unchanged: every pixel branch is gated on `obs_type == "pixels"` / config presence.
- Commit messages: **do not** add a Claude `Co-Authored-By` trailer (repo convention).

---

### Task 1: Port DrQEncoder + Augmentator into `nn_models.py`

**Files:**
- Modify: `nn_models.py` (add `import gymnasium`; append four classes after `IdentityNNConfig`, currently ending at line 463)
- Test: `tests/test_drq_encoder.py`, `tests/test_augmentator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_drq_encoder.py`:

```python
import gymnasium
import numpy as np
import torch

from nn_models import DrQEncoder, DrQEncoderArchiConfig


def _obs_space(c=9, h=64, w=64):
    return gymnasium.spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)


def test_drq_encoder_forward_shape_with_feature_dim():
    enc = DrQEncoderArchiConfig(feature_dim=256).build(_obs_space())
    out = enc(torch.zeros(4, 9, 64, 64))
    assert out.shape == (4, 256)
    assert enc.output_space.shape == (256,)


def test_drq_encoder_gradients_flow():
    enc = DrQEncoderArchiConfig(feature_dim=256).build(_obs_space())
    x = torch.rand(2, 9, 64, 64, requires_grad=True)
    enc(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_drq_encoder_rejects_non_3d_obs_space():
    bad = gymnasium.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
    try:
        DrQEncoderArchiConfig(feature_dim=32).build(bad)
        assert False, "expected AssertionError for non-3D obs_space"
    except AssertionError as e:
        assert "3D shape" in str(e)


def test_drq_encoder_name_discriminator():
    assert DrQEncoderArchiConfig().name == "drq"
```

Create `tests/test_augmentator.py`:

```python
import gymnasium
import numpy as np
import torch

from nn_models import Augmentator, AugmentatorArchiConfig


def _obs_space(c=9, h=64, w=64):
    return gymnasium.spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)


def test_augmentator_preserves_shape_dtype_device():
    aug = AugmentatorArchiConfig(pad=2).build(_obs_space())
    x = torch.rand(8, 9, 64, 64)
    out = aug(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device
    assert torch.isfinite(out).all()


def test_augmentator_actually_shifts():
    torch.manual_seed(0)
    aug = AugmentatorArchiConfig(pad=4).build(_obs_space())
    x = torch.rand(16, 9, 64, 64)
    out = aug(x)
    assert not torch.allclose(out, x), "augmentation should alter at least some samples"


def test_augmentator_rejects_non_square():
    aug = AugmentatorArchiConfig(pad=2).build(_obs_space())
    try:
        aug(torch.rand(2, 9, 64, 48))
        assert False, "expected AssertionError for non-square image"
    except AssertionError as e:
        assert "square" in str(e)


def test_augmentator_name_discriminator():
    assert AugmentatorArchiConfig().name == "random_shifts"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_drq_encoder.py tests/test_augmentator.py -q`
Expected: FAIL with `ImportError: cannot import name 'DrQEncoder' from 'nn_models'`

- [ ] **Step 3: Add `import gymnasium` to `nn_models.py`**

In `nn_models.py`, change the import block (lines 15-20) from:

```python
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn
from torch.distributions.utils import _standard_normal
```

to (add the `gymnasium` import):

```python
import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn
from torch.distributions.utils import _standard_normal
```

- [ ] **Step 4: Append the four classes verbatim to `nn_models.py`**

Append at end of file (after `IdentityNNConfig`, line 463), copied verbatim from `td_jepa/metamotivo/nn_models.py:329-409`:

```python


##########################
# Visual modules (ported verbatim from td_jepa metamotivo/nn_models.py:329-409)
##########################


class DrQEncoderArchiConfig(BaseConfig):
    name: tp.Literal["drq"] = "drq"
    feature_dim: int | None = None  # if not None, linearly project the output to feature_dim

    def build(self, obs_space):
        return DrQEncoder(obs_space, self)


class DrQEncoder(nn.Module):
    """RGB encoder from the DrQ-v2 paper"""

    def __init__(self, obs_space, cfg: DrQEncoderArchiConfig) -> None:
        super().__init__()
        self.cfg = cfg

        assert len(obs_space.shape) == 3, "obs_space must have a 3D shape (image)"

        # courtesy of https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
        self.trunk = nn.Sequential(
            nn.Conv2d(obs_space.shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            self.repr_dim = np.prod(self.trunk(torch.zeros(1, *obs_space.shape)).shape)

        if self.cfg.feature_dim is not None:
            self.proj = nn.Sequential(nn.Linear(self.repr_dim, self.cfg.feature_dim), nn.LayerNorm(self.cfg.feature_dim), nn.Tanh())
            self.repr_dim = self.cfg.feature_dim
        else:
            self.proj = nn.Identity()
            print(
                "WARNING: using a DrQ encoder with feature_dim=None. This yields very large feature vectors that are fed as input to other networks"
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.proj(self.trunk(obs))

    @property
    def output_space(self):
        return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.repr_dim,), dtype=np.float32)


class AugmentatorArchiConfig(BaseConfig):
    name: tp.Literal["random_shifts"] = "random_shifts"
    pad: int = 4

    def build(self, obs_space):
        return Augmentator(obs_space, self)


class Augmentator(nn.Module):
    """Image augmentations from DrQ-v2"""

    def __init__(self, obs_space, cfg: AugmentatorArchiConfig) -> None:
        super().__init__()
        self.cfg = cfg

        assert len(obs_space.shape) == 3, "obs_space must have a 3D shape (image)"

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        n, _, h, w = obs.size()
        assert h == w, "Augmentator only supports square images"
        padding = tuple([self.cfg.pad] * 4)
        obs = F.pad(obs, padding, "replicate")
        eps = 1.0 / (h + 2 * self.cfg.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.cfg.pad, device=obs.device, dtype=obs.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.cfg.pad + 1, size=(n, 1, 1, 2), device=obs.device, dtype=obs.dtype)
        shift *= 2.0 / (h + 2 * self.cfg.pad)
        grid = base_grid + shift
        return F.grid_sample(obs, grid, padding_mode="zeros", align_corners=False)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_drq_encoder.py tests/test_augmentator.py -q`
Expected: PASS (8 passed)

- [ ] **Step 6: Faithfulness diff (must be empty)**

Run:
```bash
diff <(sed -n '329,409p' /home/adityamohan/git/baselines/td_jepa/metamotivo/nn_models.py) \
     <(awk '/# Visual modules \(ported verbatim/{flag=1} flag' nn_models.py | sed -n '4,84p')
```
Expected: empty diff for the class bodies (header comment lines differ — only the class definitions must match). If class bodies differ, fix to match td_jepa exactly.

- [ ] **Step 7: Commit**

```bash
git add nn_models.py tests/test_drq_encoder.py tests/test_augmentator.py
git commit -m "feat(pixel-fb): port DrQEncoder + Augmentator from td_jepa"
```

---

### Task 2: Pixel offline data loading in `data/ogbench.py`

Adds an `obs_type`-gated pixel branch: reads the `pixels` key, stores per-frame **CHW uint8**, and emits an episode-relative `timestep` array for frame-stack clamping. State path is byte-for-byte unchanged.

**Files:**
- Modify: `data/ogbench.py` (`load_transitions` signature + body; `load_ogbench_dataset` signature + call)
- Test: `tests/test_pixel_data_loading.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pixel_data_loading.py`:

```python
import numpy as np

from data.ogbench import load_transitions


def _write_state_ep(path, T=6, obs_dim=4, act_dim=2):
    np.savez(
        path,
        observation=np.random.rand(T, obs_dim).astype(np.float32),
        action=np.random.rand(T, act_dim).astype(np.float32),
        physics=np.random.rand(T, 3).astype(np.float32),
        discount=np.ones(T, dtype=np.float32),
    )


def _write_pixel_ep(path, T=6, H=8, W=8, C=3, act_dim=2):
    np.savez(
        path,
        pixels=(np.random.rand(T, H, W, C) * 255).astype(np.uint8),
        action=np.random.rand(T, act_dim).astype(np.float32),
        physics=np.random.rand(T, 3).astype(np.float32),
        discount=np.ones(T, dtype=np.float32),
    )


def test_state_path_unchanged(tmp_path):
    f = tmp_path / "ep0.npz"
    _write_state_ep(f, T=6, obs_dim=4)
    s = load_transitions([f])  # default obs_type="state"
    assert s["observation"].dtype == np.float32
    assert s["observation"].shape == (5, 4)
    assert "timestep" not in s


def test_pixel_path_chw_uint8_with_timestep(tmp_path):
    f0, f1 = tmp_path / "ep0.npz", tmp_path / "ep1.npz"
    _write_pixel_ep(f0, T=6, H=8, W=8, C=3)
    _write_pixel_ep(f1, T=4, H=8, W=8, C=3)
    s = load_transitions([f0, f1], obs_type="pixels")
    # 5 + 3 transitions, CHW per frame, uint8 preserved
    assert s["observation"].shape == (8, 3, 8, 8)
    assert s["observation"].dtype == np.uint8
    assert s["next"]["observation"].shape == (8, 3, 8, 8)
    # episode-relative timestep resets at each episode boundary
    np.testing.assert_array_equal(
        s["timestep"], np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int32)
    )


def test_pixel_missing_key_raises(tmp_path):
    f = tmp_path / "ep0.npz"
    _write_state_ep(f)  # has "observation", not "pixels"
    try:
        load_transitions([f], obs_type="pixels")
        assert False, "expected KeyError-style failure for missing pixels key"
    except (KeyError, ValueError) as e:
        assert "pixels" in str(e)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pixel_data_loading.py -q`
Expected: FAIL with `TypeError: load_transitions() got an unexpected keyword argument 'obs_type'`

- [ ] **Step 3: Implement the pixel branch**

In `data/ogbench.py`, replace the `load_transitions` function (lines 27-78) with:

```python
def load_transitions(
    npz_files: List[Union[str, Path]], obs_type: str = "state"
) -> Dict:
    """Load episode .npz files into a flat transition dict.

    State (obs_type="state") path is unchanged. Pixel (obs_type="pixels")
    path reads the "pixels" key, stores per-frame CHW uint8, and emits an
    episode-relative "timestep" for frame-stack clamping.
    """
    obs_list, act_list, next_obs_list, term_list, next_phys_list = [], [], [], [], []
    ts_list = []
    has_physics = True

    for f in npz_files:
        ep = np.load(f)
        act = ep["action"].astype(np.float32)

        if obs_type == "pixels":
            if "pixels" not in ep:
                raise KeyError(
                    f"obs_type='pixels' but episode {f} has no 'pixels' key "
                    f"(keys: {list(ep.keys())})"
                )
            # [T, H, W, C] uint8 -> [T, C, H, W] uint8 (CHW per frame)
            pix = np.moveaxis(ep["pixels"], -1, 1)
            obs_full = np.ascontiguousarray(pix)
            cast = None  # keep uint8
        elif obs_type == "state":
            obs_full = ep["observation"].astype(np.float32)
            cast = np.float32
        else:
            raise ValueError(f"Unsupported obs_type {obs_type!r}")

        # EXORL-style shift (see module docstring / td_jepa base.py): action[t]
        # is the action that LED to obs[t], action[0]=0, so undo via act[1:].
        obs_list.append(obs_full[:-1])
        act_list.append(act[1:])
        next_obs_list.append(obs_full[1:])
        n_trans = obs_full.shape[0] - 1
        term_list.append(np.zeros((n_trans, 1), dtype=bool))
        ts_list.append(np.arange(n_trans, dtype=np.int32))

        if has_physics and "physics" in ep:
            phys = ep["physics"].astype(np.float32)
            next_phys_list.append(phys[1:])
        else:
            has_physics = False

    storage: Dict = {
        "observation": np.concatenate(obs_list, axis=0),
        "action": np.concatenate(act_list, axis=0),
        "next": {
            "observation": np.concatenate(next_obs_list, axis=0),
            "terminated": np.concatenate(term_list, axis=0),
        },
    }
    if obs_type == "pixels":
        storage["timestep"] = np.concatenate(ts_list, axis=0)
    if has_physics and next_phys_list:
        storage["next"]["physics"] = np.concatenate(next_phys_list, axis=0)

    return storage
```

Then update `load_ogbench_dataset` (lines 81-118): change its signature to add `obs_type` and `frame_stack`, and pass them through. Replace lines 81-118 with:

```python
def load_ogbench_dataset(
    domain: str,
    data_path: Union[str, Path] = "datasets",
    load_n_episodes: int = 1000,
    device: str = "cpu",
    n_transitions: int | None = None,
    obs_type: str = "state",
    frame_stack: int = 1,
) -> DictBuffer:
    """Load offline OGBench data into a DictBuffer.

    Parameters
    ----------
    domain : e.g. "antmaze-medium-navigate-v0"
    data_path : root dir; episodes are at <data_path>/<domain>/buffer/*.npz
    load_n_episodes : max number of episodes to load
    device : device for sampled tensors
    n_transitions : if set, truncate the buffer to this many transitions
    obs_type : "state" (default) or "pixels"
    frame_stack : number of frames to stack on sample (pixels only; 1 = off)
    """
    buf_dir = Path(data_path) / domain / "buffer"
    files = sorted(buf_dir.glob("*.npz"))[:load_n_episodes]
    if not files:
        raise FileNotFoundError(
            f"No .npz episode files found at {buf_dir}. "
            "Download the OGBench dataset or check your data_path."
        )

    storage = load_transitions(files, obs_type=obs_type)

    if n_transitions is not None:
        for k, v in list(storage.items()):
            if isinstance(v, dict):
                storage[k] = {kk: vv[:n_transitions] for kk, vv in v.items()}
            else:
                storage[k] = v[:n_transitions]

    n = len(storage["observation"])
    buffer = DictBuffer(
        capacity=n, device=device, frame_stack=frame_stack, obs_type=obs_type
    )
    buffer.extend(storage)
    return buffer
```

> Note: `DictBuffer` gains `frame_stack`/`obs_type` kwargs in Task 3. Task 2's test does not construct `DictBuffer`, so it passes independently; the `load_ogbench_dataset` change is exercised in Task 5.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pixel_data_loading.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Run regression — state data tests still green**

Run: `python -m pytest tests/test_pixel_data_loading.py::test_state_path_unchanged -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add data/ogbench.py tests/test_pixel_data_loading.py
git commit -m "feat(pixel-fb): obs_type-gated pixel path in OGBench data loader"
```

---

### Task 3: Sample-time frame-stacking in `DictBuffer`

Adds `frame_stack`/`obs_type` to `DictBuffer`. When `obs_type=="pixels"` and `frame_stack>1`, `sample()` stacks `frame_stack` frames along the channel axis, clamped at the episode start using the stored `timestep` (replicates td_jepa `parallel.py:73-89` `np.maximum(-i, -timesteps)`). `s'` drops the oldest frame of `s` and appends the stored next frame. State path / `frame_stack==1` is unchanged.

**Files:**
- Modify: `buffers/transition.py` (`DictBuffer.__init__`, `DictBuffer.sample`)
- Test: `tests/test_frame_stack_buffer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_frame_stack_buffer.py`:

```python
import numpy as np
import torch

from buffers.transition import DictBuffer


def _pixel_storage(eps):
    """eps: list of episode lengths (#transitions). Frame value = global row id
    in channel 0 so we can assert exact stacking/clamping."""
    obs, nobs, ts = [], [], []
    rid = 0
    for L in eps:
        for t in range(L):
            frame = np.full((1, 2, 2), rid, dtype=np.uint8)
            nframe = np.full((1, 2, 2), rid + 100, dtype=np.uint8)
            obs.append(frame)
            nobs.append(nframe)
            ts.append(t)
            rid += 1
    return {
        "observation": np.stack(obs).astype(np.uint8),       # (N,1,2,2)
        "action": np.zeros((len(ts), 2), np.float32),
        "next": {
            "observation": np.stack(nobs).astype(np.uint8),
            "terminated": np.zeros((len(ts), 1), bool),
        },
        "timestep": np.array(ts, dtype=np.int32),
    }


def test_frame_stack_shapes_and_channel_concat():
    data = _pixel_storage([5, 3])
    n = len(data["timestep"])
    buf = DictBuffer(capacity=n, device="cpu", frame_stack=3, obs_type="pixels")
    buf.extend(data)
    b = buf.sample(16)
    assert b["observation"].shape == (16, 3, 2, 2)       # C*frame_stack
    assert b["next"]["observation"].shape == (16, 3, 2, 2)
    assert b["action"].shape == (16, 2)                  # non-pixel untouched


def test_frame_stack_clamps_at_episode_start():
    # episode 0: rows 0..4 (ts 0..4); episode 1: rows 5..7 (ts 0..2)
    data = _pixel_storage([5, 3])
    buf = DictBuffer(capacity=8, device="cpu", frame_stack=3, obs_type="pixels")
    buf.extend(data)

    # Force deterministic indices by monkeypatching torch.randint via seed:
    torch.manual_seed(0)
    # Instead of relying on RNG, directly exercise the stacking helper:
    # row 0 (episode start, ts=0): all 3 frames clamp to row 0
    stacked = buf._stack_obs("observation", np.array([0]))
    np.testing.assert_array_equal(stacked[0, :, 0, 0].numpy(), [0, 0, 0])
    # row 6 (episode 1, ts=1): frames clamp to rows [5, 5, 6]
    stacked = buf._stack_obs("observation", np.array([6]))
    np.testing.assert_array_equal(stacked[0, :, 0, 0].numpy(), [5, 5, 6])
    # row 4 (episode 0, ts=4): frames are rows [2, 3, 4] (oldest->newest)
    stacked = buf._stack_obs("observation", np.array([4]))
    np.testing.assert_array_equal(stacked[0, :, 0, 0].numpy(), [2, 3, 4])


def test_next_obs_drops_oldest_appends_next_frame():
    data = _pixel_storage([5, 3])
    buf = DictBuffer(capacity=8, device="cpu", frame_stack=3, obs_type="pixels")
    buf.extend(data)
    # row 4: s = [2,3,4]; s' = drop oldest -> [3,4] + next_frame(=104)
    nxt = buf._stack_next_obs(np.array([4]))
    np.testing.assert_array_equal(nxt[0, :, 0, 0].numpy(), [3, 4, 104])


def test_frame_stack_1_is_passthrough():
    data = _pixel_storage([4])
    buf = DictBuffer(capacity=4, device="cpu", frame_stack=1, obs_type="pixels")
    buf.extend(data)
    b = buf.sample(4)
    assert b["observation"].shape == (4, 1, 2, 2)  # no channel multiplication


def test_state_buffer_unaffected():
    n = 10
    data = {
        "observation": np.random.rand(n, 4).astype(np.float32),
        "action": np.random.rand(n, 2).astype(np.float32),
        "next": {
            "observation": np.random.rand(n, 4).astype(np.float32),
            "terminated": np.zeros((n, 1), bool),
        },
    }
    buf = DictBuffer(capacity=n, device="cpu")  # defaults: frame_stack=1, state
    buf.extend(data)
    b = buf.sample(5)
    assert b["observation"].shape == (5, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_frame_stack_buffer.py -q`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'frame_stack'`

- [ ] **Step 3: Implement frame-stacking in `DictBuffer`**

In `buffers/transition.py`, replace `__init__` (lines 72-77) with:

```python
    def __init__(
        self,
        capacity: int,
        device: str = "cpu",
        frame_stack: int = 1,
        obs_type: str = "state",
    ):
        self.capacity = capacity
        self.device = device
        self.frame_stack = frame_stack
        self.obs_type = obs_type
        self._stack = obs_type == "pixels" and frame_stack > 1
        self._size: int = 0
        self._ptr: int = 0
        self._storage: Dict[str, np.ndarray] = {}  # flat storage
```

Then replace `sample` (lines 118-131) with:

```python
    def _stack_obs(self, key: str, idx: np.ndarray):
        """Stack `frame_stack` frames of `key` ending at idx, clamped at the
        episode start via stored `timestep` (td_jepa parallel.py:73-89 math).
        Returns a CHW-concatenated tensor (oldest->newest on channel axis)."""
        ts = self._storage["timestep"][idx]  # episode-relative index
        frames = []
        for k in range(self.frame_stack - 1, -1, -1):  # oldest -> newest
            back = np.minimum(k, ts)                    # never cross ep start
            frames.append(self._storage[key][idx - back])
        arr = np.concatenate(frames, axis=1)            # channel axis
        return torch.as_tensor(arr).to(self.device)

    def _stack_next_obs(self, idx: np.ndarray):
        """s' = drop oldest frame of s, append the stored next frame."""
        ts = self._storage["timestep"][idx]
        frames = []
        for k in range(self.frame_stack - 2, -1, -1):   # s minus its oldest
            back = np.minimum(k, ts)
            frames.append(self._storage["observation"][idx - back])
        frames.append(self._storage["next/observation"][idx])
        arr = np.concatenate(frames, axis=1)
        return torch.as_tensor(arr).to(self.device)

    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample *batch_size* transitions uniformly. Returns nested dict of tensors.

        Uses torch.randint (not np.random) so RNG consumption matches td_jepa's
        DictBuffer.sample exactly.
        """
        assert self._size > 0, "Buffer is empty"
        idx = torch.randint(0, self._size, (batch_size,)).numpy()

        if not self._stack:
            flat_tensors = {
                k: torch.as_tensor(v[idx]).to(self.device)
                for k, v in self._storage.items()
            }
            out = _unflatten(flat_tensors)
            out.pop("timestep", None)  # internal-only; not part of a batch
            return out

        out_flat = {}
        for k, v in self._storage.items():
            if k in ("observation", "next/observation", "timestep"):
                continue
            out_flat[k] = torch.as_tensor(v[idx]).to(self.device)
        out_flat["observation"] = self._stack_obs("observation", idx)
        out_flat["next/observation"] = self._stack_next_obs(idx)
        return _unflatten(out_flat)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_frame_stack_buffer.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Run regression — existing buffer/data tests still green**

Run: `python -m pytest tests/test_frame_stack_buffer.py::test_state_buffer_unaffected tests/test_pixel_data_loading.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add buffers/transition.py tests/test_frame_stack_buffer.py
git commit -m "feat(pixel-fb): sample-time frame-stacking in DictBuffer"
```

---

### Task 4: Wire pixel config selection in `train.py` + `create_ogbench_env`

`make_agent` currently hardcodes `obs_normalizer_cfg=IdentityNormalizerConfig()` and never sets `rgb_encoder_cfg`/`augmentator_cfg`. This task makes those configurable from Hydra (defaulting to the state/Identity path) and threads `frame_stack` into env creation and dataset loading.

**Files:**
- Modify: `train.py` (imports; `make_agent`; the `create_ogbench_env` + `load_ogbench_dataset` calls in `main`)
- Modify: `envs/ogbench.py` (`create_ogbench_env`: apply `PixelWrapper` for pixels)
- Test: `tests/test_pixel_wiring.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pixel_wiring.py`:

```python
from omegaconf import OmegaConf

from train import build_pixel_cfgs  # helper added in this task


def test_state_cfg_returns_identity_defaults():
    cfg = OmegaConf.create({})  # no pixel blocks
    norm, rgb, aug = build_pixel_cfgs(cfg)
    assert type(norm).__name__ == "IdentityNormalizerConfig"
    assert type(rgb).__name__ == "IdentityNNConfig"
    assert type(aug).__name__ == "IdentityNNConfig"


def test_pixel_cfg_selects_drq_rgbnorm_augmentator():
    cfg = OmegaConf.create(
        {
            "obs_normalizer": {"name": "RGBNormalizerConfig"},
            "rgb_encoder": {"name": "drq", "feature_dim": 256},
            "augmentator": {"name": "random_shifts", "pad": 2},
        }
    )
    norm, rgb, aug = build_pixel_cfgs(cfg)
    assert type(norm).__name__ == "RGBNorm" or type(norm).__name__ == "RGBNormalizerConfig"
    assert type(rgb).__name__ == "DrQEncoderArchiConfig"
    assert rgb.feature_dim == 256
    assert type(aug).__name__ == "AugmentatorArchiConfig"
    assert aug.pad == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pixel_wiring.py -q`
Expected: FAIL with `ImportError: cannot import name 'build_pixel_cfgs' from 'train'`

- [ ] **Step 3: Add `build_pixel_cfgs` and use it in `make_agent`**

In `train.py`, extend the `nn_models` import (lines 38-44) to also import the pixel configs:

```python
from nn_models import (
    AugmentatorArchiConfig,
    BackwardArchiConfig,
    DrQEncoderArchiConfig,
    ForwardArchiConfig,
    IdentityNNConfig,
    NoiseConditionedActorArchiConfig,
    SimpleActorArchiConfig,
    SimpleVectorFieldArchiConfig,
)
```

Change the normalizers import (line 45) to:

```python
from normalizers import IdentityNormalizerConfig, RGBNormalizerConfig
```

Add this helper immediately above `def make_agent` (before line 63):

```python
def build_pixel_cfgs(cfg):
    """Select obs-normalizer / rgb-encoder / augmentator configs.

    Defaults (no pixel blocks in cfg) reproduce the state path exactly:
    Identity normalizer + Identity encoder + Identity augmentator.
    """
    norm_block = cfg.get("obs_normalizer", None)
    if norm_block is not None and norm_block.get("name") == "RGBNormalizerConfig":
        obs_normalizer_cfg = RGBNormalizerConfig()
    else:
        obs_normalizer_cfg = IdentityNormalizerConfig()

    rgb_block = cfg.get("rgb_encoder", None)
    if rgb_block is not None and rgb_block.get("name") == "drq":
        rgb_encoder_cfg = DrQEncoderArchiConfig(feature_dim=rgb_block.get("feature_dim"))
    else:
        rgb_encoder_cfg = IdentityNNConfig()

    aug_block = cfg.get("augmentator", None)
    if aug_block is not None and aug_block.get("name") == "random_shifts":
        augmentator_cfg = AugmentatorArchiConfig(pad=aug_block.get("pad", 4))
    else:
        augmentator_cfg = IdentityNNConfig()

    return obs_normalizer_cfg, rgb_encoder_cfg, augmentator_cfg
```

In `make_agent`, replace the `obs_normalizer_cfg=IdentityNormalizerConfig(),` line (line 91) inside the `shared = dict(...)` with:

```python
        obs_normalizer_cfg=_obs_normalizer_cfg,
        rgb_encoder_cfg=_rgb_encoder_cfg,
        augmentator_cfg=_augmentator_cfg,
```

and immediately before `shared = dict(` (line 81) insert:

```python
    _obs_normalizer_cfg, _rgb_encoder_cfg, _augmentator_cfg = build_pixel_cfgs(cfg)
```

- [ ] **Step 4: Apply `PixelWrapper` in `create_ogbench_env`**

In `envs/ogbench.py`, add the import near the top (with the other env imports):

```python
from envs.wrappers import PixelWrapper
```

Change `create_ogbench_env` signature (lines 51-58) to add `frame_stack`:

```python
def create_ogbench_env(
    task: str,
    wrappers: tp.List = None,
    seed: tp.Optional[int] = None,
    render_height: int = 64,
    render_width: int = 64,
    obs_type: str = "state",
    frame_stack: int = 1,
):
```

Replace the `match obs_type` block (lines 61-67) with:

```python
    if wrappers is None:
        wrappers = []
    match obs_type:
        case "state":
            pass
        case "pixels":
            task = "visual-" + task
            wrappers = wrappers + [lambda e: PixelWrapper(e, frame_stack)]
        case _:
            raise ValueError(f"Unsupported observation type {obs_type}")
```

- [ ] **Step 5: Thread `frame_stack` through `main()`**

In `train.py` `main()`, replace the `create_ogbench_env(...)` call (line 168) with:

```python
    env, _ = create_ogbench_env(
        cfg.domain,
        obs_type=cfg.obs_type,
        seed=cfg.seed,
        frame_stack=cfg.get("frame_stack", 1),
    )
```

and replace the `load_ogbench_dataset(...)` call (lines 174-180) with:

```python
    buffer = load_ogbench_dataset(
        domain=cfg.domain,
        data_path=cfg.data_path,
        load_n_episodes=cfg.load_n_episodes,
        device=cfg.device,
        n_transitions=cfg.n_transitions,
        obs_type=cfg.obs_type,
        frame_stack=cfg.get("frame_stack", 1),
    )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_pixel_wiring.py -q`
Expected: PASS (2 passed)

- [ ] **Step 7: Regression — full state suite still imports/builds**

Run: `python -m pytest tests/test_pixel_wiring.py tests/test_drq_encoder.py tests/test_augmentator.py tests/test_frame_stack_buffer.py tests/test_pixel_data_loading.py -q`
Expected: PASS (all)

- [ ] **Step 8: Commit**

```bash
git add train.py envs/ogbench.py tests/test_pixel_wiring.py
git commit -m "feat(pixel-fb): wire DrQ/Augmentator/RGBNorm + PixelWrapper selection"
```

---

### Task 5: `visual_cube_single` domain config + end-to-end model build test

Adds the pixel domain config mirroring `td_jepa/scripts/train/pixel/launch_fb_ogbench.py`, and an integration test that Hydra-composes it and builds the FB model with real pixel modules (no training).

**Files:**
- Create: `configs/domain/visual_cube_single.yaml`
- Test: `tests/test_visual_cube_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_visual_cube_config.py`:

```python
import gymnasium
import numpy as np
from hydra import compose, initialize

from train import build_pixel_cfgs


def test_visual_cube_config_composes_pixel_blocks():
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["domain=visual_cube_single"])
    assert cfg.obs_type == "pixels"
    assert cfg.frame_stack == 3
    assert cfg.agent == "fb_flowbc"
    assert cfg.obs_normalizer.name == "RGBNormalizerConfig"
    assert cfg.rgb_encoder.name == "drq"
    assert cfg.rgb_encoder.feature_dim == 256
    assert cfg.augmentator.name == "random_shifts"
    assert cfg.augmentator.pad == 2
    assert cfg.L_dim == 256
    assert cfg.z_dim == 50
    assert cfg.batch_size == 256


def test_visual_cube_builds_real_pixel_modules():
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["domain=visual_cube_single"])
    norm, rgb, aug = build_pixel_cfgs(cfg)
    obs_space = gymnasium.spaces.Box(
        low=0, high=255, shape=(9, 64, 64), dtype=np.uint8
    )
    rgb_mod = rgb.build(obs_space)
    aug_mod = aug.build(obs_space)
    norm_mod = norm.build(obs_space)
    assert type(rgb_mod).__name__ == "DrQEncoder"
    assert type(aug_mod).__name__ == "Augmentator"
    assert type(norm_mod).__name__ == "RGBNorm"
    assert rgb_mod.output_space.shape == (256,)


def test_state_domain_still_identity():
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["domain=cube_single"])
    norm, rgb, aug = build_pixel_cfgs(cfg)
    assert type(rgb).__name__ == "IdentityNNConfig"
    assert type(aug).__name__ == "IdentityNNConfig"
    assert type(norm).__name__ == "IdentityNormalizerConfig"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_visual_cube_config.py -q`
Expected: FAIL — Hydra `MissingConfigException` / cannot find `domain=visual_cube_single`

- [ ] **Step 3: Create the domain config**

Create `configs/domain/visual_cube_single.yaml` (values from `td_jepa/scripts/train/pixel/launch_fb_ogbench.py`):

```yaml
# @package _global_
# Pixel cube-single FB. Mirrors td_jepa/scripts/train/pixel/launch_fb_ogbench.py.
defaults:
  - /agent: fb_flowbc
  - _self_

domain: cube-single-play-v0      # create_ogbench_env prepends "visual-" for pixels
agent: fb_flowbc
obs_type: pixels
frame_stack: 3

# Pixel module selection (consumed by train.build_pixel_cfgs)
obs_normalizer:
  name: RGBNormalizerConfig
rgb_encoder:
  name: drq
  feature_dim: 256
augmentator:
  name: random_shifts
  pad: 2

# Architecture deltas vs state (launch_fb_ogbench.py BASE_CFG)
L_dim: 256
z_dim: 50
batch_size: 256
discount: 0.99
ortho_coef: 1.0
f_target_tau: 0.005
b_target_tau: 0.005
bc_coeff: 3.0
left_encoder:
  hidden_dim: 512
  hidden_layers: 0
  norm: true
backward:
  hidden_dim: 512
  hidden_layers: 4
  norm: true
forward:
  hidden_dim: 512
  hidden_layers: 2
  embedding_layers: 2
  num_parallel: 2
actor:
  name: noise_conditioned_actor
  hidden_dim: 512
  hidden_layers: 2
  embedding_layers: 2
actor_vf:
  hidden_dim: 512
  hidden_layers: 4
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_visual_cube_config.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Full pixel-FB suite green**

Run: `python -m pytest tests/test_drq_encoder.py tests/test_augmentator.py tests/test_pixel_data_loading.py tests/test_frame_stack_buffer.py tests/test_pixel_wiring.py tests/test_visual_cube_config.py -q`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add configs/domain/visual_cube_single.yaml tests/test_visual_cube_config.py
git commit -m "feat(pixel-fb): visual_cube_single domain config + e2e build test"
```

---

### Task 6: GCIQL-pixel runner script

Mirrors `scripts/run_gciql_state.sh` but targets the existing `configs/gciql/cube_single_visual.yaml`. No training run executed — verified only by an argv-parse test (same pattern as `scripts/test_run_gciql.py`).

**Files:**
- Create: `scripts/run_gciql_visual.sh`
- Test: `tests/test_run_gciql_visual.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_gciql_visual.py`:

```python
import os
import stat
from pathlib import Path

from hydra import compose, initialize

from run_gciql import build_ogbench_argv

REPO = Path(__file__).resolve().parents[1]


def test_visual_gciql_config_builds_expected_argv():
    with initialize(version_base="1.3", config_path="../configs/gciql"):
        cfg = compose(config_name="cube_single_visual")
    cfg_d = dict(cfg)
    cfg_d["output_root"] = "/tmp/out"
    argv = build_ogbench_argv(cfg_d)
    assert "--env_name=visual-cube-single-play-v0" in argv
    assert "--agent=agents/gciql.py" in argv
    assert "--agent.encoder=impala_small" in argv
    assert "--agent.p_aug=0.5" in argv
    assert "--agent.batch_size=256" in argv


def test_runner_script_exists_and_is_executable():
    p = REPO / "scripts" / "run_gciql_visual.sh"
    assert p.exists(), "scripts/run_gciql_visual.sh must exist"
    mode = p.stat().st_mode
    assert mode & stat.S_IXUSR, "script must be executable"
    text = p.read_text()
    assert "--config-name cube_single_visual" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_gciql_visual.py -q`
Expected: FAIL — `scripts/run_gciql_visual.sh must exist`

- [ ] **Step 3: Create the runner script**

Create `scripts/run_gciql_visual.sh`:

```bash
#!/usr/bin/env bash
# Full GCIQL pixel run on visual-cube-single-play-v0 using OGBench's visual
# hyperparameters (configs/gciql/cube_single_visual.yaml: 500k steps,
# encoder=impala_small, p_aug=0.5, batch_size=256). Mirrors
# scripts/run_gciql_state.sh ergonomics. Online wandb by default so GCIQL-pixel
# eval metrics overlay with FB-pixel runs (mirrored to FB's wandb key scheme by
# tools/wandb_mode_shim).
#   SEEDS="0 1 2"  WANDB_MODE=online|offline|disabled  RUN_GROUP=...  STORAGE=shm|nvme
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SEEDS="${SEEDS:-0 1 2}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_GROUP="${RUN_GROUP:-factored-fb-gciql}"
STORAGE="${STORAGE:-shm}"

echo "[gciql-visual] seeds=$SEEDS wandb=$WANDB_MODE group=$RUN_GROUP storage=$STORAGE"
echo "[gciql-visual] env=visual-cube-single-play-v0 (500k steps, impala_small, p_aug=0.5)"
for seed in $SEEDS; do
    echo "[gciql-visual] $(date +%T)  launching seed=$seed"
    python run_gciql.py --config-name cube_single_visual \
        storage="$STORAGE" seed="$seed" wandb_mode="$WANDB_MODE" run_group="$RUN_GROUP"
    echo "[gciql-visual] $(date +%T)  done seed=$seed"
done
echo "[gciql-visual] ALL DONE"
```

- [ ] **Step 4: Make it executable**

Run: `chmod +x scripts/run_gciql_visual.sh`

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_run_gciql_visual.py -q`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add scripts/run_gciql_visual.sh tests/test_run_gciql_visual.py
git commit -m "feat(gciql-pixel): runner script for visual-cube-single GCIQL"
```

---

### Task 7: Full-suite verification + faithfulness doc

**Files:**
- Create: `docs/superpowers/notes/2026-05-19-pixel-fb-faithfulness.md`

- [ ] **Step 1: Run the entire test suite**

Run: `python -m pytest tests -q`
Expected: PASS — all pre-existing tests plus the six new pixel test modules green. If any pre-existing test fails, investigate before proceeding (must not be a regression from Tasks 1-6).

- [ ] **Step 2: Re-run the faithfulness diff and capture it**

Run:
```bash
diff <(sed -n '329,409p' /home/adityamohan/git/baselines/td_jepa/metamotivo/nn_models.py) \
     <(awk '/# Visual modules \(ported verbatim/{flag=1} flag' nn_models.py | sed -n '4,84p') \
  && echo "DRQ/AUGMENTATOR: byte-identical"
```
Expected: prints `DRQ/AUGMENTATOR: byte-identical`

- [ ] **Step 3: Write the faithfulness note**

Create `docs/superpowers/notes/2026-05-19-pixel-fb-faithfulness.md`:

```markdown
# Pixel-FB Faithfulness Notes (2026-05-19)

## Byte-identical ports (from td_jepa/metamotivo/nn_models.py:329-409)
- `DrQEncoderArchiConfig`, `DrQEncoder`
- `AugmentatorArchiConfig`, `Augmentator`
Verified by `diff` against td_jepa source (see plan Task 7 Step 2).

## Intentional, documented deviations
- **Buffer:** repo `DictBuffer` ≠ td_jepa `ParallelBuffer`. We port only the
  frame-stack offset/clamp math (`np.maximum(-i, -timesteps)` ->
  `np.minimum(k, timestep)`); not the parallel prefetch machinery.
  Behaviorally equivalent for offline data: stacked observation for transition
  t is the deterministic sequence [t-(fs-1) .. t] clamped at episode start.
- **`next.observation`:** reconstructed as "drop oldest of s, append stored
  next frame", matching td_jepa's `obs[1:1+frame_stack]` slice semantics
  without storing a redundant timestep array.
- **CHW storage:** offline pixels stored `(N,C,H,W)` to match `DrQEncoder`
  Conv2d input and `PixelWrapper` live-env output (td_jepa stores HWC and
  relies on its own loader; we normalize at load time).
- **`terminated`:** kept all-False per the existing repo convention (offline GC
  data treated as non-terminating), unchanged from the state path.

## Out of scope (recommended follow-up)
- Smoke run (~50k steps) FB-pixel on visual-cube-single-play-v0.
- GCIQL-pixel validation run via scripts/run_gciql_visual.sh.
- FB-pixel vs GCIQL-pixel comparison sweep.
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/notes/2026-05-19-pixel-fb-faithfulness.md
git commit -m "docs(pixel-fb): faithfulness notes + documented deviations"
```

---

## Self-Review

**Spec coverage:**
- DrQEncoder/Augmentator verbatim port → Task 1 ✓
- RGBNormalizerConfig already present; selected via config → Task 4 (`build_pixel_cfgs`) ✓
- Pixel offline data loading (`pixels` key, uint8, CHW) → Task 2 ✓
- Frame-stacking faithful to td_jepa offsets → Task 3 ✓
- Config plumbing (`obs_type`/`frame_stack`/module selection) → Task 4 ✓
- PixelWrapper applied for pixel env → Task 4 ✓
- `visual_cube_single` domain config mirroring launch_fb_ogbench.py → Task 5 ✓
- Unit tests 1-5 from spec → Tasks 1,1,2,3,5 (drq, augmentator, pixel data, frame-stack, visual config) ✓
- GCIQL-pixel runner script, no run → Task 6 ✓
- Faithfulness diff + documented deviations → Tasks 1 (Step 6), 7 ✓
- State path regression guards → Tasks 2,3,4,5 (explicit regression steps) ✓
- No training/smoke runs → respected (only argv/build/compose tests) ✓

**Placeholder scan:** No TBD/TODO; every code step contains full code; every command has expected output.

**Type consistency:** `build_pixel_cfgs(cfg) -> (obs_normalizer_cfg, rgb_encoder_cfg, augmentator_cfg)` defined in Task 4, consumed identically in Tasks 4 & 5. `DictBuffer(capacity, device, frame_stack, obs_type)` defined in Task 3, called with those kwargs in Task 2's `load_ogbench_dataset`. `_stack_obs(key, idx)` / `_stack_next_obs(idx)` defined and tested in Task 3. `create_ogbench_env(..., frame_stack=1)` defined in Task 4, called with `frame_stack=` in Task 4 `main()`. `load_transitions(npz_files, obs_type="state")` and `load_ogbench_dataset(..., obs_type, frame_stack)` defined in Task 2, used in Task 4. Config keys (`obs_type`, `frame_stack`, `obs_normalizer.name`, `rgb_encoder.name/feature_dim`, `augmentator.name/pad`) consistent between Task 5 YAML and Task 4 reader.
