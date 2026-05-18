import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals import analysis as A
from train import make_agent


def _cfg(domain="cube_single"):
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"),
                               version_base="1.3"):
        return compose(config_name="train",
                       overrides=[f"domain={domain}", "device=cpu",
                                  "batch_size=16"])


def test_load_checkpoint_roundtrip(tmp_path):
    cfg = _cfg()
    obs_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
    a1 = make_agent(cfg, obs_space, action_dim=4)
    ckpt = tmp_path / "step_10.pt"
    torch.save(a1.state_dict(), ckpt)

    a2 = make_agent(cfg, obs_space, action_dim=4)
    A.load_checkpoint(a2, ckpt, map_location="cpu")
    # a parameter tensor should now match
    p1 = next(iter(a1.state_dict()["model"].values()))
    p2 = next(iter(a2.state_dict()["model"].values()))
    assert torch.allclose(p1, p2)


def test_load_checkpoint_clear_error_on_mismatch(tmp_path):
    junk = tmp_path / "junk.pt"
    torch.save({"model": {"not_a_real_key": torch.zeros(1)}}, junk)
    cfg = _cfg()
    obs_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
    agent = make_agent(cfg, obs_space, action_dim=4)
    with pytest.raises(RuntimeError, match="checkpoint does not match"):
        A.load_checkpoint(agent, junk, map_location="cpu")
