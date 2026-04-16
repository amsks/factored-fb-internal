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
