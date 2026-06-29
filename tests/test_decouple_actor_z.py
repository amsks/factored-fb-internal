"""decouple_actor_z: the actor trains on B(on-trajectory future), the critic keeps
the broad sample_mixed_z. We verify by capturing the z passed to update_fb vs
update_actor on one update step and asserting they differ as designed."""
import numpy as np
import torch
from gymnasium.spaces import Box

from agents.fb.agent import FBAgent


def _agent(**kw):
    obs_space = Box(low=-1, high=1, shape=(12,), dtype=np.float32)
    return FBAgent(obs_space=obs_space, action_dim=4, batch_size=8,
                   z_dim=16, L_dim=16, device="cpu", **kw)


def _batch(with_future=True):
    B = 8
    def step(): return {
        "observation": torch.randn(B, 12),
        "action": torch.randn(B, 4).clamp(-1, 1),
        "next": {"observation": torch.randn(B, 12), "terminated": torch.zeros(B, 1),
                 "action": torch.randn(B, 4)},
    }
    b = step()
    b["next"] = b["next"]
    if with_future:
        b["future"] = {"observation": torch.randn(B, 12)}
    return b


def test_decoupled_actor_gets_traj_z_not_critic_z(monkeypatch):
    ag = _agent(decouple_actor_z=True)
    seen = {}
    orig_fb, orig_actor = ag.update_fb, ag.update_actor
    monkeypatch.setattr(ag, "update_fb", lambda **k: (seen.__setitem__("critic_z", k["z"].detach().clone()), orig_fb(**k))[1])
    monkeypatch.setattr(ag, "update_actor", lambda **k: (seen.__setitem__("actor_z", k["z"].detach().clone()), orig_actor(**k))[1])
    ag.update(_batch(with_future=True), step=0)
    assert "critic_z" in seen and "actor_z" in seen
    # decoupled => the actor z is NOT the critic z
    assert not torch.allclose(seen["critic_z"], seen["actor_z"])


def test_default_shared_actor_z_is_critic_z(monkeypatch):
    ag = _agent(decouple_actor_z=False)
    seen = {}
    orig_fb, orig_actor = ag.update_fb, ag.update_actor
    monkeypatch.setattr(ag, "update_fb", lambda **k: (seen.__setitem__("critic_z", k["z"].detach().clone()), orig_fb(**k))[1])
    monkeypatch.setattr(ag, "update_actor", lambda **k: (seen.__setitem__("actor_z", k["z"].detach().clone()), orig_actor(**k))[1])
    ag.update(_batch(with_future=True), step=0)
    assert torch.allclose(seen["critic_z"], seen["actor_z"])   # shared (byte-identical path)
