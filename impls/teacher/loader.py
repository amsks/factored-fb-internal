"""impls/teacher/loader.py — load a FROZEN pretrained BC teacher and use it to relabel a
dataset with `teacher_actions` (the teacher's action at every observation).

This is how the BC part is DETACHED from the main algorithm: the teacher runs once, offline,
over the dataset's observations; training then regresses the actor toward `teacher_actions`
instead of the raw data action (or co-training its own BC head). The teacher never enters the
training loop, so it works for any actor and any critic.

    teacher = load_teacher(resolve_ckpt(root, "dp", 128, "mw-push", seed=0))
    data["teacher_actions"] = teacher.relabel(data["observations"])
"""
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from impls.teacher.nets import MLP, FlowVF


def resolve_ckpt(root, policy, demos, task, seed=0):
    """<root>/{policy}{demos}__{task}__s{seed}/policy.pkl (matches campaign/pretrain_teacher)."""
    return os.path.join(root, f"{policy}{demos}__{task}__s{seed}", "policy.pkl")


class FrozenTeacher:
    """A loaded, frozen BC policy. `.relabel(obs)` -> teacher action per observation."""

    def __init__(self, ckpt):
        self.kind = ckpt["policy"]                       # 'dp' | 'l2bc'
        self.params = ckpt["params"]
        self.adim = int(ckpt["adim"])
        self.flow_steps = int(ckpt.get("flow_steps", 10))
        hidden = tuple(ckpt["hidden"])
        if self.kind == "l2bc":
            net = MLP(hidden, self.adim)

            @jax.jit
            def _act(obs):
                return jnp.clip(net.apply(self.params, obs), -1.0, 1.0)
        else:                                            # dp / flow, deterministic (fixed noise 128)
            net = FlowVF(hidden, self.adim)
            fs = self.flow_steps

            @jax.jit
            def _act(obs):
                a0 = jax.random.normal(jax.random.PRNGKey(128), (obs.shape[0], self.adim))

                def body(i, a):
                    t = jnp.full((obs.shape[0], 1), i / fs)
                    return a + net.apply(self.params, obs, a, t) / fs

                return jnp.clip(jax.lax.fori_loop(0, fs, body, a0), -1.0, 1.0)
        self._act = _act

    def relabel(self, observations, chunk=100_000):
        """Teacher action at each observation (chunked so a big dataset doesn't OOM)."""
        obs = np.asarray(observations, np.float32)
        out = np.empty((obs.shape[0], self.adim), np.float32)
        for i in range(0, len(obs), chunk):
            out[i:i + chunk] = np.asarray(self._act(jnp.asarray(obs[i:i + chunk])))
        return out


def load_teacher(ckpt_path):
    with open(ckpt_path, "rb") as f:
        return FrozenTeacher(pickle.load(f))
