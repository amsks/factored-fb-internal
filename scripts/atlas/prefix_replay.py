"""Counterfactual branching by reset + prefix-replay (verified bit-exact).

Snapshot/restore is NOT used: the env holds an IK-target integrator
(``_target_effector_pose``) outside ``mjData`` that is not in the dataset, so only
replay from a pinned reset reconstructs the full hidden state. ALWAYS pass task_id --
a bare reset(seed=) samples a random task from global np.random (manipspace_env.py:304)
and is not reproducible.
"""
from collections import namedtuple

import numpy as np
import ogbench

PrefixState = namedtuple('PrefixState', 'seed task_id prefix')


def make_env(task_id=1):
    return ogbench.make_env_and_datasets('cube-single-play-v0', env_only=True)


def _reset(env, seed, task_id):
    return env.reset(seed=seed, options=dict(task_id=task_id, render_goal=False))


def replay_to(env, ps):
    """Reset with (seed, task_id), replay the prefix, return the current obs.
    Leaves the env AT that state, ready to step()."""
    ob, _ = _reset(env, ps.seed, ps.task_id)
    for a in ps.prefix:
        ob, *_ = env.step(np.asarray(a, np.float32))
    return np.asarray(ob, np.float32)


def branch(env, ps, first_action, continuation):
    """replay_to(ps), execute first_action, then each action in continuation.
    Returns the list of observations (one per stepped action)."""
    replay_to(env, ps)
    out = []
    o, *_ = env.step(np.asarray(first_action, np.float32))
    out.append(np.asarray(o, np.float32))
    for a in continuation:
        o, *_ = env.step(np.asarray(a, np.float32))
        out.append(np.asarray(o, np.float32))
    return out
