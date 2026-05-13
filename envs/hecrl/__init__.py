"""HECRL ManipObj env adapter.

The factored-state variant (`ob_type='states_clean'`) gives a 6+5*num_cubes
observation vector: agent (effector xyz_scaled, cos/sin yaw, gripper*3)
followed by per-cube (xyz_scaled, cos/sin yaw).

`create_hecrl_env(domain, obs_type, seed)` is the entry point used by
train.py via the benchmark dispatcher.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import register


register(
    id="manipobj-v0",
    entry_point="envs.hecrl.manip_obj_env:ManipObjEnv",
    max_episode_steps=200,        # HECRL's collect_manipobj default
)


# Short canonical names -> ManipObjEnv kwargs.
# Adding a new task (e.g. stack-cube) is a one-line entry here.
#
# NOTE: HECRL's pre-collected HuggingFace dataset "ppp-cube-noisy-v0-mv.npz"
# was collected with num_cubes=3 (state_observations dim 21 = 6 + 5*3).
# Match that here so the env's factored state and qpos slicing align with the
# buffer.
DOMAIN_KWARGS = {
    "ppp-cube": dict(num_cubes=3, mode="task", reward_mode="step"),
}


# Per-domain task list (one canonical task per domain in HECRL — kept as a
# dict for symmetry with envs.ogbench.ALL_TASKS).
ALL_TASKS = {
    "ppp-cube": ["ppp-cube"],
}


_OB_TYPE_MAP = {
    "state": "states_clean",        # factored representation; the load-bearing default
    "state_full": "states",         # full 19+9N dims; not used in first cut
    "pixels": "pixels",             # out of scope
}


def create_hecrl_env(domain, obs_type="state", seed=None, **_unused):
    """Build a ManipObj env for the given short-name domain.

    Parameters
    ----------
    domain     : str -- one of DOMAIN_KWARGS keys ("ppp-cube")
    obs_type   : "state" | "state_full" | "pixels"
    seed       : int | None
    """
    if domain not in DOMAIN_KWARGS:
        raise ValueError(
            f"Unknown HECRL domain: {domain!r}. Known: {sorted(DOMAIN_KWARGS)}"
        )
    if obs_type not in _OB_TYPE_MAP:
        raise ValueError(
            f"Unknown obs_type: {obs_type!r}. Known: {sorted(_OB_TYPE_MAP)}"
        )

    kwargs = dict(DOMAIN_KWARGS[domain])
    kwargs["ob_type"] = _OB_TYPE_MAP[obs_type]

    env = gym.make("manipobj-v0", **kwargs)
    env.reset(seed=seed)
    return env, {}
