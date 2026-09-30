"""Evaluation loop, ported from OGBench upstream impls/utils/evaluation.py.

The agent's sample_actions absorbs the `goals` kwarg (FB conditions on cond_eval, an
inferred task latent, not a per-episode goal)."""
from collections import defaultdict

import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Split the RNG key before each call to the wrapped function."""
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)
    return wrapped


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(agent, env, task_id=None, config=None, num_eval_episodes=50,
             eval_temperature=0, eval_gaussian=None, any_step_success=False):
    """Evaluate the agent in the environment. Returns (stats, trajs).

    `any_step_success` scores an episode as a success if ANY step reaches the goal, not only
    the final step (the reward-inference eval's accounting). With OGBench's default
    `terminate_at_goal=True` the two usually coincide.
    """
    config = config or {}
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2 ** 32)))
    trajs = []
    stats = defaultdict(list)
    for _ in trange(num_eval_episodes):
        traj = defaultdict(list)
        observation, info = env.reset(options=dict(task_id=task_id, render_goal=False))
        goal = info.get('goal')
        done = False
        while not done:
            action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
            action = np.array(action)
            if not config.get('discrete'):
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            add_to(traj, dict(observation=observation, next_observation=next_observation,
                              action=action, reward=reward, done=done, info=info))
            observation = next_observation
        ep_info = info
        if any_step_success:
            ep_info = dict(info)
            ep_info['success'] = any(bool(s.get('success', False)) for s in traj['info'])
        add_to(stats, flatten(ep_info))
        trajs.append(traj)
    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajs
