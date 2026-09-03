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


def goal_qpos_channel(env):
    """The eval goal's PRIVILEGED OBJECT-COORDINATE row, for pixel density_fb.

    On a `visual-*` OGBench env `info['goal']` is the goal IMAGE, not a state -- verified on
    this box, and contrary to the pixels spec's section 2. (`compute_observation` returns
    `get_pixel_observation()` for pixel envs, and `_cur_goal_ob` is that same observation.)
    The measure's hit kernel needs object coordinates, so they have to come from somewhere
    else, and the env exposes them directly: `cur_task_info['goal_xyzs']` is the (num_cubes,
    3) array of goal block positions IN METRES -- the same quantity, in the same units, as
    the `qpos[14:17]` slice `measure_qpos_slices` gathers for training.

    APPROXIMATION, stated: OGBench renders the goal image AFTER settling the scene with two
    random steps (`cube_env.py` ~600-620), so the block in the goal image can sit a few
    millimetres off `goal_xyzs`. `goal_xyzs` is the nominal target and the closest thing the
    env exposes; nothing in the env records the settled goal qpos.
    """
    info = getattr(env.unwrapped, "cur_task_info", None)
    if info is None or "goal_xyzs" not in info:
        raise AttributeError(
            "measure_qpos_slices is set, so the agent's conditioning needs the goal's object "
            "coordinates, but this env exposes no `cur_task_info['goal_xyzs']`. Without it "
            "the only goal available is an image, which the density kernel cannot slice.")
    return np.asarray(info["goal_xyzs"], np.float32).reshape(-1)


def evaluate(agent, env, task_id=None, config=None, num_eval_episodes=50,
             num_video_episodes=0, video_frame_skip=3, eval_temperature=0, eval_gaussian=None,
             any_step_success=False):
    """Evaluate the agent in the environment. Returns (stats, trajs, renders).

    `any_step_success`: td_jepa's OGBench zero-shot accounting
    (docs/notes/2026-09-03-fb-vs-tdjepa.md #7 item 3) scores an episode's success as ANY step
    reaching it, not only the final step's `info['success']` (this env build's default
    `terminate_at_goal=True` ends the episode the instant success first fires, so the two
    usually coincide -- this reducer is what makes that non-load-bearing rather than assumed).
    False by default: every existing caller is bit-identical to before this parameter existed.
    """
    config = config or {}
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2 ** 32)))
    trajs = []
    stats = defaultdict(list)
    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes
        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        # PIXEL MEASURE: hand the agent BOTH goal channels -- the image it conditions on and
        # the object-coordinate row its kernel slices. `Agent.sample_actions` merges a dict
        # into cond. Gated on the config key that defines the channel, so every state run
        # passes the bare array exactly as before.
        if config.get('measure_qpos_slices') is not None and goal is not None:
            goal = {'goal': goal, 'goal_qpos': goal_qpos_channel(env)[None]}
        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
            action = np.array(action)
            if not config.get('discrete'):
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)
            add_to(traj, dict(observation=observation, next_observation=next_observation,
                              action=action, reward=reward, done=done, info=info))
            observation = next_observation
        if i < num_eval_episodes:
            ep_info = info
            if any_step_success:
                ep_info = dict(info)
                ep_info['success'] = any(bool(s.get('success', False)) for s in traj['info'])
            add_to(stats, flatten(ep_info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))
    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajs, renders
