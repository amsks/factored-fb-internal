"""Privileged, behavior-CONSTRAINED control oracle. NOT ground truth, no star.

J^MPC_mu(s, a, g): from the state reconstructed by prefix-replay, execute the first action,
then estimate the in-support optimal return by best-of-N closed-loop mu rollouts (proposals
from mu = on-manifold; SELECTION by realized discounted goal-hit return = critic-free, aimed
at the in-support optimum, not the behavior mean). Softmax-averaged at MPPI_TEMP.

Tractability: the N candidate rollouts step a POOL of envs in lockstep with ONE batched
mu.sample per timestep, so the jax cost is O(H) calls, not O(N*H). MUST run under the jax venv.

obs layout (cube_env.compute_observation): obs[18]=contact, obs[19:22]=(cube-[.425,0,0])*10.
"""
import numpy as np

from scripts.atlas import config as C
from scripts.atlas.prefix_replay import make_env, replay_to

_XYZ_CENTER = np.array([0.425, 0.0, 0.0], np.float32)
_QPOS_CUBE = slice(14, 17)


def goal_cube_xyz(goal_obs):
    """Physical cube xyz target from a goal OBSERVATION (obs[19:22] is scaled+offset)."""
    return np.asarray(goal_obs, np.float32)[19:22] / 10.0 + _XYZ_CENTER


def make_pool(n):
    return [make_env(task_id=1) for _ in range(n)]


def oracle_J(pool, mu, ps, first_action, goal_obs, replicates=None, H=None, gamma=0.99):
    """pool: list of n_seq envs. Returns a scalar J estimate (softmax-weighted over the
    n_seq candidate rollouts, averaged over replicates)."""
    reps = replicates or C.MPPI_REPLICATES
    H = H or C.MPPI_H
    n = len(pool)
    goal_xyz = goal_cube_xyz(goal_obs)
    per_rep = []
    for rep in range(reps):
        # reconstruct the SAME state in every env, execute the shared first action
        obs = np.zeros((n, 28), np.float32)
        for j, env in enumerate(pool):
            replay_to(env, ps)
            o, *_ = env.step(np.asarray(first_action, np.float32))
            obs[j] = o
        hit_t = np.full(n, -1, dtype=int)                  # first hit timestep, -1 = none
        for t in range(H):
            done = hit_t >= 0
            # cube position per env (physical, from qpos)
            for j, env in enumerate(pool):
                if done[j]:
                    continue
                cube = env.unwrapped._data.qpos[_QPOS_CUBE]
                if np.linalg.norm(cube - goal_xyz) <= 0.04:
                    hit_t[j] = t
                    done[j] = True
            if done.all():
                break
            acts = mu.sample(obs, n=1, seed=rep * 1_000_000 + t)[:, 0, :]   # [n,5] batched
            for j, env in enumerate(pool):
                if done[j]:
                    continue
                o, *_ = env.step(acts[j])
                obs[j] = o
        # discounted pre-hit cost: -(1-gamma^N)/(1-gamma); N = hit_t if hit else H
        N = np.where(hit_t >= 0, hit_t, H).astype(np.float64)
        rets = -(1 - gamma ** N) / (1 - gamma)
        w = np.exp((rets - rets.max()) / C.MPPI_TEMP)
        w /= w.sum()
        per_rep.append(float((w * rets).sum()))
    return float(np.mean(per_rep))
