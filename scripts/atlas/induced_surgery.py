"""Causal keep/remove value surgery via INDUCED-POLICY rollouts (plan Step 4, done right).

Each operator DRIVES the policy at every step (not just the first action, which the flow-actor
continuation washed out). At each step, N mu-reference actions are proposed at the current state
and one is selected by the operator; that action is committed. Physical candidates (dir, hold)
are scored by branching each candidate ONE step on a SCRATCH env (qpos/qvel restore, cube-exact
for one step) so the rollout env is only ever stepped with committed actions and stays exact.

Operators (sufficiency set):
  full  argmax bilinear Q(s,a,g)          mlp  argmax MLP Q(s,a,g)
  dir   argmax dx_cube(s,a).d_g           hold argmax contact-after-one-step
Success = cube within 4cm of the goal at any point within H. Sufficiency of a candidate h:
success(h-induced) ~ success(full-induced); the goal-agnostic play data floor is `mlp`/random.

Batched across episodes per operator. Run under the jax venv.
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, '/home/adityamohan/git/Factored-FB')

from scripts.atlas import physical_fields as pf
from scripts.atlas.banks import load_banks
from scripts.atlas.critics import load_critic
from scripts.atlas.jax_load import CRITICS
from scripts.atlas.mu_flow import load_mu
from scripts.atlas.oracle import goal_cube_xyz
from scripts.atlas.prefix_replay import PrefixState, make_env, replay_to

OPS = ['full', 'dir', 'hold', 'mlp']


def _restore(env, qpos, qvel):
    import mujoco
    env.reset(seed=0, options=dict(task_id=1, render_goal=False))
    u = env.unwrapped
    u._data.qpos[:] = qpos; u._data.qvel[:] = qvel
    mujoco.mj_forward(u._model, u._data)
    mujoco.mj_rnePostConstraint(u._model, u._data)


def _branch_fields(scratch, qpos, qvel, cand):
    """dx_cube [N,3] and contact [N] for N candidate actions from state (qpos,qvel), each a
    single step on a scratch env (cube-accurate for one step). Chunks over the scratch pool."""
    N = len(cand); dx = np.zeros((N, 3), np.float32); ct = np.zeros(N, np.float32)
    c0 = qpos[pf.CUBE_QPOS]
    P = len(scratch)
    for c in range(0, N, P):
        ch = cand[c:c + P]
        for j in range(len(ch)):
            _restore(scratch[j], qpos, qvel)
            ob = scratch[j].step(np.asarray(ch[j], np.float32))[0]
            dx[c + j] = scratch[j].unwrapped._data.qpos[pf.CUBE_QPOS] - c0
            ct[c + j] = ob[pf.CONTACT_IDX]
    return dx, ct


def induced_batch(op, roll_pool, scratch, bil, mlp, mu, episodes, H, N):
    """episodes: list of (PrefixState, goal_obs). One induced rollout per episode, batched in
    chunks of len(roll_pool). Returns success[len(episodes)]."""
    import jax
    succ = np.zeros(len(episodes), np.float32)
    P = len(roll_pool)
    for c0 in range(0, len(episodes), P):
        chunk = episodes[c0:c0 + P]
        m = len(chunk)
        envs = roll_pool[:m]
        goals = np.stack([g for _, g in chunk])
        gxyz = np.stack([goal_cube_xyz(g) for _, g in chunk])
        obs = np.zeros((m, 28), np.float32)
        for j, (ps, _) in enumerate(chunk):
            obs[j] = replay_to(envs[j], ps)
        hit = np.zeros(m, bool)
        for t in range(H):
            cube = np.stack([e.unwrapped._data.qpos[pf.CUBE_QPOS] for e in envs])
            hit |= np.linalg.norm(cube - gxyz, axis=1) <= 0.04
            if hit.all():
                break
            # propose N mu actions per live episode
            props = mu.sample(obs, n=N, seed=t)                       # [m, N, 5]
            for j in range(m):
                if hit[j]:
                    continue
                cand = props[j]                                      # [N,5]
                if op in ('full', 'mlp'):
                    crit = bil if op == 'full' else mlp
                    q = crit.Q(obs[j][None], cand[None], goals[j][None])[0][:, 0]
                    a = cand[int(np.argmax(q))]
                else:
                    qpos = envs[j].unwrapped._data.qpos.copy()
                    qvel = envs[j].unwrapped._data.qvel.copy()
                    dx, ct = _branch_fields(scratch, qpos, qvel, cand)
                    if op == 'dir':
                        dg = pf.goal_dir(qpos[pf.CUBE_QPOS], goals[j])[0]
                        score = dx @ dg
                    else:                                            # hold
                        score = ct
                    a = cand[int(np.argmax(score))]
                obs[j] = envs[j].step(np.clip(a, -1, 1))[0]
        succ[c0:c0 + m] = hit.astype(np.float32)
    return succ


def _ps(b, p, k):
    L = int(b['state_prefix_len'][p, k])
    return PrefixState(seed=k % 5000, task_id=1, prefix=b['state_prefix_actions'][p, k, :L])


def run(out_path, phases, k_states, N, H, pool, scratch_n):
    b = load_banks('/home/adityamohan/git/Factored-FB/analysis/atlas/banks.npz')
    roll_pool = [make_env(task_id=1) for _ in range(pool)]
    scratch = [make_env(task_id=1) for _ in range(scratch_n)]
    bil = load_critic('bilinear_flow'); mlp = load_critic('mlp_flow')
    mu = load_mu(*CRITICS['bilinear_flow'])
    goals = b['goals_eval']

    episodes, tags = [], []
    for p in phases:
        for k in range(k_states):
            for gi in range(len(goals)):
                episodes.append((_ps(b, p, k), goals[gi]))
                tags.append((int(p), gi))
    import collections
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for op in OPS:
        s = induced_batch(op, roll_pool, scratch, bil, mlp, mu, episodes, H, N)
        for (p, gi), sv in zip(tags, s):
            res[p][op].append(float(sv))
        print(f'[{op}] done, mean success {np.mean(s):.3f}', flush=True)

    out = {str(p): {op: round(float(np.mean(res[p][op])), 3) for op in OPS} for p in phases}
    json.dump(out, open(out_path, 'w'), indent=1)
    print('\n=== success by phase x operator (induced policies) ===')
    for p in phases:
        print(f"  P{p}: " + '  '.join(f"{op}={out[str(p)][op]:.2f}" for op in OPS), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--phases', default='1,2,3')
    ap.add_argument('--k-states', type=int, default=12)
    ap.add_argument('--N', type=int, default=16)
    ap.add_argument('--H', type=int, default=40)
    ap.add_argument('--pool', type=int, default=24)
    ap.add_argument('--scratch', type=int, default=16)
    a = ap.parse_args()
    run(a.out, [int(x) for x in a.phases.split(',')], a.k_states, a.N, a.H, a.pool, a.scratch)
