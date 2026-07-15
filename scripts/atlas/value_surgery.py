"""Causal keep/remove value surgery (plan Step 4). Necessity + sufficiency of a candidate
mechanism, scored by SIMULATOR OUTCOME (not correlation).

At a phase-anchored state s and eval goal g, over a mu-reference action bank we select the first
action by six operators:
  full     argmax A_bil(s,a,g)                 (the bilinear value)
  dir      argmax T_dir(s,a,g)                 (goal-directed transport candidate)
  hold     argmax T_hold(s,a)                  (control-retention candidate)
  rm_dir   argmax [A_bil - proj_{T_dir} A_bil] (bilinear with the direction component removed)
  rm_hold  argmax [A_bil - proj_{T_hold} A_bil]
  mlp      argmax A_mlp(s,a,g)                 (reference)
Then take that first action and CONTINUE with the bilinear flow actor for H steps; success =
cube within 4cm of the goal at any point. Sufficiency: success(dir/hold) ~ success(full).
Necessity: success(rm_dir/rm_hold) drops toward mlp.

Rollout scoring is BATCHED: all (state,goal,operator) episodes step a pooled set of envs in
lockstep with one flow-actor call per timestep (OOD-grid pattern). Run under the jax venv.
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

OPS = ['full', 'dir', 'hold', 'rm_dir', 'rm_hold', 'mlp']


def _proj_remove(A_col, T):
    """A_col, T are length-n vectors over actions. Return A with its T-aligned part removed."""
    tt = float(T @ T)
    if tt < 1e-12:
        return A_col.copy()
    return A_col - (float(A_col @ T) / tt) * T


def select_actions(bil, mlp, obs, acts, goal, dx, Thold):
    """Return {op: chosen action}. dx (one-step cube displacement) and Thold (control-retention)
    are goal-INDEPENDENT and precomputed once per state; only Tdir depends on the goal."""
    Qa_b = bil.Q(obs[None], acts[None], goal[None])[0][:, 0]
    Qa_m = mlp.Q(obs[None], acts[None], goal[None])[0][:, 0]
    A_b = Qa_b - Qa_b.mean()
    A_m = Qa_m - Qa_m.mean()
    dg = pf.goal_dir(obs_cube_of(obs), goal)[0]
    Tdir = pf.T_dir(dx, dg)
    return {
        'full': acts[int(np.argmax(A_b))],
        'dir': acts[int(np.argmax(Tdir))],
        'hold': acts[int(np.argmax(Thold))],
        'rm_dir': acts[int(np.argmax(_proj_remove(A_b, Tdir)))],
        'rm_hold': acts[int(np.argmax(_proj_remove(A_b, Thold)))],
        'mlp': acts[int(np.argmax(A_m))],
    }


def obs_cube_of(obs):
    """cube xyz from a state observation (obs[19:22]/10 + center)."""
    return pf.obs_cube_xyz(obs)


def _score_chunk(pool, cont_agent, episodes, H):
    """One-to-one: len(episodes) <= len(pool). Batched lockstep rollout."""
    import jax
    m = len(episodes)
    obs = np.zeros((m, 28), np.float32)
    goal_xyz = np.stack([goal_cube_xyz(g) for _, _, g in episodes])
    goals = np.stack([g for _, _, g in episodes])
    envs = pool[:m]
    for j, (ps, a0, _) in enumerate(episodes):
        replay_to(envs[j], ps)
        o, *_ = envs[j].step(np.asarray(a0, np.float32))
        obs[j] = o
    hit = np.zeros(m, bool)
    for t in range(H):
        cube = np.stack([e.unwrapped._data.qpos[pf.CUBE_QPOS] for e in envs])
        hit |= np.linalg.norm(cube - goal_xyz, axis=1) <= 0.04
        if hit.all():
            break
        a = np.clip(np.asarray(cont_agent.sample_actions(
            observations=obs, goals=goals, seed=jax.random.PRNGKey(t), temperature=0.0)), -1, 1)
        for j, e in enumerate(envs):
            if not hit[j]:
                obs[j] = e.step(a[j])[0]
    return hit.astype(np.float32)


def score_rollouts(pool, mu, cont_agent, episodes, H=100):
    """Process episodes in CHUNKS of len(pool) so no two episodes share an env (correctness)."""
    ps = len(pool)
    out = np.zeros(len(episodes), np.float32)
    for c in range(0, len(episodes), ps):
        chunk = episodes[c:c + ps]
        out[c:c + len(chunk)] = _score_chunk(pool, cont_agent, chunk, H)
    return out


def _ps(banks, p, k):
    L = int(banks['state_prefix_len'][p, k])
    return PrefixState(seed=k % 5000, task_id=1, prefix=banks['state_prefix_actions'][p, k, :L])


def run(out_path, phases, k_states, n_act, pool_size, H):
    b = load_banks('/home/adityamohan/git/Factored-FB/analysis/atlas/banks.npz')
    env = make_env(task_id=1)                     # for candidate-op sim (single)
    pool = [make_env(task_id=1) for _ in range(pool_size)]
    bil = load_critic('bilinear_flow'); mlp = load_critic('mlp_flow')
    mu = load_mu(*CRITICS['bilinear_flow'])
    goals = b['goals_eval']
    rng = np.random.RandomState(0)

    episodes, tags = [], []
    for p in phases:
        for k in range(k_states):
            obs = b['state_obs'][p, k]; qpos = b['state_qpos'][p, k]; qvel = b['state_qvel'][p, k]
            idx = rng.choice(b['A_mu'].shape[2], n_act, replace=False)
            acts = b['A_mu'][p, k][idx]
            dx, _ = pf.cube_delta_batch(env, qpos, qvel, acts)          # once per state (1-step)
            Thold = pf.T_hold_batch(pool, qpos, qvel, acts, mu)         # once per state (batched)
            for gi in range(len(goals)):
                sel = select_actions(bil, mlp, obs, acts, goals[gi], dx, Thold)
                for op, a in sel.items():
                    episodes.append((_ps(b, p, k), a, goals[gi]))
                    tags.append((int(p), int(k), gi, op))
            print(f'selected p{p} k{k}', flush=True)

    succ = score_rollouts(pool, mu, bil.agent, episodes, H=H)
    recs = [{'phase': t[0], 'k': t[1], 'goal': t[2], 'op': t[3], 'success': float(s)}
            for t, s in zip(tags, succ)]
    json.dump(recs, open(out_path, 'w'), indent=1)
    # summary
    import collections
    agg = collections.defaultdict(list)
    for r in recs:
        agg[(r['phase'], r['op'])].append(r['success'])
    print('\n=== success by phase x operator ===')
    for p in phases:
        row = ' '.join(f"{op}={np.mean(agg[(p,op)]):.2f}" for op in OPS)
        print(f'  P{p}: {row}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--phases', default='1,2,3')
    ap.add_argument('--k-states', type=int, default=16)
    ap.add_argument('--n-act', type=int, default=64)
    ap.add_argument('--pool', type=int, default=32)
    ap.add_argument('--H', type=int, default=100)
    a = ap.parse_args()
    run(a.out, [int(x) for x in a.phases.split(',')], a.k_states, a.n_act, a.pool, a.H)
