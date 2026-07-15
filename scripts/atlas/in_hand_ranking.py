"""Cheap causal test: does the bilinear critic rank post-contact outcomes better than the MLP
when BOTH are judged against the SAME shared continuation policy?

Normally Q_bil is scored against the bilinear policy's rollouts and Q_MLP against the MLP
policy's — different outcomes, so the ranking comparison is confounded by critic-actor
co-adaptation. Fix (reusing the paper's in-hand counterfactual analysis): from each in-hand
state s toward goal g, roll ONE shared continuation policy pi_shared, get a success label Y.
Score each critic by Q_m(s, a0, g) where a0 = pi_shared's first action, and compare
AUC(Q_bil, Y) vs AUC(Q_mlp, Y). If bilinear still ranks better under a common policy, its
content difference is causally relevant to post-contact control; if the gap vanishes, the
original ranking advantage was co-adaptation, not the critic.

Cheap: one rollout per (state, goal), batched; no branching, no induced policy.
Run under the jax venv.
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, '/home/adityamohan/git/Factored-FB')

from scripts.atlas.banks import load_banks
from scripts.atlas.critics import load_critic
from scripts.atlas.jax_load import CRITICS
from scripts.atlas.mu_flow import load_mu
from scripts.atlas.oracle import goal_cube_xyz
from scripts.atlas.physical_fields import CUBE_QPOS
from scripts.atlas.prefix_replay import PrefixState, make_env, replay_to


def auc(scores, labels):
    """Rank-based AUC: P(score_pos > score_neg). Returns nan if one class absent."""
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    # fraction of pos>neg pairs (+0.5 ties) via rank sum
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    r_pos = ranks[y == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _ps(b, p, k):
    L = int(b['state_prefix_len'][p, k])
    return PrefixState(seed=k % 5000, task_id=1, prefix=b['state_prefix_actions'][p, k, :L])


def rollout_labels(pool, shared_agent, episodes, H):
    """episodes: list of (PrefixState, goal). Batched lockstep rollout of shared_agent.
    Returns (Y[n] success, a0[n,5] first action taken)."""
    import jax
    n = len(episodes); P = len(pool)
    Y = np.zeros(n, np.float32); A0 = np.zeros((n, 5), np.float32)
    for c in range(0, n, P):
        chunk = episodes[c:c + P]; m = len(chunk); envs = pool[:m]
        goals = np.stack([g for _, g in chunk]); gxyz = np.stack([goal_cube_xyz(g) for _, g in chunk])
        obs = np.zeros((m, 28), np.float32)
        for j, (ps, _) in enumerate(chunk):
            obs[j] = replay_to(envs[j], ps)
        hit = np.zeros(m, bool)
        for t in range(H):
            cube = np.stack([e.unwrapped._data.qpos[CUBE_QPOS] for e in envs])
            hit |= np.linalg.norm(cube - gxyz, axis=1) <= 0.04
            a = np.clip(np.asarray(shared_agent.sample_actions(
                observations=obs, goals=goals, seed=jax.random.PRNGKey(t), temperature=0.0)), -1, 1)
            if t == 0:
                A0[c:c + m] = a
            if hit.all():
                break
            for j, e in enumerate(envs):
                if not hit[j]:
                    obs[j] = e.step(a[j])[0]
        Y[c:c + m] = hit.astype(np.float32)
    return Y, A0


def run(out_path, phases, k_states, H, pool, shared):
    b = load_banks('/home/adityamohan/git/Factored-FB/analysis/atlas/banks.npz')
    roll = [make_env(task_id=1) for _ in range(pool)]
    bil = load_critic('bilinear_flow'); mlp = load_critic('mlp_flow')
    shared_agent = (bil if shared == 'bil' else mlp).agent
    goals = b['goals_eval']

    episodes, tags = [], []
    for p in phases:
        for k in range(k_states):
            for gi in range(len(goals)):
                episodes.append((_ps(b, p, k), goals[gi])); tags.append((int(p), k, gi))
    Y, A0 = rollout_labels(roll, shared_agent, episodes, H)

    # critic value at the shared first action
    recs = []
    for e, (p, k, gi), y, a0 in zip(episodes, tags, Y, A0):
        obs = b['state_obs'][p, k]; g = goals[gi]
        qb = float(bil.Q(obs[None], a0[None][None], g[None])[0][0, 0])
        qm = float(mlp.Q(obs[None], a0[None][None], g[None])[0][0, 0])
        recs.append({'phase': p, 'k': k, 'goal': gi, 'Y': float(y), 'Qbil': qb, 'Qmlp': qm})

    out = {'shared_policy': shared, 'H': H, 'per_phase': {}}
    for p in phases:
        rs = [r for r in recs if r['phase'] == p]
        Yv = [r['Y'] for r in rs]
        out['per_phase'][str(p)] = {
            'n': len(rs), 'base_success': round(float(np.mean(Yv)), 3),
            'AUC_bil': round(auc([r['Qbil'] for r in rs], Yv), 3),
            'AUC_mlp': round(auc([r['Qmlp'] for r in rs], Yv), 3)}
    allY = [r['Y'] for r in recs]
    out['overall'] = {'n': len(recs), 'base_success': round(float(np.mean(allY)), 3),
                      'AUC_bil': round(auc([r['Qbil'] for r in recs], allY), 3),
                      'AUC_mlp': round(auc([r['Qmlp'] for r in recs], allY), 3)}
    json.dump({'summary': out, 'records': recs}, open(out_path, 'w'), indent=1)
    print(f"shared={shared} H={H}")
    for p in phases:
        s = out['per_phase'][str(p)]
        print(f"  P{p}: n={s['n']} base={s['base_success']:.2f}  AUC_bil={s['AUC_bil']:.3f}  AUC_mlp={s['AUC_mlp']:.3f}", flush=True)
    o = out['overall']
    print(f"  ALL: base={o['base_success']:.2f}  AUC_bil={o['AUC_bil']:.3f}  AUC_mlp={o['AUC_mlp']:.3f}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--phases', default='2,3')
    ap.add_argument('--k-states', type=int, default=32)
    ap.add_argument('--H', type=int, default=120)
    ap.add_argument('--pool', type=int, default=32)
    ap.add_argument('--shared', choices=['bil', 'mlp'], default='bil')
    a = ap.parse_args()
    run(a.out, [int(x) for x in a.phases.split(',')], a.k_states, a.H, a.pool, a.shared)
