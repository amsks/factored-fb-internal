"""Action-grounded mode decode: what does the bilinear value basis represent, and do the
actions the policy actually takes (vs MLP, vs the dataset) use it?

Per state s (phase-labelled) and eval goal g:
  1. A_bil, A_mlp over the mu action bank -> SVD modes u_k(a), v_k(g).
  2. Physical decode: correlate each leading u_k with cube-dx fields (x,y,z) + grasp/contact;
     each v_k with goal-direction (x,y,z) / distance / height / goal-common.
  3. Actions actually taken: a_bil, a_mlp (sample_actions), a_data (dataset), a_mu (BC flow).
     Score each on T_dir (goal-directed transport) and its loading on the top modes.
  4. Key contrasts: T_dir(a_bil) vs a_mlp vs a_data  ->  does bilinear steer more than the
     goal-agnostic dataset (directional composition) or match it (scripted primitive)?

Sharded + BOUNDED to avoid IO saturation: --n-shards / --shard, one env + 2 critics per shard,
staggered launch by the caller. Run under the jax venv.
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, '/home/adityamohan/git/Factored-FB')

from scripts.atlas import physical_fields as pf
from scripts.atlas.banks import load_banks
from scripts.atlas.build_atlas import center_and_standardize
from scripts.atlas.critics import load_critic
from scripts.atlas.jax_load import load_agent
from scripts.atlas.mu_flow import load_mu
from scripts.atlas.jax_load import CRITICS
from scripts.atlas.prefix_replay import make_env

N_ACT = 128          # mu-reference actions per state used for the atlas/modes
K_MODES = 3


def _svd_modes(A, k=K_MODES):
    U, S, Vt = np.linalg.svd(np.asarray(A, np.float64), full_matrices=False)
    return U[:, :k], S[:k], Vt[:k].T          # u_k [n_a,k], s_k, v_k [n_g,k]


def _corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float((x @ y) / d) if d > 1e-12 else 0.0


def _decode_action_modes(u_k, dx_cube, contact):
    """For each action mode, best-correlated physical field + value."""
    fields = {'cube_dx_x': dx_cube[:, 0], 'cube_dx_y': dx_cube[:, 1],
              'cube_dx_z': dx_cube[:, 2], 'contact': contact}
    out = []
    for j in range(u_k.shape[1]):
        cs = {f: _corr(u_k[:, j], v) for f, v in fields.items()}
        best = max(cs, key=lambda f: abs(cs[f]))
        out.append({'best_field': best, 'corr': round(cs[best], 3),
                    'all': {f: round(c, 3) for f, c in cs.items()}})
    return out


def _decode_goal_modes(v_k, goals, state_cube):
    dirs = np.stack([pf.goal_dir(state_cube, g)[0] for g in goals])   # [n_g,3]
    dist = np.array([pf.goal_dir(state_cube, g)[1] for g in goals])
    height = np.array([pf.obs_cube_xyz(g)[2] for g in goals])
    fields = {'goal_dir_x': dirs[:, 0], 'goal_dir_y': dirs[:, 1], 'goal_dir_z': dirs[:, 2],
              'goal_dist': dist, 'goal_height': height, 'goal_common': np.ones(len(goals))}
    out = []
    for j in range(v_k.shape[1]):
        cs = {f: _corr(v_k[:, j], v) for f, v in fields.items()}
        best = max(cs, key=lambda f: abs(cs[f]))
        out.append({'best_field': best, 'corr': round(cs[best], 3)})
    return out


def _policy_action(agent, obs, goal):
    import jax
    a = agent.sample_actions(observations=np.asarray(obs)[None], goals=np.asarray(goal)[None],
                             seed=jax.random.PRNGKey(0), temperature=0.0)
    return np.clip(np.asarray(a)[0], -1, 1)


def run_shard(shard, n_shards, out_path, goal_key='goals_eval'):
    b = load_banks('/home/adityamohan/git/Factored-FB/analysis/atlas/banks.npz')
    env = make_env(task_id=1)
    bil = load_critic('bilinear_flow')
    mlp = load_critic('mlp_flow')
    mu = load_mu(*CRITICS['bilinear_flow'])
    goals = b[goal_key]
    rng = np.random.RandomState(0)

    keys = [(p, k) for p in range(4) for k in range(b['state_obs'].shape[1])]
    keys = keys[shard::n_shards]                          # this shard's states
    recs = []
    for (p, k) in keys:
        obs = b['state_obs'][p, k]; qpos = b['state_qpos'][p, k]; qvel = b['state_qvel'][p, k]
        idx = rng.choice(b['A_mu'].shape[2], N_ACT, replace=False)
        acts = b['A_mu'][p, k][idx]
        mu_center = b['mu_center_samples'][p, k]
        state_cube = qpos[pf.CUBE_QPOS]
        # physical fields for the action bank (one-step cube dx + contact)
        dx, contact = pf.cube_delta_batch(env, qpos, qvel, acts)
        # atlas + modes for both critics
        rec = {'phase': int(p), 'k': int(k)}
        for name, crit in (('bil', bil), ('mlp', mlp)):
            Qa = crit.Q(obs[None], acts[None], goals)[0]
            Qm = crit.Q(obs[None], mu_center[None], goals)[0]
            A = center_and_standardize(Qa, Qm)
            u_k, s_k, v_k = _svd_modes(A)
            rec[f'{name}_action_modes'] = _decode_action_modes(u_k, dx, contact)
            rec[f'{name}_goal_modes'] = _decode_goal_modes(v_k, goals, state_cube)
            rec[f'{name}_sv'] = [round(float(x), 3) for x in s_k]
        # actions actually taken, scored on T_dir per goal (mean over goals)
        a_bil = _policy_action(bil.agent, obs, goals[0])
        a_mlp = _policy_action(mlp.agent, obs, goals[0])
        a_mu = mu.sample(obs[None], n=1, seed=0)[0, 0]
        taken = {'bil': a_bil, 'mlp': a_mlp, 'mu': a_mu}
        # dataset action Δx is added by the aggregator from recorded consecutive obs; here we
        # score the model/mu actions by simulating their one-step dx.
        tdir = {}
        for nm, a in taken.items():
            d1, _ = pf.cube_delta_batch(env, qpos, qvel, a[None])
            scores = [float(pf.T_dir(d1, pf.goal_dir(state_cube, g)[0])[0]) for g in goals]
            tdir[nm] = round(float(np.mean(scores)), 4)
        rec['T_dir_taken'] = tdir
        recs.append(rec)
    json.dump(recs, open(out_path, 'w'), indent=1)
    print(f'shard {shard}/{n_shards}: {len(recs)} states -> {out_path}', flush=True)


def _phase_of(obs, qpos):
    contact = obs[pf.CONTACT_IDX] > 0.5
    lifted = qpos[pf.CUBE_QPOS][2] > 0.05
    if contact and lifted:
        return 3
    if contact and not lifted:
        return 2
    return 1


def run_dataset(out_path, n_per_phase=40):
    """Part B: at DATASET states, is the bilinear policy's action MORE goal-directed than the
    play-policy action that was actually recorded? a_data Δx is free (recorded s->s'); the
    policy Δx is one-step simulated. Answers the scripted-primitive vs directional-composition
    question against the real behavior distribution."""
    import numpy as np
    d = np.load('/home/adityamohan/.ogbench/data/cube-single-play-v0.npz')
    obs, act, qpos, qvel, term = d['observations'], d['actions'], d['qpos'], d['qvel'], d['terminals']
    ends = set(np.flatnonzero(term).tolist())
    env = make_env(task_id=1)
    bil = load_critic('bilinear_flow'); mlp = load_critic('mlp_flow')
    goals = load_banks('/home/adityamohan/git/Factored-FB/analysis/atlas/banks.npz')['goals_eval']
    rng = np.random.RandomState(0)

    buckets = {1: [], 2: [], 3: []}
    order = rng.permutation(len(obs) - 1)
    for i in order:
        if int(i) in ends:
            continue
        ph = _phase_of(obs[i], qpos[i])
        if len(buckets[ph]) < n_per_phase:
            buckets[ph].append(int(i))
        if all(len(v) >= n_per_phase for v in buckets.values()):
            break

    recs = []
    for ph, idxs in buckets.items():
        for i in idxs:
            sc = qpos[i][pf.CUBE_QPOS]
            dgs = [pf.goal_dir(sc, g)[0] for g in goals]
            # dataset action: recorded one-step cube dx (free, exact)
            dx_data = (qpos[i + 1][pf.CUBE_QPOS] - sc).astype(np.float32)
            tdir_data = float(np.mean([dx_data @ dg for dg in dgs]))
            # policy actions: simulate one step from the dataset state
            row = {'phase': ph, 'idx': int(i), 'T_dir': {'data': round(tdir_data, 4)}}
            for nm, crit in (('bil', bil), ('mlp', mlp)):
                a = _policy_action(crit.agent, obs[i], goals[0])
                dx, _ = pf.cube_delta_batch(env, qpos[i], qvel[i], a[None])
                row['T_dir'][nm] = round(float(np.mean([pf.T_dir(dx, dg)[0] for dg in dgs])), 4)
            recs.append(row)
    json.dump(recs, open(out_path, 'w'), indent=1)
    print(f'dataset part: {len(recs)} states ({n_per_phase}/phase) -> {out_path}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['bank', 'dataset'], default='bank')
    ap.add_argument('--n-shards', type=int, default=1)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--goal-key', default='goals_eval')   # goals_eval (5) | goals_grid (64)
    a = ap.parse_args()
    if a.part == 'dataset':
        run_dataset(a.out)
    else:
        run_shard(a.shard, a.n_shards, a.out, goal_key=a.goal_key)
