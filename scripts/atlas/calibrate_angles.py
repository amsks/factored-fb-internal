"""Calibrate the bilinear-vs-MLP goal-mode principal angles against baselines:
  same-critic different-seed (noise floor), cross-critic, and random subspaces (ceiling).
If theta(bil,mlp) >> theta(bil_i,bil_j), the content difference is real, not seed noise.

Oracle-free. Run under the jax venv, CPU:
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES="" \
    /dev/shm/.venv-jax/bin/python scripts/atlas/calibrate_angles.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/home/adityamohan/git/Factored-FB')

from scripts.atlas.banks import load_banks
from scripts.atlas.build_atlas import center_and_standardize
from scripts.atlas.critics import _JaxGCIQL
from scripts.atlas.jax_load import load_agent
from scripts.atlas.mode_analysis import principal_angles

_B = os.path.expanduser('~/bridge_ckpt_backup/gciql_outputs/factored-fb')
SEEDS = {
    'bil': [(f'{_B}/GCIQL-State-Flow-Bilinear-d50/sd000_20260615_011444', 500000),
            (f'{_B}/GCIQL-State-Flow-Bilinear-d50/sd001_20260615_011447', 500000),
            (f'{_B}/GCIQL-State-Flow-Bilinear-d50/sd002_20260615_011450', 500000)],
    'mlp': [(f'{_B}/GCIQL-State-Flow/sd000_20260609_234409', 1000000),
            (f'{_B}/GCIQL-State-Flow/sd001_20260609_234411', 1000000),
            (f'{_B}/GCIQL-State-Flow/sd002_20260609_234413', 1000000)],
}
N_STATES = 12        # phases 0 & 2, 6 each
K_MODES = 3


def atlas_for(critic, banks, keys, acts_all, mu_all, goals):
    obs = np.stack([banks['state_obs'][p, k] for p, k in keys])
    Qa = critic.Q(obs, acts_all, goals)
    Qm = critic.Q(obs, mu_all, goals)
    return [center_and_standardize(Qa[s], Qm[s]) for s in range(len(keys))]


def mean_angles(As, Bs):
    """Mean principal angles (deg, top-K) over matched states."""
    ang = [principal_angles(A, B, k=K_MODES) for A, B in zip(As, Bs)]
    return np.rad2deg(np.mean(ang, axis=0))


def main():
    b = load_banks('/home/adityamohan/git/Factored-FB/analysis/atlas/banks.npz')
    goals = b['goals_grid']
    keys = [(p, k) for p in (0, 2) for k in range(N_STATES // 2)]
    rng = np.random.RandomState(0)
    idxs = [rng.choice(b['A_mu'].shape[2], 128, replace=False) for _ in keys]
    acts_all = np.stack([b['A_mu'][p, k][ix] for (p, k), ix in zip(keys, idxs)])
    mu_all = np.stack([b['mu_center_samples'][p, k] for p, k in keys])

    # atlas per (critic, seed)
    A = {}
    for fam, runs in SEEDS.items():
        for si, (d, step) in enumerate(runs):
            crit = _JaxGCIQL(load_agent(d, step))
            A[(fam, si)] = atlas_for(crit, b, keys, acts_all, mu_all, goals)
            print(f'atlas built: {fam} seed{si}', flush=True)

    out = {}
    # same-critic different-seed (noise floor)
    for fam in ('bil', 'mlp'):
        pairs = [(0, 1), (0, 2), (1, 2)]
        angs = [mean_angles(A[(fam, i)], A[(fam, j)]) for i, j in pairs]
        out[f'{fam}_vs_{fam}_seed'] = [round(float(x), 1) for x in np.mean(angs, axis=0)]
    # cross-critic (all 3x3 seed pairs)
    cross = [mean_angles(A[('bil', i)], A[('mlp', j)]) for i in range(3) for j in range(3)]
    out['bil_vs_mlp'] = [round(float(x), 1) for x in np.mean(cross, axis=0)]
    # random rank-K subspaces in R^{n_g} (ceiling)
    ng = goals.shape[0]
    rand = []
    for _ in range(200):
        R1 = np.linalg.qr(rng.randn(ng, K_MODES))[0]
        R2 = np.linalg.qr(rng.randn(ng, K_MODES))[0]
        s = np.linalg.svd(R1.T @ R2, compute_uv=False)
        rand.append(np.rad2deg(np.arccos(np.clip(s, -1, 1))))
    out['random_vs_random'] = [round(float(x), 1) for x in np.mean(rand, axis=0)]

    print('\n=== principal angles (deg, top-3): 0=identical, 90=orthogonal ===')
    for k in ('bil_vs_bil_seed', 'mlp_vs_mlp_seed', 'bil_vs_mlp', 'random_vs_random'):
        print(f'  {k:20} {out[k]}')
    p = '/home/adityamohan/git/Factored-FB/analysis/atlas/angle_calibration.json'
    json.dump(out, open(p, 'w'), indent=1)
    print(f'\nwrote {p}', flush=True)


if __name__ == '__main__':
    main()
