"""Run the oracle-free atlas analysis over all critics and states. No simulator.
Usage (GPU 0, no contention with the CPU-only oracle run):
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
    /dev/shm/.venv-jax/bin/python scripts/atlas/run_oracle_free.py
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, '/home/adityamohan/git/Factored-FB')

from scripts.atlas.banks import load_banks
from scripts.atlas.build_atlas import center_and_standardize
from scripts.atlas.critics import load_critic
from scripts.atlas.mode_analysis import (basin_count, effective_rank,
                                         principal_angles, top_sv_share)

CRITICS = ['bilinear_flow', 'mlp_flow', 'simplesf', 'fb']  # fb loads under torch; skip if jax-only
N_STATES = 8          # per phase
N_ACT = 128           # subsample of the 256-action mu bank
GOALS = 'goals_grid'  # 64 goals -> effective rank is not n_g-capped


def atlas_col(critic, obs, acts, mu_center, goals):
    # one critic.Q call each (the adapter loops goals internally); centered+standardized.
    Q = critic.Q(obs[None], acts[None], goals)[0]          # [n_a, n_g]
    Qmu = critic.Q(obs[None], mu_center[None], goals)[0]   # [N_MU, n_g]
    return center_and_standardize(Q, Qmu)


def batched_atlas(critic, obs, acts_all, mu_all, goals):
    """Vectorized: evaluate ALL states in one go per critic.
    acts_all [S,na,5], mu_all [S,N_MU,5] (same obs order). Returns list of (A, acts) per state.
    Collapses the S*na*ng and S*N_MU*ng grids into single critic calls -> far fewer dispatches.
    """
    S = obs.shape[0]
    outs = []
    # action grid: [S, na, ng]
    Qa = critic.Q(obs, acts_all, goals)                    # [S, na, ng]
    Qm = critic.Q(obs, mu_all, goals)                      # [S, N_MU, ng]
    for s in range(S):
        outs.append((center_and_standardize(Qa[s], Qm[s]), acts_all[s]))
    return outs


def main():
    b = load_banks('/home/adityamohan/git/Factored-FB/analysis/atlas/banks.npz')
    goals = b[GOALS]
    rng = np.random.RandomState(0)
    per_critic = {}
    A_store = {}    # (critic, phase, k) -> A  (kept for cross-critic angles)

    for name in CRITICS:
        try:
            crit = load_critic(name)
        except Exception as e:
            print(f'SKIP {name}: {type(e).__name__}: {e}', flush=True)
            continue
        t0 = time.time()
        erank, tsv, basins = [], [], []
        # gather all (phase,k) states into one batched critic evaluation
        keys = [(p, k) for p in range(4) for k in range(N_STATES)]
        obs_all = np.stack([b['state_obs'][p, k] for p, k in keys])           # [S,28]
        idxs = [rng.choice(b['A_mu'].shape[2], N_ACT, replace=False) for _ in keys]
        acts_all = np.stack([b['A_mu'][p, k][ix] for (p, k), ix in zip(keys, idxs)])
        mu_all = np.stack([b['mu_center_samples'][p, k] for p, k in keys])     # [S,N_MU,5]
        results = batched_atlas(crit, obs_all, acts_all, mu_all, goals)
        for (p, k), (A, acts) in zip(keys, results):
            A_store[(name, p, k)] = (A, acts)
            erank.append(effective_rank(A))
            tsv.append(top_sv_share(A))
            basins.append(np.mean([basin_count(A[:, gi], acts)
                                   for gi in range(0, goals.shape[0], 8)]))
        per_critic[name] = dict(
            eff_rank=float(np.mean(erank)), eff_rank_sd=float(np.std(erank)),
            top_sv_share=float(np.mean(tsv)), mean_basins=float(np.mean(basins)),
            n=len(erank), secs=round(time.time() - t0))
        print(f'{name:14} eff_rank={per_critic[name]["eff_rank"]:.2f}  '
              f'top_sv={per_critic[name]["top_sv_share"]:.3f}  '
              f'basins={per_critic[name]["mean_basins"]:.2f}  ({per_critic[name]["secs"]}s)',
              flush=True)

    # cross-critic: principal angles between bilinear and MLP goal-mode subspaces (matched states)
    angles = []
    for p in range(4):
        for k in range(N_STATES):
            if ('bilinear_flow', p, k) in A_store and ('mlp_flow', p, k) in A_store:
                a = principal_angles(A_store[('bilinear_flow', p, k)][0],
                                     A_store[('mlp_flow', p, k)][0], k=3)
                angles.append(a)
    if angles:
        ang = np.rad2deg(np.mean(angles, axis=0))
        per_critic['_bilinear_vs_mlp_principal_angles_deg'] = [round(float(x), 1) for x in ang]
        print(f'\nbilinear vs MLP goal-mode principal angles (deg, top-3): '
              f'{[round(float(x),1) for x in ang]}  (0=identical structure, 90=orthogonal)')

    out = '/home/adityamohan/git/Factored-FB/analysis/atlas/oracle_free_results.json'
    json.dump(per_critic, open(out, 'w'), indent=1)
    print(f'\nwrote {out}', flush=True)


if __name__ == '__main__':
    main()
