"""Exp 5 (partial, static): advantage-geometry metrics for the bilinear critic at one
checkpoint step, over the frozen atlas banks. Run at 500k and 1M to read drift after
convergence (teacher-then-drift leaves a signature here even with only 2 points).

  effective_rank : exp(entropy of the singular spectrum) of the centered operator A(a,g)
                   over the 64-goal grid, averaged over atlas states. (Finding 2 baseline ~6.7.)
  offref_peak_gap: max_{a in WIDE} Q(s,a,g) - max_{a in mu-ref} Q(s,a,g), averaged over
                   states x eval goals. >0 = the critic's action-maximum sits OFF the mu
                   support (Finding 1). Growth 500k->1M = the off-support peak sharpening.
  support_frac   : fraction of (state,goal) where the wide-set argmax beats every mu-ref action.

Run under /dev/shm/.venv-jax/bin/python.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.atlas.build_atlas import center_and_standardize   # noqa: E402


def _eff_rank(sv):
    p = sv / (sv.sum() + 1e-12)
    ent = -(p * np.log(p + 1e-12)).sum()
    return float(np.exp(ent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--critic', default='bilinear_flow')
    ap.add_argument('--run-dir', default=None, help='override checkpoint dir (default: registry)')
    ap.add_argument('--n-states', type=int, default=16)
    ap.add_argument('--out-tag', default='')
    args = ap.parse_args()

    from scripts.atlas.critics import _JaxGCIQL
    from scripts.atlas.jax_load import CRITICS, load_agent

    d = args.run_dir or CRITICS[args.critic][0]
    crit = _JaxGCIQL(load_agent(d, args.step))
    banks = np.load(REPO / 'analysis/atlas/banks.npz')
    grid = banks['goals_grid'][:16]       # subsample for a fast effective-rank estimate
    ev = banks['goals_eval']              # [5,28]
    wide = banks['A_global']              # [128,5] shared Sobol space-filling (off-ref probe)

    eff, gaps, wins = [], [], []
    for phase in range(4):
        for k in range(min(args.n_states, banks['state_obs'].shape[1])):
            obs = banks['state_obs'][phase, k]
            A_mu = banks['A_mu'][phase, k]              # [256,5] mu-reference
            mu_center = banks['mu_center_samples'][phase, k]
            # advantage effective rank over the 64-goal grid
            Q = crit.Q(obs[None], A_mu[None], grid)[0]          # [256,64]
            Qc = center_and_standardize(Q, crit.Q(obs[None], mu_center[None], grid)[0])
            sv = np.linalg.svd(Qc, compute_uv=False)
            eff.append(_eff_rank(sv))
            # off-reference peak gap over eval goals
            for gi in range(ev.shape[0]):
                g = ev[gi][None]
                q_ref = crit.Q(obs[None], A_mu[None], g)[0, :, 0]
                q_wide = crit.Q(obs[None], wide[None], g)[0, :, 0]
                gaps.append(float(q_wide.max() - q_ref.max()))
                wins.append(float(q_wide.max() > q_ref.max()))

    out = dict(step=args.step, critic=args.critic,
               effective_rank=float(np.mean(eff)),
               offref_peak_gap=float(np.mean(gaps)),
               offref_win_frac=float(np.mean(wins)),
               n_states=len(eff))
    print(f"  step {args.step}: eff_rank {out['effective_rank']:.2f}  "
          f"offref_peak_gap {out['offref_peak_gap']:+.3f}  win_frac {out['offref_win_frac']:.2f}",
          flush=True)
    p = REPO / f"analysis/atlas/ckpt_static_{args.critic}_{args.step}{args.out_tag}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[ckpt-static] -> {p}", flush=True)


if __name__ == '__main__':
    main()
