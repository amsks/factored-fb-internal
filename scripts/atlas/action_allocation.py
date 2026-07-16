"""Does the learned critic allocate value-variation to ACTIONS or to STATES?

Tests the research note's speculation that a full successor-measure loss is dominated by the
shared future M-bar_mu, so the trained FB representation under-resolves the action-dependent
part Delta M_mu. Empirical signature: in the trained FB critic, almost none of the Q-variation
is action-driven.

On the frozen atlas bank (states x eval goals x mu-sampled actions), decompose Q-variance:
  within(g)  = mean_s Var_{a~mu} Q(s,a,g)          # action-driven (at fixed state)
  between(g) = Var_s [ E_{a~mu} Q(s,a,g) ]         # state-driven
  action_fraction(g) = within / (within + between)  # scale-invariant
Also the centered-operator energy: mean_s || Q(s,.,g) - mean_a Q ||^2 / || Q(s,.,g) ||^2.
For the factored reps additionally report the forward-map action-energy fraction
  ||Delta F_mu||^2 / ||F||^2   (FB: F=phi(s,a,z);  bilinear: F=phi(s,a)).

Run FB under the repo torch .venv; bilinear/mlp under /dev/shm/.venv-jax.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--critic', required=True, choices=['bilinear_flow', 'mlp_flow', 'fb'])
    ap.add_argument('--n-act', type=int, default=64, help='mu actions per state from A_mu')
    ap.add_argument('--out-tag', default='')
    args = ap.parse_args()

    from scripts.atlas.critics import load_critic

    banks = np.load(REPO / 'analysis/atlas/banks.npz')
    goals = banks['goals_eval']                       # [5,28]
    crit = load_critic(args.critic)

    # gather Q[state, action] per goal over all atlas states
    within, between, adv_energy = [], [], []
    P, K = banks['state_obs'].shape[:2]
    for gi in range(goals.shape[0]):
        g = goals[gi][None]                           # [1,28]
        meanA, varA, adv = [], [], []
        for phase in range(P):
            for k in range(K):
                obs = banks['state_obs'][phase, k]
                A = banks['A_mu'][phase, k][:args.n_act]        # [n_act,5]
                Q = np.asarray(crit.Q(obs[None], A[None], g))[0, :, 0]   # [n_act]
                meanA.append(Q.mean())
                varA.append(Q.var())
                c = Q - Q.mean()
                adv.append((c @ c) / (Q @ Q + 1e-12))
        meanA = np.array(meanA); varA = np.array(varA)
        within.append(float(varA.mean()))
        between.append(float(meanA.var()))
        adv_energy.append(float(np.mean(adv)))
    within = np.array(within); between = np.array(between)
    frac = within / (within + between + 1e-12)

    out = dict(critic=args.critic, n_act=args.n_act,
               action_fraction=float(frac.mean()),
               action_fraction_per_goal=frac.tolist(),
               within_state_var=float(within.mean()),
               between_state_var=float(between.mean()),
               centered_advantage_energy=float(np.mean(adv_energy)))
    print(f"  [{args.critic}] action_fraction {out['action_fraction']:.3f}  "
          f"(within {within.mean():.3g} / between {between.mean():.3g})  "
          f"centered_adv_energy {out['centered_advantage_energy']:.3f}", flush=True)

    p = REPO / f"analysis/atlas/action_allocation_{args.critic}{args.out_tag}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[action-alloc] -> {p}", flush=True)


if __name__ == '__main__':
    main()
