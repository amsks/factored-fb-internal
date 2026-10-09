"""The headline figure fits FB weighted by rho and the density model unweighted.
Score both estimators under BOTH conventions, so the estimator effect and the weighting
effect are separated. Closed-form only -- no training, no seeds except the CE restart."""
import os, sys
sys.argv = [sys.argv[0], "--corridor14"]
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import run as R

DR = 8
print(f"  {'k':>3} {'estimator / weight':34s} {'occ sq err':>11s} {'door TV':>9s} {'margin':>8s}")
for K in (1, 10, 40):
    P = R.sticky_action_matrices(K); Ppi = R.policy_matrix_uniform(P)
    rho, _, _ = R.stationary(Ppi)
    unif = np.full(R.S, 1.0 / R.S)
    M = R.successor_measure(P, Ppi, R.GAMMA)
    gt, _, _, _ = R.door_metrics({a: (M @ R.R_ROOM)[R.ROW_D[a]] for a in R.ACTIONS})
    p_true = {a: (1 - R.GAMMA) * M[R.ROW_D[a]] for a in (R.A_CROSS, R.A_FAIL_SHOW)}

    def rep(tag, occ, Q):
        err = float(np.mean((occ - M) ** 2))
        tv = np.mean([R.tv_half(np.clip(occ[R.ROW_D[a]], 0, None) /
                                np.clip(occ[R.ROW_D[a]], 0, None).sum(), p_true[a])
                      for a in (R.A_CROSS, R.A_FAIL_SHOW)])
        g, _, _, _ = R.door_metrics({a: Q[R.ROW_D[a]] for a in R.ACTIONS})
        print(f"  {K:>3} {tag:34s} {err:11.4f} {tv:9.3f} {g/gt:8.3f}")

    for wn, w in [("rho (as published)", rho), ("uniform (density's convention)", unif)]:
        F, B = R.fb_fit(M, w, DR)
        rep(f"FB SVD / {wn}", (F @ B.T) * w[None, :], R.fb_readout(F, B, w, R.R_ROOM))
    Fc, Bc, _ = R.density_fit((1 - R.GAMMA) * M, DR, seed=DR * 17 + K * 3, restarts=1)
    Qc, pc = R.density_readout(Fc, Bc, R.R_ROOM, R.GAMMA)
    rep("Density CE / uniform (as published)", pc / (1 - R.GAMMA), Qc)
    print()
