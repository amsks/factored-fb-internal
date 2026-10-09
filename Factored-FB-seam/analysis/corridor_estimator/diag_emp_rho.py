"""The candidates u are drawn from the DATASET's next-states, not from the analytic
stationary distribution (the dataset resets every 500 steps). Weight the readout by the
empirical distribution of sp_idx instead of by rho, and see whether the margin recovers."""
import os, sys
sys.argv = [sys.argv[0], "--corridor14"]
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, jax.numpy as jnp
from jax import random
import run as R

K, DR, STEPS = 40, 8, 100_000
P = R.sticky_action_matrices(K); Ppi = R.policy_matrix_uniform(P)
rho, _, _ = R.stationary(Ppi)
M = R.successor_measure(P, Ppi, R.GAMMA)
gt, _, _, _ = R.door_metrics({a: (M @ R.R_ROOM)[R.ROW_D[a]] for a in R.ACTIONS})
p_true = {a: (1 - R.GAMMA) * M[R.ROW_D[a]] for a in (R.A_CROSS, R.A_FAIL_SHOW)}
print(f"true margin {gt:.3f}")

for sd in (0, 1, 2):
    s_i, a_i, sp_i = R.generate_dataset_td(K, sd)
    emp = np.bincount(np.asarray(sp_i), minlength=R.S).astype(float); emp /= emp.sum()
    if sd == 0:
        print(f"  |rho - empirical|_1 = {np.abs(rho - emp).sum():.4f}   "
              f"door: rho {rho[R.D]:.4f} emp {emp[R.D]:.4f}")
    key = random.split(random.PRNGKey(20_000 * K + sd))[1]
    F, B, _ = R._train_density_td(key, jnp.asarray(s_i), jnp.asarray(a_i), jnp.asarray(sp_i),
                                  DR, STEPS, R.TRAIN_LR, R.TRAIN_TAU,
                                  R.TRAIN_BATCH_T, R.TRAIN_BATCH_U, R.GAMMA)
    _, p = R.density_readout(np.asarray(F), np.asarray(B), R.R_ROOM, R.GAMMA)
    out = []
    for tag, wgt in [("plain", None), ("rho", rho), ("empirical", emp)]:
        pp = p if wgt is None else (p * wgt[None, :]) / (p * wgt[None, :]).sum(-1, keepdims=True)
        Q = (1 / (1 - R.GAMMA)) * (pp @ R.R_ROOM)
        g, _, _, _ = R.door_metrics({a: Q[R.ROW_D[a]] for a in R.ACTIONS})
        tv = np.mean([R.tv_half(np.clip(pp[R.ROW_D[a]], 0, None) /
                                np.clip(pp[R.ROW_D[a]], 0, None).sum(), p_true[a])
                      for a in (R.A_CROSS, R.A_FAIL_SHOW)])
        out.append(f"{tag:9s} margin {g/gt:.3f} TV {tv:.3f}")
    print(f"  seed {sd}: " + " | ".join(out))
