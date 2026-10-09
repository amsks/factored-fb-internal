"""Is TD-trained FB approximating the SVD optimum? Measure BOTH solutions against M in
FB's OWN objective (rho-weighted squared error), and against the door margin.

If the SVD optimum has LOWER squared error but a WORSE margin, then TD is not
approximating it -- and the squared loss is choosing to discard the door."""
import os, sys
sys.argv = [sys.argv[0], "--corridor14"]
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, jax.numpy as jnp
from jax import random
import run as R

K, DR, STEPS = 40, 8, 100_000
P = R.sticky_action_matrices(K); Ppi = R.policy_matrix_uniform(P)
rho, _, _ = R.stationary(Ppi)
M = R.successor_measure(P, Ppi, R.GAMMA)          # [S*A, S], rows sum to 1/(1-gamma)
gt, _, _, _ = R.door_metrics({a: (M @ R.R_ROOM)[R.ROW_D[a]] for a in R.ACTIONS})

# FB models the density ratio: M[sa, u] ~= F_sa . B_u * rho[u]
Mrat = M / rho[None, :]

def sq_err(F, B, w):
    """rho-weighted squared error of the density-ratio fit, FB's own objective."""
    E = (F @ B.T) - Mrat
    return float(np.sum(w[None, :] * E ** 2) / E.shape[0])

def margin(F, B, w):
    Q = R.fb_readout(F, B, w, R.R_ROOM)
    g, _, _, _ = R.door_metrics({a: Q[R.ROW_D[a]] for a in R.ACTIONS})
    return g / gt

F_s, B_s = R.fb_fit(M, rho, DR)
print(f"true margin {gt:.3f}\n")
print(f"  {'solution':22s} {'rho-wtd sq err':>15s} {'norm margin':>12s}")
print(f"  {'SVD optimum (rho)':22s} {sq_err(F_s, B_s, rho):15.4f} {margin(F_s, B_s, rho):12.3f}")

for sd in (0, 1, 2):
    s_i, a_i, sp_i = R.generate_dataset_td(K, sd)
    emp = np.bincount(np.asarray(sp_i), minlength=R.S).astype(float); emp /= emp.sum()
    key = random.split(random.PRNGKey(20_000 * K + sd))[0]
    F, B, _ = R._train_fb_td(key, jnp.asarray(s_i), jnp.asarray(a_i), jnp.asarray(sp_i),
                             DR, STEPS, R.TRAIN_LR, R.TRAIN_TAU,
                             R.TRAIN_BATCH_T, R.TRAIN_BATCH_U, R.GAMMA)
    F, B = np.asarray(F), np.asarray(B)
    print(f"  {'TD-trained FB s'+str(sd):22s} {sq_err(F, B, emp):15.4f} {margin(F, B, emp):12.3f}")

# and the same for the SVD optimum scored under the empirical weight, for fairness
print(f"  {'SVD optimum (emp wt)':22s} {sq_err(F_s, B_s, emp):15.4f} {margin(F_s, B_s, emp):12.3f}")
