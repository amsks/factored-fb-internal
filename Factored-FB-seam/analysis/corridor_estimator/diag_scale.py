"""Is TD-FB's solution just mis-SCALED? Both the maps and the normalized margin are
scale-invariant; the squared error is not. Check the row sums (a successor measure row
must integrate to 1/(1-gamma)) and the squared error after fitting one global scale."""
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
Mrat = M / rho[None, :]
target_rowsum = 1.0 / (1.0 - R.GAMMA)
print(f"a successor-measure row must sum to {target_rowsum:.1f}\n")

def stats(name, F, B, w):
    X = F @ B.T                       # models M/rho
    occ = X * w[None, :]              # implied occupancy
    rs = occ.sum(1)
    E = X - Mrat
    raw = float(np.sum(w[None, :] * E ** 2) / E.shape[0])
    # best single global scale c minimising ||c X - Mrat||^2_w
    num = float(np.sum(w[None, :] * X * Mrat)); den = float(np.sum(w[None, :] * X * X))
    c = num / den
    Ec = c * X - Mrat
    scaled = float(np.sum(w[None, :] * Ec ** 2) / Ec.shape[0])
    print(f"  {name:24s} rowsum mean {rs.mean():10.2f}  min {rs.min():9.2f}  max {rs.max():10.2f} "
          f"| sq err {raw:12.2f} -> after best scale c={c:.4g}: {scaled:10.2f}")

F_s, B_s = R.fb_fit(M, rho, DR)
stats("SVD optimum", F_s, B_s, rho)
for sd in (0, 1, 2):
    s_i, a_i, sp_i = R.generate_dataset_td(K, sd)
    emp = np.bincount(np.asarray(sp_i), minlength=R.S).astype(float); emp /= emp.sum()
    key = random.split(random.PRNGKey(20_000 * K + sd))[0]
    F, B, _ = R._train_fb_td(key, jnp.asarray(s_i), jnp.asarray(a_i), jnp.asarray(sp_i),
                             DR, STEPS, R.TRAIN_LR, R.TRAIN_TAU,
                             R.TRAIN_BATCH_T, R.TRAIN_BATCH_U, R.GAMMA)
    stats(f"TD-trained FB seed {sd}", np.asarray(F), np.asarray(B), emp)
