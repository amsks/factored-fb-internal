"""Is the door advantage TD, or is it the WEIGHT? Refit the closed-form SVD under the
empirical dataset distribution and compare like for like. No training involved."""
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
print(f"true margin {gt:.3f}\n")
print(f"  {'solution':30s} {'occ sq err':>11s} {'door TV':>9s} {'margin':>8s}")

def report(name, occ, Q):
    err = float(np.mean((occ - M) ** 2))
    tv = np.mean([R.tv_half(np.clip(occ[R.ROW_D[a]], 0, None) /
                            np.clip(occ[R.ROW_D[a]], 0, None).sum(), p_true[a])
                  for a in (R.A_CROSS, R.A_FAIL_SHOW)])
    g, _, _, _ = R.door_metrics({a: Q[R.ROW_D[a]] for a in R.ACTIONS})
    print(f"  {name:30s} {err:11.4f} {tv:9.3f} {g/gt:8.3f}")

s_i, a_i, sp_i = R.generate_dataset_td(K, 0)
emp = np.bincount(np.asarray(sp_i), minlength=R.S).astype(float); emp /= emp.sum()
print(f"  (|rho - emp|_1 = {np.abs(rho-emp).sum():.3f}; door rho {rho[R.D]:.4f} emp {emp[R.D]:.4f}; "
      f"corridor mass rho {sum(rho[R.CELL_TO_IDX[(R.CORRIDOR_ROW,c)]] for c in range(1,1+R.CORRIDOR_LEN)):.3f} "
      f"emp {sum(emp[R.CELL_TO_IDX[(R.CORRIDOR_ROW,c)]] for c in range(1,1+R.CORRIDOR_LEN)):.3f})\n")

for wname, w in [("rho", rho), ("empirical", emp)]:
    F, B = R.fb_fit(M, w, DR)
    report(f"FB SVD optimum, fit under {wname}", (F @ B.T) * w[None, :],
           R.fb_readout(F, B, w, R.R_ROOM))

kfb, _ = random.split(random.PRNGKey(20_000 * K + 0))
F, B, _ = R._train_fb_td(kfb, jnp.asarray(s_i), jnp.asarray(a_i), jnp.asarray(sp_i),
                         DR, STEPS, R.TRAIN_LR, R.TRAIN_TAU, R.TRAIN_BATCH_T,
                         R.TRAIN_BATCH_U, R.GAMMA)
F, B = np.asarray(F), np.asarray(B)
report("FB TD seed 0 (trained on emp)", (F @ B.T) * emp[None, :],
       R.fb_readout(F, B, emp, R.R_ROOM))
