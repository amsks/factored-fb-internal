"""FB's readout has the same bug: run_trained_sweep weights its row by the ANALYTIC
stationary rho, but the states its B tower is trained against come from the dataset.
Re-read trained FB under all three weights so the comparison is like-for-like."""
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
print(f"true margin {gt:.3f}   (closed-form SVD optimum norm margin = 0.144)")

for sd in (0, 1, 2):
    s_i, a_i, sp_i = R.generate_dataset_td(K, sd)
    emp = np.bincount(np.asarray(sp_i), minlength=R.S).astype(float); emp /= emp.sum()
    key = random.split(random.PRNGKey(20_000 * K + sd))[0]
    F, B, _ = R._train_fb_td(key, jnp.asarray(s_i), jnp.asarray(a_i), jnp.asarray(sp_i),
                             DR, STEPS, R.TRAIN_LR, R.TRAIN_TAU,
                             R.TRAIN_BATCH_T, R.TRAIN_BATCH_U, R.GAMMA)
    F, B = np.asarray(F), np.asarray(B)
    out = []
    for tag, wgt in [("rho", rho), ("empirical", emp)]:
        Q = R.fb_readout(F, B, wgt, R.R_ROOM)
        g, _, _, _ = R.door_metrics({a: Q[R.ROW_D[a]] for a in R.ACTIONS})
        tv = np.mean([R.tv_half(
            np.clip((1 - R.GAMMA) * R.fb_row(F, B, R.ROW_D[a]) * wgt, 0, None) /
            np.clip((1 - R.GAMMA) * R.fb_row(F, B, R.ROW_D[a]) * wgt, 0, None).sum(), p_true[a])
            for a in (R.A_CROSS, R.A_FAIL_SHOW)])
        out.append(f"{tag:9s} margin {g/gt:.3f} TV {tv:.3f}")
    print(f"  FB TD 100k seed {sd}: " + " | ".join(out))
