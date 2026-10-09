"""Disentangle the two changes I made at once: training budget vs readout convention.
Trains density once per budget and reads the SAME towers out both ways."""
import os, sys, time
sys.argv = [sys.argv[0], "--corridor14"]
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, jax, jax.numpy as jnp
from jax import random
import run as R

K, DR = 40, 8
P = R.sticky_action_matrices(K); Ppi = R.policy_matrix_uniform(P)
rho, _, _ = R.stationary(Ppi)
M = R.successor_measure(P, Ppi, R.GAMMA)
gt, _, _, _ = R.door_metrics({a: (M @ R.R_ROOM)[R.ROW_D[a]] for a in R.ACTIONS})
print(f"true margin {gt:.3f}\n")

# where does the CLOSED-FORM CE optimum land under each readout? (known-good reference)
F_c, B_c, _ = R.density_fit((1 - R.GAMMA) * M, DR, seed=DR * 17 + K * 3, restarts=1)
_, p_c = R.density_readout(F_c, B_c, R.R_ROOM, R.GAMMA)
w = p_c * rho[None, :]; p_cr = w / w.sum(-1, keepdims=True)
for tag, p in [("plain", p_c), ("rho-wtd", p_cr)]:
    Q = (1 / (1 - R.GAMMA)) * (p @ R.R_ROOM)
    g, _, _, _ = R.door_metrics({a: Q[R.ROW_D[a]] for a in R.ACTIONS})
    print(f"  closed-form CE optimum, {tag:8s} margin {g:6.3f}  norm {g/gt:.3f}")

print()
for steps in (20_000, 100_000):
    for sd in (0, 1, 2):
        s_i, a_i, sp_i = R.generate_dataset_td(K, sd)
        key = random.split(random.PRNGKey(20_000 * K + sd))[1]
        F, B, _ = R._train_density_td(key, jnp.asarray(s_i), jnp.asarray(a_i), jnp.asarray(sp_i),
                                      DR, steps, R.TRAIN_LR, R.TRAIN_TAU,
                                      R.TRAIN_BATCH_T, R.TRAIN_BATCH_U, R.GAMMA)
        _, p = R.density_readout(np.asarray(F), np.asarray(B), R.R_ROOM, R.GAMMA)
        w = p * rho[None, :]; pr = w / w.sum(-1, keepdims=True)
        out = []
        for tag, pp in [("plain", p), ("rho", pr)]:
            Q = (1 / (1 - R.GAMMA)) * (pp @ R.R_ROOM)
            g, _, _, _ = R.door_metrics({a: Q[R.ROW_D[a]] for a in R.ACTIONS})
            tv = np.mean([R.tv_half(np.clip(pp[R.ROW_D[a]], 0, None) /
                                    np.clip(pp[R.ROW_D[a]], 0, None).sum(),
                                    (1 - R.GAMMA) * M[R.ROW_D[a]])
                          for a in (R.A_CROSS, R.A_FAIL_SHOW)])
            out.append(f"{tag:5s} margin {g/gt:.3f}  TV {tv:.3f}")
        print(f"  density TD {steps//1000:>3}k seed {sd}: " + " | ".join(out))
