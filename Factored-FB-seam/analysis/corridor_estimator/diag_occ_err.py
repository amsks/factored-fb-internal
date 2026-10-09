"""Compare each solution to M on the OCCUPANCY itself, which carries no ratio convention:
   occupancy = (F B^T) * w, with w the distribution that solution's B was fitted against
   (rho for the closed-form fit, the dataset's empirical next-state distribution for TD).
Reports squared error and total-variation to the true M, plus the door margin."""
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
print(f"true margin {gt:.3f};  M rows sum to {1/(1-R.GAMMA):.0f}\n")
print(f"  {'solution':22s} {'occ sq err':>12s} {'door-row TV':>12s} {'margin':>8s}")

def report(name, occ, Q):
    err = float(np.mean((occ - M) ** 2))
    tv = np.mean([R.tv_half(np.clip(occ[R.ROW_D[a]], 0, None) /
                            np.clip(occ[R.ROW_D[a]], 0, None).sum(), p_true[a])
                  for a in (R.A_CROSS, R.A_FAIL_SHOW)])
    g, _, _, _ = R.door_metrics({a: Q[R.ROW_D[a]] for a in R.ACTIONS})
    print(f"  {name:22s} {err:12.4f} {tv:12.3f} {g/gt:8.3f}")

F_s, B_s = R.fb_fit(M, rho, DR)
report("FB SVD optimum", (F_s @ B_s.T) * rho[None, :], R.fb_readout(F_s, B_s, rho, R.R_ROOM))

F_c, B_c, _ = R.density_fit((1 - R.GAMMA) * M, DR, seed=DR * 17 + K * 3, restarts=1)
Qc, p_c = R.density_readout(F_c, B_c, R.R_ROOM, R.GAMMA)
report("Density CE optimum", p_c / (1 - R.GAMMA), Qc)

for sd in (0, 1, 2):
    s_i, a_i, sp_i = R.generate_dataset_td(K, sd)
    emp = np.bincount(np.asarray(sp_i), minlength=R.S).astype(float); emp /= emp.sum()
    kfb, kd = random.split(random.PRNGKey(20_000 * K + sd))
    F, B, _ = R._train_fb_td(kfb, jnp.asarray(s_i), jnp.asarray(a_i), jnp.asarray(sp_i),
                             DR, STEPS, R.TRAIN_LR, R.TRAIN_TAU, R.TRAIN_BATCH_T,
                             R.TRAIN_BATCH_U, R.GAMMA)
    F, B = np.asarray(F), np.asarray(B)
    report(f"FB TD seed {sd}", (F @ B.T) * emp[None, :], R.fb_readout(F, B, emp, R.R_ROOM))

    Fd, Bd, _ = R._train_density_td(kd, jnp.asarray(s_i), jnp.asarray(a_i), jnp.asarray(sp_i),
                                    DR, STEPS, R.TRAIN_LR, R.TRAIN_TAU, R.TRAIN_BATCH_T,
                                    R.TRAIN_BATCH_U, R.GAMMA)
    _, p = R.density_readout(np.asarray(Fd), np.asarray(Bd), R.R_ROOM, R.GAMMA)
    w = p * emp[None, :]; pn = w / w.sum(-1, keepdims=True)
    Qd = (1 / (1 - R.GAMMA)) * (pn @ R.R_ROOM)
    report(f"Density TD seed {sd}", pn / (1 - R.GAMMA), Qd)
