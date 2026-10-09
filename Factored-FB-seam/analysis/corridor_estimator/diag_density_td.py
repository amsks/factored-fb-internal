"""Why is the TD-trained Density-FB map worse than its closed-form optimum?

Tests, at k=40 d=8 seed 0:
  (a) readout convention -- softmax over all states (current) vs rho-weighted, which is
      what FB's own readout does (`* rho` in run_trained_sweep);
  (b) is it just under-trained -- steps and lr sweep against the closed-form CE optimum,
      which we know represents the door correctly at this rank.
"""
import os, sys, time
sys.argv = [sys.argv[0], "--corridor14"]
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, jax, jax.numpy as jnp
from jax import random
import run as R

K, DR, SEED = 40, 8, 0
P = R.sticky_action_matrices(K)
Ppi = R.policy_matrix_uniform(P)
rho, _, _ = R.stationary(Ppi)
M = R.successor_measure(P, Ppi, R.GAMMA)
p_true = {w: (1 - R.GAMMA) * M[R.ROW_D[a]]
          for w, a in [("exit", R.A_CROSS), ("fail", R.A_FAIL_SHOW)]}

print(f"rho: door {rho[R.D]:.4f}   mean room cell "
      f"{np.mean([rho[R.CELL_TO_IDX[(r,c)]] for r in R.ROOM_ROWS for c in R.ROOM_COLS]):.4f}"
      if hasattr(R, "ROOM_ROWS") else f"rho: door {rho[R.D]:.4f}")

def report(tag, rows):
    out = []
    for w in ("exit", "fail"):
        v = np.clip(rows[w], 0, None); v = v / v.sum()
        out.append(f"{w} door {R.door_mass_fraction(v):.3f} room {R.room_mass_fraction(v):.3f} "
                   f"TV {R.tv_half(v, p_true[w]):.3f}")
    print(f"  {tag:38s} | " + " | ".join(out))

print("\n--- reference: the closed-form optima (what training should reach) ---")
F_fb, B_fb = R.fb_fit(M, rho, DR)
report("FB, SVD optimum", {w: (1 - R.GAMMA) * R.fb_row(F_fb, B_fb, R.ROW_D[a]) * rho
                           for w, a in [("exit", R.A_CROSS), ("fail", R.A_FAIL_SHOW)]})
F_c, B_c, _ = R.density_fit((1 - R.GAMMA) * M, DR, seed=1000 * SEED + DR * 17 + K * 3, restarts=1)
_, p_c = R.density_readout(F_c, B_c, R.R_ROOM, R.GAMMA)
report("Density, CE optimum", {w: p_c[R.ROW_D[a]] for w, a in
                               [("exit", R.A_CROSS), ("fail", R.A_FAIL_SHOW)]})

s_idx, a_idx, sp_idx = R.generate_dataset_td(K, SEED)
s_j, a_j, sp_j = jnp.asarray(s_idx), jnp.asarray(a_idx), jnp.asarray(sp_idx)

def train_density(steps, lr, batch_u=R.TRAIN_BATCH_U):
    key = random.split(random.PRNGKey(20_000 * K + SEED))[1]
    F, B, _ = R._train_density_td(key, s_j, a_j, sp_j, DR, steps, lr, R.TRAIN_TAU,
                                  R.TRAIN_BATCH_T, batch_u, R.GAMMA)
    return np.asarray(F), np.asarray(B)

print("\n--- (a) readout convention, current training budget ---")
F, B = train_density(R.TRAIN_STEPS, R.TRAIN_LR)
logits = F @ B.T
p_plain = np.asarray(jax.nn.softmax(jnp.asarray(logits), axis=-1))
w = p_plain * rho[None, :]; p_rho = w / w.sum(-1, keepdims=True)
report("TD, softmax over all states (current)",
       {wd: p_plain[R.ROW_D[a]] for wd, a in [("exit", R.A_CROSS), ("fail", R.A_FAIL_SHOW)]})
report("TD, rho-weighted readout",
       {wd: p_rho[R.ROW_D[a]] for wd, a in [("exit", R.A_CROSS), ("fail", R.A_FAIL_SHOW)]})

print("\n--- (b) is it under-trained? ---")
for steps, lr in [(20_000, 3e-4), (100_000, 3e-4), (300_000, 3e-4), (100_000, 1e-3)]:
    t = time.time()
    F, B = train_density(steps, lr)
    _, p_all = R.density_readout(F, B, R.R_ROOM, R.GAMMA)
    wgt = p_all * rho[None, :]; p_r = wgt / wgt.sum(-1, keepdims=True)
    report(f"TD {steps//1000}k lr {lr:g} plain",
           {wd: p_all[R.ROW_D[a]] for wd, a in [("exit", R.A_CROSS), ("fail", R.A_FAIL_SHOW)]})
    report(f"TD {steps//1000}k lr {lr:g} rho-wtd",
           {wd: p_r[R.ROW_D[a]] for wd, a in [("exit", R.A_CROSS), ("fail", R.A_FAIL_SHOW)]})
    print(f"     ({time.time()-t:.0f}s)")
