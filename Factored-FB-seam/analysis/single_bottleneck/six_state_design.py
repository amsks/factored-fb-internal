"""Can a SIX-state family carry the whole 6.1 story? For discount in {0.8, 0.9, 0.95} and six-state draws
(entry 1, bottleneck, clusters 2+2, mutual leak; uniform and random policies): closed-form FB fraction at ranks
2 and 3, and TD-trained FB / Density-FB fractions (10 runs) at ranks 2 and 3.
    MPLCONFIGDIR=/tmp/mpl JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python analysis/single_bottleneck/six_state_design.py
"""
import importlib.util, os, pathlib, sys, statistics as st
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sb_run", HERE / "run.py"); sb = importlib.util.module_from_spec(spec)
sys.argv, _a = [sys.argv[0]], sys.argv; spec.loader.exec_module(sb); sys.argv = _a
fam, dsb, bf = sb.fam, sb.dsb, sb.bf

def process(gamma, pi_seed=None):
    edges, b, A, B, T = dsb.build(entry=1, cluster=2, tail=0, mutual=True)
    n = len(edges)
    pi = np.full((n, 2), 0.5) if pi_seed is None else np.stack([np.random.default_rng(pi_seed).uniform(0.25, 0.75, n)] * 2, 1)
    if pi_seed is not None: pi[:, 1] = 1 - pi[:, 0]
    for mod in (fam, dsb, sb, bf):
        if hasattr(mod, "GAMMA"): mod.GAMMA = gamma
    m = sb.finish(900 + (pi_seed or 0), f"g{gamma}", dict(entry=1, cluster=2, tail=0, mutual=True), edges, pi, b, A, B, T)
    return m

def closed_fb(m, d):
    X = m["n"] * m["m_exact"]; _, _, Vt = np.linalg.svd(X, full_matrices=False); P = Vt[:d].T @ Vt[:d]
    dv = X[2 * m["decision"]] - X[2 * m["decision"] + 1]; return float((dv @ P @ m["r"]) / (dv @ m["r"]))

print(f"{'gamma':>5s} {'policy':>7s} {'gap':>5s} | closed FB d2 d3 | trained FB d2 d3 | trained DFB d2 d3")
for gamma in (0.8, 0.9, 0.95):
    for pol in (None, 1, 2):
        m = process(gamma, pol)
        out = [closed_fb(m, 2), closed_fb(m, 3)]
        tr = {}
        for d in (2, 3):
            for meth in ("fb", "density"):
                cfg, pred, F, B = fam.train_group(m, d, meth, list(fam.SEEDS), list(fam.RATES), steps=fam.TRAIN_STEPS)
                fr = []
                for i in range(len(cfg)):
                    P = bf.readout(pred[i], fam._base_method(meth), gamma, m["n"])
                    fr.append(fam.gap_stats(m, P @ m["r"] / (1 - gamma))["gap_ratio"])
                tr[(meth, d)] = st.mean(fr)
        print(f"{gamma:5.2f} {('uniform' if pol is None else f'rand{pol}'):>7s} {m['gap']:5.2f} | {out[0]:6.2f} {out[1]:5.2f} | "
              f"{tr[('fb',2)]:6.2f} {tr[('fb',3)]:5.2f} | {tr[('density',2)]:6.2f} {tr[('density',3)]:5.2f}", flush=True)
