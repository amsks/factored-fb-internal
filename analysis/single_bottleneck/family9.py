"""Gap-recovery table and graph figures for a policy family on the nine-state pass-through process.

Run: JAX_PLATFORMS=cpu python analysis/single_bottleneck/family9.py [--figures-only]
"""
import csv, importlib.util, json, os, pathlib, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent; REPO = HERE.parent.parent
FIG = REPO / "figures" / "nine_state"
TAIL_REWARD, GAMMA, RANKS = 0.5, 0.8, list(range(1, 9))
METHODS = {"fb": "A", "normsq": "B", "softsq": "C", "density": "D", "density_soft": "F"}


def load_sb():
    spec = importlib.util.spec_from_file_location("sb_run", HERE / "run.py"); sb = importlib.util.module_from_spec(spec)
    argv = sys.argv; sys.argv = [argv[0]]; spec.loader.exec_module(sb); sys.argv = argv
    sys.path.insert(0, str(REPO / "analysis" / "lemma_check"))
    import design_single_bottleneck as dsb
    return sb, dsb


def make(sb, dsb, name, pi):
    edges, b, A, B, T = dsb.build_passthrough(entry=2, tail=4)
    m = sb.finish(0, name, dict(entry=2, tail=4, passthrough=True), edges, pi, b, A, B, T)
    r = m["r"].copy(); r[T] = TAIL_REWARD
    q = m["m_exact"] @ r
    m.update(r=r, q_exact=q, gap=float(q[2 * b] - q[2 * b + 1]), variant=name)
    return m


def closed_rank2(sb, m):
    _, B = sb.fam.svd_fit(m["X"], 2)
    P = B @ np.linalg.pinv(B); b = m["decision"]
    dv = m["X"][2 * b] - m["X"][2 * b + 1]
    return float(dv @ P @ m["r"]) / float(dv @ m["r"])


def processes(sb, dsb):
    n = 9
    out = [make(sb, dsb, "P9", np.full((n, 2), 0.5))]
    for k in range(1, 6):
        rng = np.random.default_rng(2000 + k)
        for attempt in range(1000):
            p0 = rng.uniform(0.25, 0.75, size=n)
            m = make(sb, dsb, f"R{k}", np.stack([p0, 1 - p0], 1))
            if m["gap"] > 0.5 and closed_rank2(sb, m) < 0.5:
                m["attempts"] = attempt + 1
                out.append(m); break
    return out


def train_one(name):
    sb, dsb = load_sb()
    m = [p for p in processes(sb, dsb) if p["variant"] == name][0]
    b, n = m["decision"], m["n"]
    rows = []
    for d in RANKS:
        for meth, arm in METHODS.items():
            cfg, pred, _, _ = sb.fam.train_group(m, d, meth)
            for c_, p in zip(cfg, pred):
                P = sb.bf.readout(p, sb.fam._base_method(meth), GAMMA, n); q = P @ m["r"] / (1 - GAMMA)
                rows.append(dict(process=name, arm=arm, d=d, seed=c_["seed"], lr=c_["lr"],
                                 gap_ratio=float((q[2 * b] - q[2 * b + 1]) / m["gap"])))
        print(f"[{name}] d={d} done", flush=True)
    return rows


def p9_rows():
    c = dict(np.load(HERE / "paper_figs_61_p9_cache.npz"))
    return [dict(process="P9", arm=chr(int(a)), d=int(d), seed=-1, lr=-1, gap_ratio=float(g))
            for a, d, _, _, g in c["rows"]]


def draw(sb, procs):
    import matplotlib; matplotlib.use("Agg")
    for m in procs:
        title = f"{m['variant']}: exact gap {m['gap']:.2f}"
        sb.fam.draw_graph(m, layout=sb.sb_layout, node_style=sb.sb_node_style, stem=f"fig_family_{m['variant']}",
                          title=title, fig_dir=FIG, figsize=(3.8, 2.0), xlim=(-2.7, 2.7))


def table(rows):
    names = ["P9"] + [f"R{k}" for k in range(1, 6)]
    print("\nsmallest d with mean gap fraction >= 0.9 (TD):")
    out = {}
    for arm in "ABCDF":
        line = []
        for nm in names:
            ds = [d for d in RANKS if np.mean([r["gap_ratio"] for r in rows if r["process"] == nm and r["arm"] == arm and r["d"] == d] or [np.nan]) >= 0.9]
            line.append(min(ds) if ds else None)
        out[arm] = line
        print(arm, line)
    return out


if __name__ == "__main__":
    sb, dsb = load_sb()
    procs = processes(sb, dsb)
    json.dump([dict(name=m["variant"], gap=m["gap"], attempts=int(m.get("attempts", 1)),
                    p_first=np.round(m["pi"][:, 0], 3).tolist()) for m in procs],
              open(HERE / "family9_processes.json", "w"), indent=1)
    for m in procs:
        print(m["variant"], "gap", round(m["gap"], 3), "attempts", m.get("attempts", 1), "p_first", np.round(m["pi"][:, 0], 2))
    if "--figures-only" not in sys.argv:
        from multiprocessing import get_context
        with get_context("spawn").Pool(5) as pool:
            res = pool.map(train_one, [f"R{k}" for k in range(1, 6)])
        rows = p9_rows() + [r for rs in res for r in rs]
        with open(HERE / "family9_trained.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    rows = [dict(r, d=int(r["d"]), gap_ratio=float(r["gap_ratio"])) for r in csv.DictReader(open(HERE / "family9_trained.csv"))]
    draw(sb, procs)
    table(rows)
