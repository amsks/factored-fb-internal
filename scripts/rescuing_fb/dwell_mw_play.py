"""Dwell battery on the Metaworld play-resample datasets (same definitions as dwell_stats.py).

Object track = achieved_goal [N,T,3] (object xyz, = obs[4:7]).
eps = the task's own success radius (0.05 push / 0.07 pick-place).
Adds the external-review statistics: M_dwell vs M_reentry mass split.
"""
import json
import numpy as np

GAMMA = 0.99
ROOT = "/home/adityamohan/git/datasets/factoredfb-metaworld-play/data"
EPS = {"mw-push": 0.05, "mw-pick-place": 0.07}
FILES = [(r, t, f"{ROOT}/{r}/metaworld/{t}.npz")
         for r in ("play-resample", "play-resample-hard")
         for t in ("mw-push", "mw-pick-place")]
rng = np.random.default_rng(0)
out = {}

for regime, task, path in FILES:
    d = np.load(path)
    c = d["achieved_goal"]          # [N,T,3]
    N, T, _ = c.shape
    eps = EPS[task]
    flat = c.reshape(-1, 3)
    centers = flat[rng.choice(len(flat), 50, replace=False)]

    runs, entries_per_ep, arrive, revisit_num, revisit_den = [], [], 0, 0, 0
    mass_act, mass_mod, m_dwell_tot, m_reentry_tot = [], [], 0.0, 0.0
    for g in centers:
        inr = (np.linalg.norm(c - g, axis=2) <= eps)   # [N,T]
        for i in range(N):
            row = inr[i]
            if not row.any():
                continue
            arrive += 1
            # contiguous runs
            db = np.diff(row.astype(np.int8))
            starts = list(np.where(db == 1)[0] + 1)
            ends = list(np.where(db == -1)[0] + 1)
            if row[0]:
                starts = [0] + starts
            if row[-1]:
                ends = ends + [T]
            rl = [e - s for s, e in zip(starts, ends)]
            runs += rl
            entries_per_ep.append(len(rl))
            if len(rl) >= 1 and ends[0] < T:
                revisit_den += 1
                if len(rl) > 1:
                    revisit_num += 1
            t_idx = np.arange(T)
            disc = GAMMA ** t_idx
            mass = float((disc * row).sum())
            mass_act.append(mass)
            # dwell vs re-entry mass split (first sojourn vs later)
            first_mask = np.zeros(T, bool)
            first_mask[starts[0]:ends[0]] = True
            m_dwell_tot += float((disc * (row & first_mask)).sum())
            m_reentry_tot += float((disc * (row & ~first_mask)).sum())
            # model: gamma^t_first x (1-gamma^Lbar)/(1-gamma) with this center's mean run
        # per-center model over episodes
    # recompute model fit per center (arrival prob x mean discount at arrival x dwell const)
    fit_a, fit_m = [], []
    for g in centers:
        inr = (np.linalg.norm(c - g, axis=2) <= eps)
        pa, tf, rls, masses = 0, [], [], []
        for i in range(N):
            row = inr[i]
            t_idx = np.arange(T)
            masses.append(float((GAMMA ** t_idx * row).sum()))
            if row.any():
                pa += 1
                t0 = int(np.argmax(row))
                tf.append(GAMMA ** t0)
                db = np.diff(row.astype(np.int8))
                s = [0] + list(np.where(db == 1)[0] + 1) if row[0] else list(np.where(db == 1)[0] + 1)
                e = list(np.where(db == -1)[0] + 1) + ([T] if row[-1] else [])
                rls += [b - a for a, b in zip(s, e)]
        if pa == 0:
            continue
        L = float(np.mean(rls))
        model = (pa / N) * float(np.mean(tf)) * (1 - GAMMA ** L) / (1 - GAMMA)
        fit_a.append(float(np.mean(masses)))
        fit_m.append(model)
    fa, fm = np.array(fit_a), np.array(fit_m)
    ok = (fa > 0) & (fm > 0)
    logr = np.corrcoef(np.log(fa[ok]), np.log(fm[ok]))[0, 1] if ok.sum() > 2 else float("nan")
    runs = np.array(runs)
    out[f"{regime}/{task}"] = dict(
        eps=eps, episodes=N, T=T,
        p_arrive_ep=round(arrive / (N * 50), 3),
        run_mean=round(float(runs.mean()), 1),
        run_med=float(np.median(runs)),
        run_p90=float(np.percentile(runs, 90)),
        entries_per_visiting_ep=round(float(np.mean(entries_per_ep)), 2),
        revisit_prob=round(revisit_num / max(revisit_den, 1), 3),
        dwell_mass_frac=round(m_dwell_tot / max(m_dwell_tot + m_reentry_tot, 1e-9), 3),
        ratio_med=round(float(np.median(fa[ok] / fm[ok])), 3),
        log_pearson=round(float(logr), 4),
    )
    print(f"{regime}/{task}: {out[f'{regime}/{task}']}")

json.dump(out, open("/tmp/claude-1001/-home-adityamohan-git-Factored-FB/1ba5ea79-da0c-4583-a89e-0a11cfad5238/scratchpad/dwell/dwell_mw_play.json", "w"), indent=1)
