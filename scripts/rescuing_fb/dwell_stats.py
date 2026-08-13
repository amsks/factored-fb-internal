#!/usr/bin/env python
"""Dwell/revisit statistics for eps-ball regions over object positions, across
offline manipulation datasets.

Conventions copied from scripts/rescuing_fb/smb_common.py /
eval_zeroshot_rewards.py:
  - OGBench cube/scene: object xyz = obs[19:22] (+9 per extra cube), OBS units
    (1 unit = 10 cm, obs = (m - CEN)*10), eps = EPS_OBS = 0.4 obs units = 4 cm.
  - gamma = 0.99.
  - regions: eps-balls with centers sampled from dataset object positions
    (same convention as the renewal_head / T3 MC region stats).
For the HF Factor-FB-Offline datasets (meters), eps = 0.04 m = the same 4 cm.

Per (dataset, ~50 centers): contiguous in-region run lengths (mean/median/p90),
entries per visiting episode, longest-run fraction of in-region time, revisit
probability after an exit, and mean discounted occupancy mass per episode vs
the arrival x dwell-constant model
    mass_model = P(arrive) * E[gamma^t_first | arrive] * (1-gamma^Lbar)/(1-gamma).

Outputs dwell_stats.json next to this script.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OGB = Path.home() / ".ogbench" / "data"
HF = HERE / "hf" / "data"

GAMMA = 0.99
EPS_OGB = 0.4     # obs units = 4 cm (EPS_OBS in eval_zeroshot_rewards.py)
EPS_M = 0.04      # meters, same physical 4 cm, for the HF datasets
N_CENTERS = 50
SEED = 0

# name -> (loader kwargs)
DATASETS = {
    "cube-single-play-v0":    dict(kind="ogbench", blocks=[(19, 22)], eps=EPS_OGB),
    "cube-double-play-v0":    dict(kind="ogbench", blocks=[(19, 22), (28, 31)], eps=EPS_OGB),
    "cube-triple-play-v0":    dict(kind="ogbench", blocks=[(19, 22), (28, 31), (37, 40)], eps=EPS_OGB),
    "cube-quadruple-play-v0": dict(kind="ogbench", blocks=[(19, 22), (28, 31), (37, 40), (46, 49)], eps=EPS_OGB),
    "scene-play-v0":          dict(kind="ogbench", blocks=[(19, 22)], eps=EPS_OGB),
    "hf-ms-pick-cube":        dict(kind="hf", path=HF / "maniskill" / "ms-pick-cube.npz",
                                   slc=(29, 32), eps=EPS_M),
    "hf-mw-bin-picking":      dict(kind="hf", path=HF / "metaworld" / "mw-bin-picking.npz",
                                   slc=(4, 7), eps=EPS_M),
}


def load_tracks(name: str, spec: dict):
    """-> list of [n_ep, T, 3] float32 arrays (one per tracked object block)."""
    if spec["kind"] == "ogbench":
        with np.load(OGB / f"{name}.npz") as f:
            obs = f["observations"]
            term = f["terminals"]
            ends = np.nonzero(term)[0]
            starts = np.concatenate([[0], ends[:-1] + 1])
            lens = ends - starts + 1
            assert (lens == lens[0]).all(), "variable episode lengths"
            T = int(lens[0])
            tracks = []
            for a, b in spec["blocks"]:
                x = np.ascontiguousarray(obs[:, a:b], dtype=np.float32)
                tracks.append(x.reshape(-1, T, 3))
            del obs
        return tracks
    # HF unified format: observation [N, T, obs_dim]
    with np.load(spec["path"]) as f:
        obs = f["observation"]
        a, b = spec["slc"]
        return [np.ascontiguousarray(obs[:, :, a:b], dtype=np.float32)]


def runs_of(row: np.ndarray):
    """Bool [T] -> list of (start, length) of contiguous True runs."""
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    return list(zip(starts, ends - starts))


def analyze(name: str, spec: dict) -> dict:
    rng = np.random.default_rng(SEED)
    tracks = load_tracks(name, spec)
    eps = spec["eps"]
    n_ep, T = tracks[0].shape[:2]
    n_blocks = len(tracks)

    # sample centers from dataset object positions: (block, episode, t) triples
    centers = []
    for _ in range(N_CENTERS):
        blk = int(rng.integers(n_blocks))
        e = int(rng.integers(n_ep))
        t = int(rng.integers(T))
        centers.append((blk, tracks[blk][e, t].astype(np.float64)))

    disc = GAMMA ** np.arange(T)
    run_lengths = []
    entries_per_visit = []      # runs per (center, episode) with >= 1 entry
    longest_frac = []           # max run / total in-region steps, per visiting ep
    total_exits = 0
    total_reentries = 0
    exit_eps = 0                # (center, episode) pairs with >= 1 exit
    exit_eps_reentered = 0
    per_center = []

    for blk, c in centers:
        pos = tracks[blk]                            # [n_ep, T, 3]
        d2 = ((pos - c.astype(np.float32)) ** 2).sum(-1)
        member = d2 <= eps * eps                     # [n_ep, T] bool
        ep_any = member.any(axis=1)
        p_arrive = float(ep_any.mean())
        mass = (member * disc).sum(axis=1)           # discounted mass per ep
        actual_mass = float(mass.mean())

        c_runs, c_tfirst = [], []
        for e in np.nonzero(ep_any)[0]:
            rr = runs_of(member[e])
            lens = [l for _, l in rr]
            c_runs.extend(lens)
            entries_per_visit.append(len(rr))
            longest_frac.append(max(lens) / sum(lens))
            c_tfirst.append(rr[0][0])
            exits = sum(1 for s, l in rr if s + l < T)   # left region before ep end
            total_exits += exits
            total_reentries += len(rr) - 1
            if exits:
                exit_eps += 1
                exit_eps_reentered += int(len(rr) > 1)
        run_lengths.extend(c_runs)

        lbar = float(np.mean(c_runs)) if c_runs else 0.0
        dwell_factor = (1 - GAMMA ** lbar) / (1 - GAMMA) if lbar else 0.0
        e_disc_arr = float(np.mean(GAMMA ** np.array(c_tfirst))) if c_tfirst else 0.0
        per_center.append(dict(
            block=blk, center=[round(float(v), 4) for v in c],
            p_arrive=p_arrive, mean_run=lbar, e_disc_arrival=e_disc_arr,
            actual_mass=actual_mass,
            model_mass=p_arrive * e_disc_arr * dwell_factor,
            model_mass_nodisc=p_arrive * dwell_factor,
        ))

    rl = np.array(run_lengths, float)
    epv = np.array(entries_per_visit, float)
    am = np.array([c["actual_mass"] for c in per_center])
    mm = np.array([c["model_mass"] for c in per_center])
    ok = (am > 0) & (mm > 0)
    ratio = am[ok] / mm[ok]
    # correlation across centers between actual and modeled mass
    if ok.sum() > 2:
        pear = float(np.corrcoef(am[ok], mm[ok])[0, 1])
        lg = float(np.corrcoef(np.log(am[ok]), np.log(mm[ok]))[0, 1])
    else:
        pear = lg = float("nan")

    out = dict(
        n_episodes=n_ep, ep_len=T, n_blocks=n_blocks, eps=eps,
        n_centers=N_CENTERS, n_centers_visited=int(ok.sum()),
        p_arrive_mean=float(np.mean([c["p_arrive"] for c in per_center])),
        n_runs=int(len(rl)),
        run_mean=float(rl.mean()) if len(rl) else None,
        run_median=float(np.median(rl)) if len(rl) else None,
        run_p90=float(np.percentile(rl, 90)) if len(rl) else None,
        run_max=float(rl.max()) if len(rl) else None,
        entries_per_visiting_ep_mean=float(epv.mean()) if len(epv) else None,
        longest_run_frac_mean=float(np.mean(longest_frac)) if longest_frac else None,
        revisit_prob_per_exit=(total_reentries / total_exits) if total_exits else None,
        revisit_prob_per_exit_episode=(exit_eps_reentered / exit_eps) if exit_eps else None,
        mean_actual_mass=float(am.mean()),
        mean_model_mass=float(mm.mean()),
        mass_ratio_median=float(np.median(ratio)) if len(ratio) else None,
        mass_ratio_iqr=[float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))] if len(ratio) else None,
        mass_model_pearson=pear, mass_model_log_pearson=lg,
        per_center=per_center,
    )
    return out


def main():
    results = {}
    for name, spec in DATASETS.items():
        src = spec.get("path", OGB / f"{name}.npz")
        if not Path(src).exists():
            print(f"[skip] {name}: {src} missing")
            continue
        print(f"[run ] {name}")
        results[name] = analyze(name, spec)
        r = results[name]
        print(f"       ep {r['n_episodes']}x{r['ep_len']}  runs {r['n_runs']}  "
              f"run mean/med/p90 {r['run_mean']:.1f}/{r['run_median']:.0f}/{r['run_p90']:.0f}  "
              f"entries/visit {r['entries_per_visiting_ep_mean']:.2f}  "
              f"revisit {r['revisit_prob_per_exit_episode']}  "
              f"mass ratio {r['mass_ratio_median']}")
    with open(HERE / "dwell_stats.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", HERE / "dwell_stats.json")


if __name__ == "__main__":
    main()
