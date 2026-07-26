#!/usr/bin/env python
"""scripts/rescuing_fb/smb_tasks.py -- Spec 11 item 5.1: build the FROZEN SMB
task artifact `analysis/rescuing-fb/smb_cube_single_v1.json` from the DATASET
ONLY (no checkpoint is ever loaded here; the artifact must exist before any
checkpoint sees a task). The JSON is the benchmark: committed once, never
regenerated silently -- `--check` verifies byte-identical regeneration.

Contents (spec 11 sections 1-2):
  t3_occhoice   5 (stable, transient) region pairs. Stable = on-table cube
                positions with HIGH MC post-arrival discounted occupancy
                (settled mode); transient = elevated positions (table z + 0.8
                obs units = +8 cm) with dataset support and LOW post-arrival
                occupancy (pass-through). Weights: w_S = 1, w_T =
                sqrt(reach-proxy-ratio x occupancy-ratio) clipped to [2, 8]
                (reach proxy gamma^median-first-arrival-from-episode-start);
                w_T defaults to 4 when the interval is degenerate. Validation
                R_S/R_T >= 5 per pair; failures are RECORDED, never fudged.
  t4_union3     3 tasks x 3 eps-balls at well-separated on-table dataset
                positions (pairwise >= 3 eps).
  t5_corridor   per native task: avoid-ball center = highest-dataset-density
                cube position within 0.15 obs units of the straight start->goal
                segment (start = median episode-start cube position), radius
                0.5 obs units; lambda in {1, 5}.
  wingb_battery 64 eps-balls at stratified dataset positions (>= 8 elevated),
                32 random axis-aligned boxes (sides 0.4-1.2 obs units), 8
                unions-of-2-balls (disjoint, >= 3 eps apart). Box/union queries
                additionally freeze `rep_rows` -- dataset row indices inside
                the region -- so Wing B's kernel-average surrogate is
                deterministic.

All geometry in OBS units (obs[19:22], 1 unit = 10 cm); eps = 0.4; gamma = 0.99.
Everything is driven by ONE seeded rng (seed 0) so regeneration is bit-exact.

Run (CPU; ogbench needed for the dataset + native goal xyz, jax NOT needed):
  JAX_PLATFORMS=cpu MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python \\
    scripts/rescuing_fb/smb_tasks.py --out analysis/rescuing-fb/smb_cube_single_v1.json
  # verify the frozen artifact regenerates byte-identically:
  ... smb_tasks.py --check --out analysis/rescuing-fb/smb_cube_single_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from smb_common import (  # noqa: E402
    AVOID_R, ELEV_DZ, EPS_OBS, EVAL_TASKS, GAMMA, SPEC, TASK_JSON_VERSION,
    cube_m_to_ou, mc_region_stats, split_episodes, to_native,
)

CUBE_OBS = slice(19, 22)


# ======================= pure geometry / selection helpers ========================

def table_z_mode(cube_z, bin_width: float = 0.02) -> float:
    """Typical on-table cube z = histogram mode of the dataset cube z (the cube
    rests on the table for the vast majority of play steps)."""
    z = np.asarray(cube_z, np.float64)
    lo, hi = float(z.min()), float(z.max())
    nbins = max(int(np.ceil((hi - lo) / bin_width)), 1)
    hist, edges = np.histogram(z, bins=nbins, range=(lo, lo + nbins * bin_width))
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


def fps_select(points, k: int, rng) -> np.ndarray:
    """Deterministic farthest-point subset (the stratification device: spread
    across the workspace without density bias). Start index drawn from rng;
    the rest is argmax (ties -> lowest index)."""
    pts = np.asarray(points, np.float64)
    n = len(pts)
    k = min(int(k), n)
    if k <= 0:
        return np.zeros(0, np.int64)
    sel = [int(rng.integers(n))]
    dmin = np.linalg.norm(pts - pts[sel[0]], axis=-1)
    for _ in range(k - 1):
        j = int(np.argmax(dmin))
        sel.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(pts - pts[j], axis=-1))
    return np.asarray(sel, np.int64)


def point_seg_dist(p, a, b) -> np.ndarray:
    """Distance from points p [N,3] to the segment a->b."""
    p = np.asarray(p, np.float64)
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return np.linalg.norm(p - a, axis=-1)
    t = np.clip((p - a) @ ab / denom, 0.0, 1.0)
    proj = a[None] + t[:, None] * ab[None]
    return np.linalg.norm(p - proj, axis=-1)


def support_counts(cube, centers, eps: float = EPS_OBS, g_chunk: int = 32,
                   t_chunk: int = 100_000) -> np.ndarray:
    """Number of `cube` rows within eps of each center (chunked both ways)."""
    cube = np.asarray(cube, np.float64)
    centers = np.asarray(centers, np.float64)
    out = np.zeros(len(centers), np.int64)
    for i0 in range(0, len(centers), g_chunk):
        gg = centers[i0:i0 + g_chunk]
        for t0 in range(0, len(cube), t_chunk):
            c = cube[t0:t0 + t_chunk]
            d2 = ((gg[:, None, :] - c[None, :, :]) ** 2).sum(-1)
            out[i0:i0 + len(gg)] += (d2 <= eps * eps).sum(1)
    return out


# ======================= T3: occupancy-choice pairs ================================

def select_t3_pairs(cube, spans, table_z, rng, gamma: float = GAMMA,
                    eps: float = EPS_OBS, n_pairs: int = 5, n_cand: int = 160,
                    min_support: int = 30, min_entries: int = 5,
                    elev_dz: float = ELEV_DZ, on_table_tol: float = 0.1,
                    elev_band: float = 0.25, w_lo: float = 2.0, w_hi: float = 8.0,
                    gap_req: float = 5.0, pool_n: int = 4000):
    """The R-discriminator regions (spec 11 T3). Stable candidates = FPS-spread
    on-table dataset positions ranked by MC post-arrival occupancy (descending);
    transient candidates = dataset positions near table_z + elev_dz with the z
    pinned to exactly table_z + elev_dz, ranked ascending. Weight validation
    per pair: w_T strictly between the reaching ratio gamma^dT_S/gamma^dT_T and
    the occupancy ratio R_S/R_T (geometric mean, clipped to [w_lo, w_hi]);
    R-gap >= gap_req or validation = 'failed' (stats recorded either way)."""
    cube = np.asarray(cube, np.float64)
    ontab = np.nonzero(np.abs(cube[:, 2] - table_z) <= on_table_tol)[0]
    elev = np.nonzero(np.abs(cube[:, 2] - (table_z + elev_dz)) <= elev_band)[0]
    meta = {"n_on_table_rows": int(len(ontab)), "n_elevated_rows": int(len(elev)),
            "n_cand": int(n_cand), "min_support": int(min_support),
            "min_entries": int(min_entries), "gap_req": float(gap_req)}
    assert len(ontab) > 0, "no on-table rows found (table_z estimate broken?)"
    assert len(elev) > 0, f"no dataset rows near table_z + {elev_dz}"

    # Candidates = uniform dataset rows (density-weighted positions: settled
    # modes dominate the on-table rows, which is exactly what T3 stable wants);
    # spatial SPREAD is enforced later by FPS over the occupancy-ranked head.
    st_rows = np.sort(rng.choice(ontab, size=min(n_cand, len(ontab)), replace=False))
    st_centers = cube[st_rows]
    tr_rows = np.sort(rng.choice(elev, size=min(n_cand, len(elev)), replace=False))
    tr_centers = cube[tr_rows].copy()
    tr_centers[:, 2] = table_z + elev_dz          # pinned elevated height

    st = mc_region_stats(cube, spans, st_centers, gamma, eps)
    tr = mc_region_stats(cube, spans, tr_centers, gamma, eps)
    st_ok = np.nonzero((st["support"] >= min_support)
                       & (st["n_entries"] >= min_entries)
                       & np.isfinite(st["occ_mean"]))[0]
    tr_ok = np.nonzero((tr["support"] >= min_support)
                       & (tr["n_entries"] >= min_entries)
                       & np.isfinite(tr["occ_mean"]))[0]
    meta["n_stable_valid"] = int(len(st_ok))
    meta["n_transient_valid"] = int(len(tr_ok))
    assert len(st_ok) >= 1 and len(tr_ok) >= 1, \
        f"too few valid T3 candidates (stable {len(st_ok)}, transient {len(tr_ok)})"

    # stable: highest occupancy first, FPS over the head for spread
    order_s = st_ok[np.argsort(-st["occ_mean"][st_ok], kind="stable")]
    head_s = order_s[:max(4 * n_pairs, n_pairs)]
    sel_s = head_s[fps_select(st_centers[head_s], n_pairs, rng)]
    # transient: lowest occupancy first
    order_t = tr_ok[np.argsort(tr["occ_mean"][tr_ok], kind="stable")]
    head_t = order_t[:max(4 * n_pairs, n_pairs)]
    sel_t = head_t[fps_select(tr_centers[head_t], n_pairs, rng)]
    # deterministic pairing: both sides sorted by x
    sel_s = sel_s[np.argsort(st_centers[sel_s][:, 0], kind="stable")]
    sel_t = sel_t[np.argsort(tr_centers[sel_t][:, 0], kind="stable")]

    def _side(stats, centers, rows, i, gamma):
        fm = stats["first_med"][i]
        return {
            "center": centers[i].tolist(), "row": int(rows[i]),
            "R_mc": float(stats["occ_mean"][i]),
            "n_entries": int(stats["n_entries"][i]),
            "first_arrival_median": None if not np.isfinite(fm) else float(fm),
            "reach_proxy": None if not np.isfinite(fm) else float(gamma ** fm),
            "hit_episode_frac": float(stats["hit_frac"][i]),
            "support": int(stats["support"][i]),
        }

    pairs = []
    for p in range(min(n_pairs, len(sel_s), len(sel_t))):
        si, ti = int(sel_s[p]), int(sel_t[p])
        s_rec = _side(st, st_centers, st_rows, si, gamma)
        t_rec = _side(tr, tr_centers, tr_rows, ti, gamma)
        R_s, R_t = s_rec["R_mc"], t_rec["R_mc"]
        occ_ratio = R_s / max(R_t, 1e-9)
        reach_s, reach_t = s_rec["reach_proxy"], t_rec["reach_proxy"]
        degenerate = (reach_s is None or reach_t is None or reach_t <= 0)
        if not degenerate:
            lo_b = reach_s / reach_t              # w_T must EXCEED to flip reaching
            hi_b = occ_ratio                      # w_T must stay BELOW to keep occ->S
            degenerate = not (np.isfinite(lo_b) and np.isfinite(hi_b)
                              and 0 < lo_b < hi_b)
        else:
            lo_b = hi_b = None
        if degenerate:
            w_raw, w_t = None, 4.0                # spec default when degenerate
        else:
            w_raw = float(np.sqrt(lo_b * hi_b))
            w_t = float(np.clip(w_raw, w_lo, w_hi))
        validation = "ok" if occ_ratio >= gap_req else "failed"
        pairs.append({
            "name": f"pair{p}", "stable": s_rec, "transient": t_rec,
            "w_S": 1.0, "w_T": w_t, "w_T_raw": w_raw,
            "w_interval": None if degenerate else [float(lo_b), float(hi_b)],
            "w_degenerate": bool(degenerate),
            "R_ratio": float(occ_ratio),
            "validation": validation,
            # occupancy-optimal region under the frozen weights (the 'correct'
            # choice the tier scores against):
            "correct_region": "stable" if R_s * 1.0 >= R_t * w_t else "transient",
        })
    return pairs, meta


# ======================= T4: union-of-3-balls tasks ================================

def select_union3(cube, table_z, rng, eps: float = EPS_OBS, n_tasks: int = 3,
                  n_balls: int = 3, min_sep: float = 3 * EPS_OBS,
                  min_support: int = 50, pool_n: int = 2000,
                  on_table_tol: float = 0.1):
    """3 tasks x 3 well-separated on-table eps-balls (pairwise >= min_sep):
    FPS-order the supported pool, then greedily assemble triples."""
    cube = np.asarray(cube, np.float64)
    ontab = np.nonzero(np.abs(cube[:, 2] - table_z) <= on_table_tol)[0]
    pool = ontab[rng.choice(len(ontab), size=min(pool_n, len(ontab)), replace=False)]
    sup = support_counts(cube, cube[pool], eps)
    pool = pool[sup >= min_support]
    assert len(pool) >= n_tasks * n_balls, \
        f"too few supported on-table positions for union3 ({len(pool)})"
    order = pool[fps_select(cube[pool], len(pool), rng)]
    tasks, current = [], []
    for row in order:
        c = cube[row]
        if all(np.linalg.norm(c - cube[r]) >= min_sep for r in current):
            current.append(int(row))
        if len(current) == n_balls:
            tasks.append(current)
            current = []
        if len(tasks) == n_tasks:
            break
    assert len(tasks) == n_tasks, \
        f"could not assemble {n_tasks} union3 tasks with sep >= {min_sep}"
    out = []
    for k, rows in enumerate(tasks):
        centers = [cube[r].tolist() for r in rows]
        dists = [float(np.linalg.norm(np.asarray(centers[i]) - np.asarray(centers[j])))
                 for i in range(n_balls) for j in range(i + 1, n_balls)]
        out.append({"name": f"union{k}", "centers": centers, "rows": rows,
                    "eps": eps, "pairwise_dist_min": min(dists)})
    return out


# ======================= T5: corridor avoid-ball ===================================

def select_corridor(cube, start_ou, goal_ou, rng, halfwidth: float = 0.15,
                    avoid_r: float = AVOID_R, density_eps: float = EPS_OBS,
                    max_cand: int = 3000, samp_n: int = 200_000):
    """Avoid-ball center = the highest-dataset-density cube position within
    `halfwidth` of the straight start->goal segment. Positions whose avoid ball
    would swallow the goal ball (dist < avoid_r + eps) or the start point
    (dist < avoid_r) are excluded -- an unsatisfiable task measures nothing."""
    cube = np.asarray(cube, np.float64)
    start_ou = np.asarray(start_ou, np.float64)
    goal_ou = np.asarray(goal_ou, np.float64)
    d_seg = point_seg_dist(cube, start_ou, goal_ou)
    used_halfwidth = float(halfwidth)
    for _ in range(3):                      # widen if the corridor is empty
        cand = np.nonzero(d_seg <= used_halfwidth)[0]
        keep = ((np.linalg.norm(cube[cand] - goal_ou, axis=-1) > avoid_r + density_eps)
                & (np.linalg.norm(cube[cand] - start_ou, axis=-1) > avoid_r))
        cand = cand[keep]
        if len(cand) > 0:
            break
        used_halfwidth *= 2.0
    assert len(cand) > 0, "empty corridor even after widening"
    if len(cand) > max_cand:
        cand = np.sort(rng.choice(cand, size=max_cand, replace=False))
    samp_rows = rng.choice(len(cube), size=min(samp_n, len(cube)), replace=False)
    dens = support_counts(cube[samp_rows], cube[cand], density_eps)
    j = int(np.argmax(dens))
    return {"avoid_center": cube[cand[j]].tolist(), "row": int(cand[j]),
            "avoid_radius": avoid_r, "density_count": int(dens[j]),
            "density_sample_n": int(len(samp_rows)),
            "seg_dist": float(d_seg[cand[j]]), "halfwidth_used": used_halfwidth,
            "n_candidates": int(len(cand))}


# ======================= Wing B query battery ======================================

def _rep_rows_in_ball(samp_pts, samp_rows, center, eps, rep_k, rng, fallback_row):
    inside = np.nonzero(np.linalg.norm(samp_pts - center, axis=-1) <= eps)[0]
    if len(inside) == 0:
        return [int(fallback_row)]
    pick = rng.choice(inside, size=min(int(rep_k), len(inside)), replace=False)
    return [int(x) for x in np.sort(samp_rows[pick])]


def build_battery(cube, table_z, rng, eps: float = EPS_OBS, n_balls: int = 64,
                  n_elev: int = 8, n_boxes: int = 32, n_unions: int = 8,
                  min_support: int = 50, pool_n: int = 4000, rep_k: int = 8,
                  samp_n: int = 200_000, elev_dz_min: float = 0.5,
                  box_half_lo: float = 0.2, box_half_hi: float = 0.6,
                  union_sep: float = 3 * EPS_OBS):
    """Frozen Wing-B query battery (spec 11 section 2): 64 eps-balls at
    stratified supported dataset positions (>= n_elev elevated), n_boxes random
    axis-aligned boxes (sides 2*[box_half_lo, box_half_hi] = 0.4-1.2 obs units,
    centers inside the dataset cube bounding box, >= 1 dataset point inside),
    n_unions unions-of-2-balls (disjoint, >= union_sep apart). Box/union
    queries freeze `rep_rows` (dataset rows inside the region, <= rep_k) for
    the kernel-average prediction surrogate."""
    cube = np.asarray(cube, np.float64)
    n = len(cube)
    pool_rows = rng.choice(n, size=min(pool_n, n), replace=False)
    sup = support_counts(cube, cube[pool_rows], eps)
    ok = sup >= min_support
    pool_rows = pool_rows[ok]
    pool_pts = cube[pool_rows]
    assert len(pool_rows) >= n_balls, f"supported pool too small ({len(pool_rows)})"
    elev_thresh = float(table_z + elev_dz_min)
    elev_mask = pool_pts[:, 2] >= elev_thresh
    elev_rows = pool_rows[elev_mask]
    flat_rows = pool_rows[~elev_mask]
    if len(elev_rows) >= n_elev:
        ball_rows_e = elev_rows[fps_select(cube[elev_rows], n_elev, rng)]
    else:
        ball_rows_e = elev_rows                       # short: recorded below
    ball_rows_f = flat_rows[fps_select(cube[flat_rows], n_balls - len(ball_rows_e), rng)]
    ball_rows = np.concatenate([ball_rows_f, ball_rows_e])
    balls = [{"type": "ball", "center": cube[r].tolist(), "eps": eps,
              "row": int(r), "elevated": bool(cube[r, 2] >= elev_thresh)}
             for r in ball_rows]

    samp_rows = np.sort(rng.choice(n, size=min(samp_n, n), replace=False))
    samp_pts = cube[samp_rows]
    lo, hi = cube.min(0), cube.max(0)
    boxes, attempts = [], 0
    while len(boxes) < n_boxes and attempts < 5000:
        attempts += 1
        center = rng.uniform(lo, hi)
        half = rng.uniform(box_half_lo, box_half_hi, size=3)
        inside = np.nonzero((np.abs(samp_pts - center) <= half).all(-1))[0]
        if len(inside) == 0:
            continue
        pick = rng.choice(inside, size=min(rep_k, len(inside)), replace=False)
        boxes.append({"type": "box", "center": center.tolist(), "half": half.tolist(),
                      "rep_rows": [int(x) for x in np.sort(samp_rows[pick])],
                      "n_inside_sample": int(len(inside))})
    assert len(boxes) == n_boxes, f"only {len(boxes)} non-empty boxes in {attempts} draws"

    unions, attempts = [], 0
    while len(unions) < n_unions and attempts < 5000:
        attempts += 1
        i, j = rng.choice(len(pool_rows), size=2, replace=False)
        c1, c2 = pool_pts[i], pool_pts[j]
        sep = float(np.linalg.norm(c1 - c2))
        if sep < union_sep:
            continue
        rep1 = _rep_rows_in_ball(samp_pts, samp_rows, c1, eps, rep_k, rng, pool_rows[i])
        rep2 = _rep_rows_in_ball(samp_pts, samp_rows, c2, eps, rep_k, rng, pool_rows[j])
        unions.append({"type": "union2", "centers": [c1.tolist(), c2.tolist()],
                       "eps": eps, "rows": [int(pool_rows[i]), int(pool_rows[j])],
                       "rep_rows": [rep1, rep2], "sep": sep})
    assert len(unions) == n_unions, f"only {len(unions)} unions in {attempts} draws"

    return {"balls": balls, "boxes": boxes, "unions": unions,
            "eps": eps, "elev_thresh": elev_thresh,
            "min_support": int(min_support), "rep_k": int(rep_k),
            "n_elevated": int(sum(b["elevated"] for b in balls))}


# ======================= artifact assembly =========================================

def build_artifact(cube, spans, goal_ou: dict, seed: int, gamma: float = GAMMA,
                   eps: float = EPS_OBS, lambdas=(1, 5), t3_kwargs=None,
                   t4_kwargs=None, t5_kwargs=None, battery_kwargs=None) -> dict:
    """Assemble the full artifact dict from dataset arrays + native goal xyz
    (obs units). Deterministic given (cube, spans, goal_ou, seed): one rng
    drives every stochastic choice in a fixed order. The *_kwargs passthroughs
    exist for the synthetic-data unit tests; production uses the defaults."""
    rng = np.random.default_rng(seed)
    cube = np.asarray(cube, np.float64)
    table_z = table_z_mode(cube[:, 2])
    start_rows = [a for a, _b in spans]
    start_ou = np.median(cube[start_rows], axis=0)

    t3_pairs, t3_meta = select_t3_pairs(cube, spans, table_z, rng, gamma, eps,
                                        **(t3_kwargs or {}))
    t4_tasks = select_union3(cube, table_z, rng, eps, **(t4_kwargs or {}))
    t5_tasks = []
    for i in sorted(goal_ou):
        rec = select_corridor(cube, start_ou, np.asarray(goal_ou[i]), rng,
                              **(t5_kwargs or {}))
        rec.update({"task": int(i), "goal_center": list(map(float, goal_ou[i])),
                    "start_center": start_ou.tolist(), "lambdas": list(lambdas)})
        t5_tasks.append(rec)
    battery = build_battery(cube, table_z, rng, eps, **(battery_kwargs or {}))

    return {
        "version": TASK_JSON_VERSION,
        "spec": SPEC,
        "seed": int(seed),
        "env": "cube-single-play-v0",
        "env_geometry": {
            "units": "obs (1 = 10 cm); cube xyz = obs[19:22]",
            "cube_obs_slice": [19, 22], "eps": eps, "gamma": gamma,
            "table_z": table_z, "elev_dz": ELEV_DZ, "avoid_r": AVOID_R,
        },
        "dataset": {
            "n_transitions": int(len(cube)), "n_episodes": int(len(spans)),
            "start_median_ou": start_ou.tolist(),
            "goal_ou": {f"t{i}": list(map(float, goal_ou[i])) for i in sorted(goal_ou)},
        },
        "t3_occhoice": {"pairs": t3_pairs, "selection_meta": t3_meta},
        "t4_union3": {"tasks": t4_tasks},
        "t5_corridor": {"tasks": t5_tasks, "lambdas": list(lambdas)},
        "wingb_battery": battery,
    }


def artifact_text(art: dict) -> str:
    """Canonical serialization (sort_keys => --check is byte-exact)."""
    return json.dumps(to_native(art), indent=2, sort_keys=True) + "\n"


# ======================= dataset / env plumbing ====================================

def get_native_goal_ou() -> dict:
    """Native task goal cube xyz in OBS units, obtained exactly the way
    eval_zeroshot_rewards.py does (env.unwrapped.cur_task_info['goal_xyzs'])."""
    import ogbench
    goal_ou = {}
    for i in range(1, 6):
        env = ogbench.make_env_and_datasets(EVAL_TASKS[i], env_only=True)
        env.reset(seed=0)
        gx = np.asarray(env.unwrapped.cur_task_info["goal_xyzs"]).reshape(-1)[:3]
        goal_ou[i] = cube_m_to_ou(gx)
        env.close()
    return goal_ou


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-path", default="~/.ogbench/data",
                    help="ogbench dataset_dir containing cube-single-play-v0.npz")
    ap.add_argument("--n-transitions", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "analysis" / "rescuing-fb"
                                         / "smb_cube_single_v1.json"))
    ap.add_argument("--check", action="store_true",
                    help="regenerate and verify byte-identity against --out "
                         "(the artifact is frozen; a mismatch is an error)")
    args = ap.parse_args()

    t0 = time.time()
    from renewal_head import load_dataset
    obs, _nobs, _acts, term = load_dataset(args.dataset_path, args.n_transitions)
    cube = obs[:, CUBE_OBS].astype(np.float64)
    spans = split_episodes(term)
    print(f"[smb_tasks] dataset: {len(obs)} transitions, {len(spans)} episodes "
          f"({time.time()-t0:.0f}s)", flush=True)
    goal_ou = get_native_goal_ou()
    print(f"[smb_tasks] native goals (obs units): "
          f"{ {i: np.round(g, 3).tolist() for i, g in goal_ou.items()} }", flush=True)

    art = build_artifact(cube, spans, goal_ou, args.seed)
    text = artifact_text(art)
    outp = Path(args.out)

    if args.check:
        assert outp.exists(), f"--check: {outp} does not exist"
        old = outp.read_text()
        if old == text:
            print(f"[smb_tasks] CHECK OK: {outp} regenerates byte-identically "
                  f"({time.time()-t0:.0f}s)", flush=True)
            return 0
        print(f"[smb_tasks] CHECK FAILED: regeneration differs from {outp} "
              f"(len {len(text)} vs {len(old)})", flush=True)
        return 1

    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(text)
    for p in art["t3_occhoice"]["pairs"]:
        print(f"[smb_tasks][t3] {p['name']}: R_S={p['stable']['R_mc']:.2f} "
              f"(sup {p['stable']['support']}) R_T={p['transient']['R_mc']:.2f} "
              f"(sup {p['transient']['support']}) ratio={p['R_ratio']:.1f} "
              f"w_T={p['w_T']:.2f} validation={p['validation']}", flush=True)
    b = art["wingb_battery"]
    print(f"[smb_tasks] battery: {len(b['balls'])} balls "
          f"({b['n_elevated']} elevated), {len(b['boxes'])} boxes, "
          f"{len(b['unions'])} unions", flush=True)
    print(f"[smb_tasks] DONE ({time.time()-t0:.0f}s) -> {outp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
