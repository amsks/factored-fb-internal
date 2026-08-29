#!/usr/bin/env python
"""compile_results_csvs.py — regenerate the two living results CSVs from wandb.

  results/summary_agents.csv : canonical agents x envs x actors (uniform @500k protocol)
  results/bridge_grafts.csv  : every FB<->GCIQL bridge graft/ablation cell
  results/compile_audit.txt  : backward audit — every wandb group accounted for
                               (mapped / excluded-with-reason / UNMAPPED = loud failure),
                               per-run extraction status, partial seeds, protocol flags.

Protocol: per-run success at the 500k evaluation point. If a run trained past 500k
(1M/1.5M campaigns), the value is extracted from wandb history at step<=510k (cached in
analysis/results_cache/); runs that died before 490k are listed as partial and EXCLUDED
from the mean. mean +- ci95 (1.96*sd/sqrt(n)) over the surviving seeds.

Both CSVs additionally carry the LEARNING CURVE: mean_{100,200,300,400}k / n_*k / per_seed_*k,
always history-sourced (the wandb summary only holds the last eval). These are collected even
for runs dropped as partial @500k -- a run killed at 333k still has a citable 100k-300k curve.
per_seed_*k is ordered by run name (== seed order), so a fresh run can be compared seed-by-seed
against an existing baseline at the same step.

Metrics: jax agents log evaluation/overall_success; torch FB-family logs success/overall.

Update discipline: after any new campaign, add its group to scripts/results_map.py and
rerun:  .venv/bin/python scripts/compile_results_csvs.py            (uses cached dump)
        .venv/bin/python scripts/compile_results_csvs.py --refresh  (refetch wandb)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
from results_map import MAP  # noqa: E402

ENTITY_PROJECT = 'amsks/factored-fb'
DUMP = REPO / 'analysis' / 'results_cache' / 'wandb_dump.json'
HIST_CACHE = REPO / 'analysis' / 'results_cache' / 'hist'
OUT_SUMMARY = REPO / 'results' / 'summary_agents.csv'
OUT_BRIDGE = REPO / 'results' / 'bridge_grafts.csv'
OUT_AUDIT = REPO / 'results' / 'compile_audit.txt'
STEP_LO, STEP_HI = 490_000, 510_000
# Intermediate eval points, reported alongside the canonical @500k figure so a cell's
# LEARNING CURVE (not just its endpoint) is citable -- and so a freshly launched run can be
# sanity-checked against an existing baseline at the same step, same seed. Runs eval every
# 100k; TOL absorbs the off-by-a-few-steps of the logged eval point.
CKPTS = (100_000, 200_000, 300_000, 400_000)
CKPT_TOL = 10_000


def fetch_dump():
    import wandb
    api = wandb.Api(timeout=120)
    dump = []
    for r in api.runs(ENTITY_PROJECT, per_page=500):
        s = r.summary
        dump.append({'id': r.id, 'name': r.name, 'group': r.group, 'state': r.state,
                     'overall_success': s.get('evaluation/overall_success'),
                     'success_overall': s.get('success/overall'),
                     'step': s.get('step') or s.get('_step')})
    DUMP.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dump, open(DUMP, 'w'))
    return dump


def hist_series(run_id, metric):
    """Full [[step, value]] eval history of `metric` (sampled history(); eval series are <=100
    points so samples=2000 loses nothing). _step is the training step in both stacks. jax logs
    the overall metric directly; torch logs only per-task eval/reward/<task>/success rows —
    their mean IS success/overall (verified exact on fb_state_plain). Cached per run id.

    The cache is keyed by run id ALONE, so it is only written for runs that have STOPPED
    (finished/crashed/failed). Caching a still-training run would freeze it at whatever partial
    history it had when first fetched, and every later compile would silently report that stale
    curve -- a run that has since reached 500k would keep showing up as a partial. `--refresh`
    refetches the dump but not these; `--refresh-hist` drops them all."""
    HIST_CACHE.mkdir(parents=True, exist_ok=True)
    cf = HIST_CACHE / f'{run_id}.json'
    if cf.exists():
        return json.load(open(cf))
    import wandb
    api = wandb.Api(timeout=120)
    run = api.run(f'{ENTITY_PROJECT}/{run_id}')
    if metric == 'success/overall':
        keys = [k for k in run.summary.keys()
                if k.startswith('eval/reward/') and k.endswith('/success')
                and not k.endswith('/eval/success')]
        pts = []
        if keys:
            for row in run.history(keys=keys, samples=2000, pandas=False):
                vals = [row[k] for k in keys if row.get(k) is not None]
                if vals:
                    pts.append([row['_step'], sum(vals) / len(vals)])
    else:
        pts = [[row['_step'], row[metric]]
               for row in run.history(keys=[metric], samples=2000, pandas=False)
               if row.get(metric) is not None]
    if run.state != 'running':          # never freeze a live run's partial curve into the cache
        json.dump(pts, open(cf, 'w'))
    return pts


def hist_value_at_500k(run_id, metric):
    """Value at the eval step closest to (and <=) 510k. None if no logged point in [1, 510k]
    (e.g. resumed runs whose history starts >500k)."""
    below = [(s, v) for s, v in hist_series(run_id, metric) if 1 <= s <= STEP_HI]
    if not below:
        return None, None
    s, v = max(below, key=lambda t: t[0])
    return v, s


def hist_value_at(run_id, metric, target, tol=CKPT_TOL):
    """Value at the logged eval point NEAREST `target` within +-tol. Returns (None, None) if the
    run never logged an eval there (died early, or eval_interval doesn't land on it)."""
    near = [(s, v) for s, v in hist_series(run_id, metric) if abs(s - target) <= tol]
    if not near:
        return None, None
    s, v = min(near, key=lambda t: abs(t[0] - target))
    return v, s


def ci95(vals):
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return 1.96 * sd / math.sqrt(len(vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--refresh', action='store_true', help='refetch the wandb run dump')
    ap.add_argument('--refresh-hist', action='store_true',
                    help='also drop the per-run history cache (needed for runs still training)')
    args = ap.parse_args()

    if args.refresh_hist and HIST_CACHE.exists():
        n = 0
        for f in HIST_CACHE.glob('*.json'):
            f.unlink(); n += 1
        print(f'[compile] dropped {n} cached run histories')

    dump = fetch_dump() if args.refresh or not DUMP.exists() else json.load(open(DUMP))
    by_group = {}
    for d in dump:
        by_group.setdefault(d['group'] or 'NONE', []).append(d)

    unmapped = sorted(set(by_group) - set(MAP))
    missing = sorted(set(k for k, v in MAP.items() if v['csv'] != 'exclude') - set(by_group))
    audit = []
    rows = {'summary': [], 'bridge': []}

    for grp, spec in sorted(MAP.items()):
        if spec['csv'] == 'exclude':
            audit.append(f"EXCLUDED {grp}: {spec['reason']} (n={len(by_group.get(grp, []))})")
            continue
        runs = by_group.get(grp, [])
        dropped = [r for r in runs if (r['name'] or '') in spec.get('exclude_runs', ())]
        if dropped:
            runs = [r for r in runs if r not in dropped]
            audit.append(f"DROPPED-RUNS {grp}: {','.join(r['name'] for r in dropped)} "
                         f"({spec.get('exclude_runs_reason', 'per-run exclusion in results_map')})")
        metric = 'overall_success' if any(r['overall_success'] is not None for r in runs) else 'success_overall'
        wkey = 'evaluation/overall_success' if metric == 'overall_success' else 'success/overall'
        vals, seeds, partial = [], [], []
        ck_vals = {t: [] for t in CKPTS}      # intermediate learning-curve points
        ck_seeds = {t: [] for t in CKPTS}
        for r in sorted(runs, key=lambda r: r['name'] or ''):   # stable per-seed column order
            v, step = r[metric], r['step']
            if v is None and r['overall_success'] is None and r['success_overall'] is None:
                partial.append(f"{r['name']}(no-metric)")
                continue
            def seed_label(val, s):
                # step marker only when the extracted point is NOT the nominal 500k
                return f"{val:.3f}" if s == 500_000 else f"{val:.3f}@{s}"
            if step is not None and STEP_LO <= step <= STEP_HI:
                vals.append(v); seeds.append(seed_label(v, step))
            elif step is not None and step > STEP_HI:
                hv, hs = hist_value_at_500k(r['id'], wkey)
                if hv is not None:
                    vals.append(hv); seeds.append(seed_label(hv, hs))
                else:
                    partial.append(f"{r['name']}(no-500k-point)")
            else:  # died early or step unknown -> try history anyway
                hv, hs = hist_value_at_500k(r['id'], wkey)
                if hv is not None and hs >= STEP_LO:
                    vals.append(hv); seeds.append(seed_label(hv, hs))
                else:
                    partial.append(f"{r['name']}(died@{step})")
            # Intermediate points are ALWAYS history-sourced (summary only holds the last eval),
            # and are collected even for runs dropped as partial @500k -- a run killed at 333k
            # still has a valid, citable 100k/200k/300k curve.
            for t in CKPTS:
                cv, cs = hist_value_at(r['id'], wkey, t)
                if cv is not None:
                    ck_vals[t].append(cv)
                    ck_seeds[t].append(f"{cv:.3f}" if cs == t else f"{cv:.3f}@{cs}")
        m = round(sum(vals) / len(vals), 4) if vals else None
        c = round(ci95(vals), 4) if vals else None
        row = {
            'env': spec['env'], 'agent': spec['agent'], 'actor': spec['actor'],
            'variant': spec.get('variant', ''), 'graft': spec.get('graft', ''),
            'mean': m, 'ci95': c, 'n_seeds': len(vals),
            'per_seed': ' '.join(seeds), 'partial_seeds': ' '.join(partial),
            'wandb_group': grp, 'notes': spec.get('notes', ''),
        }
        for t in CKPTS:
            k = t // 1000
            cvs = ck_vals[t]
            row[f'mean_{k}k'] = round(sum(cvs) / len(cvs), 4) if cvs else None
            row[f'n_{k}k'] = len(cvs)
            row[f'per_seed_{k}k'] = ' '.join(ck_seeds[t])
        rows[spec['csv']].append(row)
        if partial:
            audit.append(f"PARTIAL {grp}: {len(vals)} used, dropped: {', '.join(partial)}")

    ck_cols = [c for t in CKPTS for c in (f'mean_{t // 1000}k', f'n_{t // 1000}k', f'per_seed_{t // 1000}k')]
    order = {'cube': 0, 'scene': 1, 'antmaze': 2, 'cube-pixel': 3}
    for key, out, cols in (
        ('summary', OUT_SUMMARY, ['env', 'agent', 'actor', 'variant', 'mean', 'ci95', 'n_seeds', 'per_seed'] + ck_cols + ['partial_seeds', 'wandb_group', 'notes']),
        ('bridge', OUT_BRIDGE, ['env', 'agent', 'actor', 'variant', 'graft', 'mean', 'ci95', 'n_seeds', 'per_seed'] + ck_cols + ['partial_seeds', 'wandb_group', 'notes']),
    ):
        rs = sorted(rows[key], key=lambda r: (order.get(r['env'], 9), r['agent'], r['actor'], r['variant']))
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader(); w.writerows(rs)
        print(f'[compile] {out}: {len(rs)} rows')

    audit.insert(0, f'groups in wandb: {len(by_group)} | mapped: '
                    f'{sum(1 for v in MAP.values() if v["csv"] != "exclude")} | '
                    f'excluded: {sum(1 for v in MAP.values() if v["csv"] == "exclude")}')
    if unmapped:
        audit.insert(1, 'UNMAPPED GROUPS (add to results_map.py!): ' + ', '.join(unmapped))
    if missing:
        audit.insert(1, 'MAPPED BUT ABSENT FROM WANDB: ' + ', '.join(missing))
    OUT_AUDIT.write_text('\n'.join(audit) + '\n')
    print(f'[compile] audit -> {OUT_AUDIT}' + (f'  !! {len(unmapped)} UNMAPPED' if unmapped else '  (all groups accounted for)'))
    return 1 if unmapped else 0


if __name__ == '__main__':
    sys.exit(main())
