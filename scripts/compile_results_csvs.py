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


def hist_value_at_500k(run_id, metric):
    """Value of `metric` at the eval step closest to (and <=) 510k, from run history
    (sampled history(); eval series are <=100 points so samples=2000 loses nothing).
    _step is the training step in both stacks. jax logs the overall metric directly;
    torch logs only per-task eval/reward/<task>/success rows — their mean IS
    success/overall (verified exact on fb_state_plain). Cached per run id.
    None if no logged point in [1, 510k] (e.g. resumed runs whose history starts >500k)."""
    HIST_CACHE.mkdir(parents=True, exist_ok=True)
    cf = HIST_CACHE / f'{run_id}.json'
    if cf.exists():
        pts = json.load(open(cf))
    else:
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
        json.dump(pts, open(cf, 'w'))
    below = [(s, v) for s, v in pts if 1 <= s <= STEP_HI]
    if not below:
        return None, None
    s, v = max(below, key=lambda t: t[0])
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
    args = ap.parse_args()

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
        metric = 'overall_success' if any(r['overall_success'] is not None for r in runs) else 'success_overall'
        wkey = 'evaluation/overall_success' if metric == 'overall_success' else 'success/overall'
        vals, seeds, partial = [], [], []
        for r in runs:
            v, step = r[metric], r['step']
            if v is None and r['overall_success'] is None and r['success_overall'] is None:
                partial.append(f"{r['name']}(no-metric)")
                continue
            if step is not None and STEP_LO <= step <= STEP_HI:
                vals.append(v); seeds.append(f"{v:.3f}@{step}")
            elif step is not None and step > STEP_HI:
                hv, hs = hist_value_at_500k(r['id'], wkey)
                if hv is not None:
                    vals.append(hv); seeds.append(f"{hv:.3f}@{hs}")
                else:
                    partial.append(f"{r['name']}(no-500k-point)")
            else:  # died early or step unknown -> try history anyway
                hv, hs = hist_value_at_500k(r['id'], wkey)
                if hv is not None and hs >= STEP_LO:
                    vals.append(hv); seeds.append(f"{hv:.3f}@{hs}")
                else:
                    partial.append(f"{r['name']}(died@{step})")
        m = round(sum(vals) / len(vals), 4) if vals else None
        c = round(ci95(vals), 4) if vals else None
        rows[spec['csv']].append({
            'env': spec['env'], 'agent': spec['agent'], 'actor': spec['actor'],
            'variant': spec.get('variant', ''), 'graft': spec.get('graft', ''),
            'mean': m, 'ci95': c, 'n_seeds': len(vals),
            'per_seed': ' '.join(seeds), 'partial_seeds': ' '.join(partial),
            'wandb_group': grp, 'notes': spec.get('notes', ''),
        })
        if partial:
            audit.append(f"PARTIAL {grp}: {len(vals)} used, dropped: {', '.join(partial)}")

    order = {'cube': 0, 'scene': 1, 'antmaze': 2, 'cube-pixel': 3}
    for key, out, cols in (
        ('summary', OUT_SUMMARY, ['env', 'agent', 'actor', 'variant', 'mean', 'ci95', 'n_seeds', 'per_seed', 'partial_seeds', 'wandb_group', 'notes']),
        ('bridge', OUT_BRIDGE, ['env', 'agent', 'actor', 'variant', 'graft', 'mean', 'ci95', 'n_seeds', 'per_seed', 'partial_seeds', 'wandb_group', 'notes']),
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
