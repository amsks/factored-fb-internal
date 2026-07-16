"""The bridge table at the PAPER's protocol: cube-single state, 3 seeds (0,1,2), @500k.

results/bridge_grafts.csv averages over EVERY seed a group ever ran (n=10/12/13 for the old
campaigns), which is not what PAPER/fb-gciql-bridge/fb-gciql-bridge.tex reports
("Every combination we ran, cube-single state, 3 seeds @ 500k", tab:master). This script
rebuilds the same cells restricted to seeds {0,1,2} so every row is protocol-identical.

Extraction is reused verbatim from compile_results_csvs (success at the 500k eval point;
runs that trained to 1M/1.5M are read from wandb history at step<=510k), so the only thing
that changes versus the living CSV is WHICH SEEDS are averaged.

OneStepFB is excluded by request. antmaze rows are excluded (paper table is cube-single).

Usage:  .venv/bin/python scripts/bridge_table_500k.py [--csv] [--seeds 0,1,2]
"""
import argparse
import collections
import csv as _csv
import sys

sys.path.insert(0, '/home/adityamohan/git/Factored-FB/scripts')

from compile_results_csvs import ENTITY_PROJECT, ci95, hist_value_at_500k  # noqa: E402
from results_map import MAP  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--seeds', default='0,1,2')
p.add_argument('--csv', action='store_true')
a = p.parse_args()
WANT = {int(s) for s in a.seeds.split(',')}

import wandb  # noqa: E402

api = wandb.Api()

# every bridge cell, cube only, OneStepFB dropped
CELLS = {g: v for g, v in MAP.items()
         if v.get('csv') == 'bridge' and v.get('env') == 'cube' and v.get('agent') != 'OneStepFB'}

# the paper's three anchors (tab:master): IQL-bilinear ~90, GCIQL-MLP ~73, plain FB 34 @500k.
# Pulled through the SAME extraction so the protocol is self-validating.
ANCHORS = {'GCIQL-State-Flow-Bilinear-d50': 'ANCHOR IQL-bilinear d50',
           'GCIQL-State-Flow': 'ANCHOR GCIQL MLP critic',
           'FB-State-Flow': 'ANCHOR plain FB (measure)'}
for _g, _lab in ANCHORS.items():
    _m = MAP.get(_g, {})
    CELLS[_g] = dict(_m, csv='bridge', env='cube', variant=_lab,
                     graft='paper anchor — protocol check')

runs_by_group = collections.defaultdict(list)
for r in api.runs(ENTITY_PROJECT, per_page=500):
    if r.group in CELLS:
        runs_by_group[r.group].append(r)

def seed_of(r):
    s = r.config.get('seed')
    if s is None and isinstance(r.config.get('agent'), dict):
        s = r.config['agent'].get('seed')
    return s


rows = []
dupe_note = []
for g, meta in CELLS.items():
    runs = runs_by_group.get(g, [])
    # A group can hold SEVERAL runs at the same seed (a native-500k campaign and a later
    # 1M campaign both used seeds 0/1/2). Averaging both double-counts that seed. Keep ONE
    # run per seed, preferring the run that natively trained to 500k (the paper's campaign),
    # then the most recent.
    by_seed = collections.defaultdict(list)
    dropped = 0
    for r in runs:
        s = seed_of(r)
        if s not in WANT:
            dropped += 1
            continue
        by_seed[s].append(r)
    n_dupe = sum(len(v) - 1 for v in by_seed.values())
    if n_dupe:
        dupe_note.append((meta.get('variant', g), n_dupe))

    kept = []
    for s, rs in by_seed.items():
        rs.sort(key=lambda r: (r.config.get('train_steps') != 500_000,
                               -(r.summary.get('_timestamp') or 0)))
        r = rs[0]
        metric = ('evaluation/overall_success'
                  if r.summary.get('evaluation/overall_success') is not None
                  or 'GCIQL' in g or 'ISFB-State' in g else 'success/overall')
        v, step = hist_value_at_500k(r.id, metric)
        if v is not None:
            kept.append((s, float(v)))
    kept.sort()
    vals = [v for _, v in kept]
    rows.append(dict(
        group=g, agent=meta.get('agent'), actor=meta.get('actor'),
        variant=meta.get('variant', ''), graft=meta.get('graft', ''),
        n=len(vals), seeds=[s for s, _ in kept],
        mean=(sum(vals) / len(vals)) if vals else None,
        ci95=ci95(vals) if len(vals) > 1 else 0.0,
        per_seed=' '.join(f'{v:.3f}' for v in vals),
        n_all=len(runs), n_dropped=dropped))

rows.sort(key=lambda r: -(r['mean'] if r['mean'] is not None else -1))

if a.csv:
    out = '/home/adityamohan/git/Factored-FB/results/bridge_grafts_3seed_500k.csv'
    with open(out, 'w', newline='') as fh:
        w = _csv.writer(fh)
        w.writerow(['agent', 'actor', 'variant', 'experiment', 'mean_500k', 'ci95', 'n_seeds',
                    'seeds', 'per_seed', 'n_runs_in_group', 'n_seeds_dropped', 'wandb_group'])
        for r in rows:
            w.writerow([r['agent'], r['actor'], r['variant'], r['graft'],
                        '' if r['mean'] is None else f"{r['mean']:.3f}", f"{r['ci95']:.3f}",
                        r['n'], ' '.join(map(str, r['seeds'])), r['per_seed'],
                        r['n_all'], r['n_dropped'], r['group']])
    print(f'[csv] {out}: {len(rows)} rows')
    sys.exit(0)

print(f'BRIDGE TABLE — cube-single state, seeds {sorted(WANT)}, @500k  (paper protocol)')
print(f'{len(rows)} cells; OneStepFB and antmaze excluded\n')
print(f'{"agent":<6}{"variant":<30}{"mean":>7}{"ci95":>7}{"n":>3}  {"per-seed":<22} experiment')
print('-' * 128)
for r in rows:
    mv = '  n/a ' if r['mean'] is None else f"{r['mean']:6.3f}"
    print(f"{r['agent']:<6}{r['variant'][:29]:<30}{mv}{r['ci95']:7.3f}{r['n']:>3}  "
          f"{r['per_seed']:<22} {r['graft'][:44]}")

short = [r for r in rows if r['mean'] is not None and r['n'] < len(WANT)]
none = [r for r in rows if r['mean'] is None]
print(f"\n{len(rows) - len(none)} cells with data; {len(none)} with NO run at seeds {sorted(WANT)} @500k.")
if short:
    print(f"{len(short)} cells have FEWER than {len(WANT)} seeds:")
    for r in short:
        print(f"   {r['variant']:<30} n={r['n']} seeds={r['seeds']}")
if none:
    print("cells with no usable seed:", ', '.join(r['variant'] for r in none))
